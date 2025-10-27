import os, math
import numpy as np
from math import cos, sin, pi
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ---------- small helpers ----------

def _ensure_gif_filename(path):
    root, ext = os.path.splitext(path)
    return f"{path}.gif" if ext == "" else path

def _finite(*vals):
    return all(math.isfinite(v) for v in vals)

def _norm_angle(theta):
    return (theta + pi) % (2*pi) - pi

def _pixels_per_data_y(ax):
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = ax.transData.transform((0, 1))
    return abs(y1 - y0)

def _world_to_px(x, y, scale, H):
    # world origin bottom-left -> pixel origin top-left
    return int(round(x*scale)), int(round(H - y*scale))

def _bbox_outside(px_bbox, img_bbox, pad_px):
    ix0, iy0, ix1, iy1 = img_bbox
    x0, y0, x1, y1 = px_bbox
    return (x1 < ix0 - pad_px or x0 > ix1 + pad_px or
            y1 < iy0 - pad_px or y0 > iy1 + pad_px)


# ---- generic plot-layer merging ----

def _infer_plot_type(o):
    # Prefer explicit 'type' if present and valid
    t = o.get('type')
    if t in ('circle', 'segment'):
        return t
    # Infer from fields
    if 'radius' in o and 'length' in o:
        return 'segment'
    if 'radius' in o and 'length' not in o:
        return 'circle'
    return None

def _is_plot_entity(o):
    if not isinstance(o, dict):
        return False
    loc = o.get('location')
    if not (isinstance(loc, (tuple, list)) and len(loc) == 2):
        return False
    return _infer_plot_type(o) is not None

def _is_entity_map(v):
    # A plotting layer is a dict whose values are plot-entities (allow empty)
    return isinstance(v, dict) and all(_is_plot_entity(x) for x in v.values())

def merge_plot_layers(data, merged_key='agents'):
    """
    For each frame (dict), find any dict-of-entities and merge them under `merged_key`.
    If `merged_key` exists already, it will be used as the base and augmented.
    ID collisions are resolved by prefixing with the original source key.
    """
    merged_frames = []
    for step in data:
        # Shallow copy step so we don't mutate caller data
        step_out = dict(step)
        base = dict(step_out.get(merged_key, {}))
        # Find all entity maps in this frame
        entity_sources = []
        for k, v in step.items():
            if k == merged_key:
                continue
            if _is_entity_map(v):
                entity_sources.append((k, v))

        # If nothing to merge, keep the frame as-is
        if not entity_sources:
            merged_frames.append(step_out)
            continue

        # Merge, resolving collisions by prefixing with source key
        existing_ids = set(base.keys())
        for src_key, src_map in entity_sources:
            for ent_id, ent in src_map.items():
                out_id = ent_id
                if out_id in existing_ids:
                    out_id = f"{src_key}:{ent_id}"
                # Ensure type is present (gif code may rely on it)
                if 'type' not in ent or ent['type'] not in ('circle', 'segment'):
                    t = _infer_plot_type(ent)
                    if t is not None:
                        ent = {**ent, 'type': t}
                base[out_id] = ent
                existing_ids.add(out_id)

        # Write merged layer and drop the sources we merged
        step_out[merged_key] = base
        for src_key, _ in entity_sources:
            if src_key != merged_key:
                step_out.pop(src_key, None)

        merged_frames.append(step_out)

    return merged_frames


# ---------- main renderer ----------


def simulation_to_gif(
    data,
    config,
    agents_key='agents',
    filename='simulation.gif',
    out_dir='out',
    skip_frames=1,
    figure_size_inches=(6, 6),
    dpi=90,
    show_time_title=False,
    # culling/clamping controls:
    world_pad=50.0,       # extra world-units beyond env_size to still draw
    max_line_px=40,       # max segment diameter (2*radius) in pixels
    max_radius_px=40,     # max circle radius in pixels
):
    """
    Fast Matplotlib renderer using:
      - culling (off-screen/non-finite)
      - pixel-size clamping
      - angle normalization
      - artist pooling (no add/remove)
      - background blitting (no full redraw per frame)
    """
    # ---- merge any plot-capable layers into one
    if isinstance(data, (list, tuple)) and data:
        data = merge_plot_layers(data, merged_key=agents_key)
    # ---- paths
    os.makedirs(out_dir, exist_ok=True)
    filename = _ensure_gif_filename(filename)
    out_path = filename if os.path.dirname(filename) else os.path.join(out_dir, filename)
    out_path = os.path.abspath(os.path.expanduser(out_path))

    # ---- config / frames
    env_size = float(config['env_size'])
    barriers = config.get('barriers', [])
    frames = list(data)[::skip_frames]
    if not frames:
        raise ValueError("No frames to render.")

    # ---- figure/canvas once
    fig = plt.figure(figsize=figure_size_inches, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, env_size)
    ax.set_ylim(0, env_size)
    ax.set_axis_off()
    ax.set_autoscale_on(False)

    # Prepare transforms to compute pixel scaling
    canvas.draw()
    H = fig.canvas.get_width_height()[1]
    ypix_per_unit = _pixels_per_data_y(ax)

    # World->pixel scale (assumes square world)
    # For square axes with equal aspect, 1 world-unit maps to ypix_per_unit pixels vertically.
    # We'll use that for thickness clamping and barrier widths.
    img_bbox = (0, 0, fig.canvas.get_width_height()[0]-1, H-1)
    pad_px = int(round(world_pad * ypix_per_unit))

    # ---- static barriers once (pixel linewidth, clamped)
    barrier_artists = []
    for b in barriers:
        (sx, sy), (ex, ey) = b['start'], b['end']
        if not _finite(sx, sy, ex, ey):
            continue
        t_world = float(b.get('thickness', 1.0))
        lw_px = max(1, min(int(round(t_world * ypix_per_unit)), max_line_px))
        art = Line2D([sx, ex], [sy, ey], linewidth=lw_px, color='gray', antialiased=False)
        ax.add_line(art)
        barrier_artists.append(art)

    # ---- pre-scan to size pools (counts only)
    max_circles = 0
    max_segments = 0
    for step in frames:
        vals = list(step[agents_key].values())
        max_circles  = max(max_circles,  sum(1 for o in vals if o.get('type') == 'circle'))
        max_segments = max(max_segments, sum(1 for o in vals if o.get('type') == 'segment'))

    # ---- build artist pools (hidden initially)
    circle_pool = []
    for _ in range(max_circles):
        c = Circle((0, 0), 1.0, fill=True, antialiased=False, visible=False)
        ax.add_patch(c)
        circle_pool.append(c)

    segment_pool = []
    for _ in range(max_segments):
        ln = Line2D([0, 0], [0, 0], linewidth=1.0, solid_capstyle='round',
                    antialiased=False, visible=False)
        ax.add_line(ln)
        segment_pool.append(ln)

    title_obj = ax.set_title("") if show_time_title else None

    # ---- cache background for blit (after adding static items)
    canvas.draw()
    background = canvas.copy_from_bbox(ax.bbox)

    # ---- helpers to update pools with culling/clamping
    def update_from_step(step):
        # returns visible counts for circles, segments
        c_vis = 0
        s_vis = 0

        # Circles
        for o in (x for x in step[agents_key].values() if x.get('type') == 'circle'):
            cx, cy = o['location']
            r_world = float(o['radius'])
            if not _finite(cx, cy, r_world) or r_world <= 0:
                continue
            # pixel radius (clamped)
            rpx_nom = int(round(r_world * ypix_per_unit))
            rpx = max(1, min(rpx_nom, max_radius_px))

            # pixel bbox for culling
            # Map to pixel space via Axes transform (faster to approximate using world->pixel scale vertically)
            # We'll compute bbox in pixel coordinates using transform + clamp
            px0, py0 = ax.transData.transform((cx - r_world, cy - r_world))
            px1, py1 = ax.transData.transform((cx + r_world, cy + r_world))
            bbox_px = (int(px0), int(py0), int(px1), int(py1))
            if _bbox_outside(bbox_px, img_bbox, pad_px):
                continue

            # update pooled artist
            art = circle_pool[c_vis]
            art.center = (cx, cy)
            art.set_radius(r_world)  # keep radius in data coords
            art.set_visible(True)
            c_vis += 1

            if c_vis >= len(circle_pool):
                break

        # hide unused circles
        for i in range(c_vis, len(circle_pool)):
            if circle_pool[i].get_visible():
                circle_pool[i].set_visible(False)

        # Segments
        for o in (x for x in step[agents_key].values() if x.get('type') == 'segment'):
            L = float(o['length'])
            r_world = float(o['radius'])
            ang = _norm_angle(float(o['angle']))
            cx, cy = o['location']
            if not _finite(cx, cy, L, r_world, ang) or L <= 0 or r_world <= 0:
                continue

            dx = cos(ang) * (L/2.0)
            dy = sin(ang) * (L/2.0)
            x0w, y0w = cx - dx, cy - dy
            x1w, y1w = cx + dx, cy + dy

            # pixel linewidth (diameter; clamped)
            lw_nom = int(round(2.0 * r_world * ypix_per_unit))
            lw_px = max(1, min(lw_nom, max_line_px))

            # cull via pixel bbox
            p0 = ax.transData.transform((x0w, y0w))
            p1 = ax.transData.transform((x1w, y1w))
            xmin = int(min(p0[0], p1[0]) - lw_px/2)
            xmax = int(max(p0[0], p1[0]) + lw_px/2)
            ymin = int(min(p0[1], p1[1]) - lw_px/2)
            ymax = int(max(p0[1], p1[1]) + lw_px/2)
            if _bbox_outside((xmin, ymin, xmax, ymax), img_bbox, pad_px):
                continue

            # update pooled artist
            art = segment_pool[s_vis]
            art.set_xdata([x0w, x1w])
            art.set_ydata([y0w, y1w])
            art.set_linewidth(lw_px)  # pixel linewidth
            art.set_visible(True)
            s_vis += 1

            if s_vis >= len(segment_pool):
                break

        # hide unused segments
        for i in range(s_vis, len(segment_pool)):
            if segment_pool[i].get_visible():
                segment_pool[i].set_visible(False)

        return c_vis, s_vis

    # ---- render all frames with blit & buffer_rgba
    pil_frames = []
    for step in frames:
        # update artists (with culling/clamping)
        update_from_step(step)
        if show_time_title and title_obj is not None:
            title_obj.set_text(f"t={step.get('time', 0):.1f}")

        # fast redraw
        canvas.restore_region(background)
        # draw static barrier artists (already in background if truly static; harmless to skip)
        # for art in barrier_artists:
        #     ax.draw_artist(art)
        if show_time_title and title_obj is not None:
            ax.draw_artist(title_obj)
        for art in circle_pool:
            if art.get_visible():
                ax.draw_artist(art)
        for art in segment_pool:
            if art.get_visible():
                ax.draw_artist(art)
        canvas.blit(ax.bbox)

        # grab pixels without triggering a redraw
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        rgb = rgba[:, :, :3].copy()
        pil_frames.append(Image.fromarray(rgb))

    plt.close(fig)

    # ---- save GIF with Pillow
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,  # ms per frame
        loop=0,
        optimize=False,
        disposal=2
    )
    print(f"GIF saved to {out_path}")
    return out_path
