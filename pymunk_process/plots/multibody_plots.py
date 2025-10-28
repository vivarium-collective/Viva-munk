import os, math, random
import numpy as np
from math import pi
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import hsv_to_rgb

# ----------------- LineWidthData  -----------------
class LineWidthData(Line2D):
    """
    A Line2D whose linewidth is specified in *data units* (world units),
    not points. We convert to points on-the-fly using the axes transform.
    """
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop('linewidth', 1.0)
        super().__init__(*args, **kwargs)
        self._lw_data = float(_lw_data)

    def _get_lw(self):
        if self.axes is None:
            return 1.0
        x0, y0 = self.axes.transData.transform((0, 0))
        x1, y1 = self.axes.transData.transform((0, 1))
        px_per_data = abs(y1 - y0)
        ppd = 72.0 / self.axes.figure.dpi  # points per pixel
        return max(0.1, px_per_data * self._lw_data * ppd)

    def _set_lw(self, lw):
        self._lw_data = float(lw)

    _linewidth = property(_get_lw, _set_lw)

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

def _bbox_outside(px_bbox, img_bbox, pad_px):
    ix0, iy0, ix1, iy1 = img_bbox
    x0, y0, x1, y1 = px_bbox
    return (x1 < ix0 - pad_px or x0 > ix1 + pad_px or
            y1 < iy0 - pad_px or y0 > iy1 + pad_px)

# ---- generic plot-layer merging ----
def _infer_plot_type(o):
    t = o.get('type')
    if t in ('circle', 'segment'):
        return t
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
    return isinstance(v, dict) and all(_is_plot_entity(x) for x in v.values())

def merge_plot_layers(data, merged_key='agents'):
    merged_frames = []
    for step in data:
        step_out = dict(step)
        base = dict(step_out.get(merged_key, {}))
        entity_sources = []
        for k, v in step.items():
            if k == merged_key:
                continue
            if _is_entity_map(v):
                entity_sources.append((k, v))

        if not entity_sources:
            merged_frames.append(step_out)
            continue

        existing_ids = set(base.keys())
        for src_key, src_map in entity_sources:
            for ent_id, ent in src_map.items():
                out_id = ent_id if ent_id not in existing_ids else f"{src_key}:{ent_id}"
                if 'type' not in ent or ent['type'] not in ('circle', 'segment'):
                    t = _infer_plot_type(ent)
                    if t is not None:
                        ent = {**ent, 'type': t}
                base[out_id] = ent
                existing_ids.add(out_id)

        step_out[merged_key] = base
        for src_key, _ in entity_sources:
            if src_key != merged_key:
                step_out.pop(src_key, None)

        merged_frames.append(step_out)

    return merged_frames

# ---- phylogeny color helpers ----
def _mother_id(agent_id: str):
    """Return mother id if agent_id ends with _0 or _1; else None."""
    if len(agent_id) >= 2 and agent_id[-2] == '_' and agent_id[-1] in ('0', '1'):
        return agent_id[:-2]
    return None

def _mutate_hsv(h, s, v, rng, dh=0.05, ds=0.05, dv=0.05):
    """Small HSV mutation with clamping + hue wrap."""
    h_new = (h + rng.uniform(-dh, dh)) % 1.0
    s_new = min(1.0, max(0.0, s + rng.uniform(-ds, ds)))
    v_new = min(1.0, max(0.0, v + rng.uniform(-dv, dv)))
    return h_new, s_new, v_new

def build_phylogeny_colors(frames, agents_key='agents',
                           seed=None, base_s=0.70, base_v=0.95,
                           dh=0.05, ds=0.03, dv=0.03):
    """
    Scan frames in order, assign colors by phylogeny.
    Returns dict id -> (r,g,b) in 0..1
    """
    rng = random.Random(seed)
    hsv_map = {}  # id -> (h,s,v)

    for step in frames:
        layer = step.get(agents_key, {}) or {}
        for aid in layer.keys():
            if aid in hsv_map:
                continue
            mom = _mother_id(aid)
            if mom and mom in hsv_map:
                h, s, v = _mutate_hsv(*hsv_map[mom], rng, dh=dh, ds=ds, dv=dv)
                hsv_map[aid] = (h, s, v)
            else:
                hsv_map[aid] = (rng.random(), base_s, base_v)

    return {aid: tuple(hsv_to_rgb(hsv_map[aid])) for aid in hsv_map}

# ===================== Factored rendering helpers =====================

def _prepare_data_and_paths(data, config, agents_key, out_dir, filename, skip_frames):
    """Normalize frames, build path, read env/barriers, downsample frames."""
    if isinstance(data, (list, tuple)) and data:
        data = merge_plot_layers(data, merged_key=agents_key)

    os.makedirs(out_dir, exist_ok=True)
    filename = _ensure_gif_filename(filename)
    out_path = filename if os.path.dirname(filename) else os.path.join(out_dir, filename)
    out_path = os.path.abspath(os.path.expanduser(out_path))

    env_size = float(config['env_size'])
    barriers = config.get('barriers', [])
    frames = list(data)[::skip_frames]
    if not frames:
        raise ValueError("No frames to render.")

    return out_path, env_size, barriers, frames

def _init_figure(env_size, figure_size_inches, dpi, show_time_title):
    """Set up fig/canvas/axes with equal aspect and fixed limits."""
    fig = plt.figure(figsize=figure_size_inches, dpi=dpi)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, env_size)
    ax.set_ylim(0, env_size)
    ax.set_axis_off()
    ax.set_autoscale_on(False)

    # Establish transforms & pixel geometry
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    ypix_per_unit = _pixels_per_data_y(ax)
    img_bbox = (0, 0, w - 1, h - 1)
    title_obj = ax.set_title("") if show_time_title else None
    return fig, canvas, ax, title_obj, ypix_per_unit, img_bbox

def _draw_static_barriers(ax, barriers, ypix_per_unit, max_line_px, color='gray'):
    """Draw once; returns list of Line2D."""
    dpi_fig = ax.figure.dpi
    out = []
    for b in barriers:
        (sx, sy), (ex, ey) = b['start'], b['end']
        if not _finite(sx, sy, ex, ey):
            continue
        t_world = float(b.get('thickness', 1.0))
        lw_px = max(1, min(int(round(t_world * ypix_per_unit)), max_line_px))
        lw_pt = lw_px * 72.0 / dpi_fig
        art = Line2D([sx, ex], [sy, ey], linewidth=lw_pt, color=color, antialiased=False)
        ax.add_line(art)
        out.append(art)
    return out

def _precompute_pool_sizes(frames, agents_key):
    """Scan once to size object pools."""
    max_circles = 0
    max_segments = 0
    for step in frames:
        vals = list(step[agents_key].values())
        max_circles  = max(max_circles,  sum(1 for o in vals if o.get('type') == 'circle'))
        max_segments = max(max_segments, sum(1 for o in vals if o.get('type') == 'segment'))
    return max_circles, max_segments

def _build_artist_pools(ax, max_circles, max_segments):
    """Allocate invisible artists we’ll reuse."""
    circle_pool = []
    for _ in range(max_circles):
        c = Circle((0, 0), 1.0, fill=True, antialiased=False, visible=False)
        ax.add_patch(c)
        circle_pool.append(c)
    segment_pool = []
    for _ in range(max_segments):
        ln = LineWidthData([0, 0], [0, 0],
                           linewidth=1.0,
                           solid_capstyle='round',
                           antialiased=False,
                           visible=False)
        ax.add_line(ln)
        segment_pool.append(ln)
    return circle_pool, segment_pool

def _update_pools_for_step(
    step,
    agents_key,
    ax,
    circle_pool,
    segment_pool,
    ypix_per_unit,
    img_bbox,
    pad_px,
    rgb_colors,
    default_rgb,
    max_line_px,
    max_radius_px
):
    """Populate pooled artists for a single frame; return visible counts."""
    c_vis = 0
    s_vis = 0

    # Circles
    for aid, o in ((k, v) for k, v in step[agents_key].items() if v.get('type') == 'circle'):
        cx, cy = o['location']
        r_world = float(o['radius'])
        if not _finite(cx, cy, r_world) or r_world <= 0:
            continue

        # cull
        px0, py0 = ax.transData.transform((cx - r_world, cy - r_world))
        px1, py1 = ax.transData.transform((cx + r_world, cy + r_world))
        if _bbox_outside((int(px0), int(py0), int(px1), int(py1)), img_bbox, pad_px):
            continue

        rgb = rgb_colors.get(aid, default_rgb)
        art = circle_pool[c_vis]
        art.center = (cx, cy)
        art.set_radius(r_world)  # data-units radius (no clamping for drawing)
        art.set_facecolor(rgb)
        art.set_edgecolor(rgb)
        art.set_visible(True)
        c_vis += 1
        if c_vis >= len(circle_pool):
            break

    for i in range(c_vis, len(circle_pool)):
        if circle_pool[i].get_visible():
            circle_pool[i].set_visible(False)

    # Segments (capsules)
    for aid, o in ((k, v) for k, v in step[agents_key].items() if v.get('type') == 'segment'):
        L = float(o['length'])
        r_world = float(o['radius'])
        ang = _norm_angle(float(o['angle']))
        cx, cy = o['location']
        if not _finite(cx, cy, L, r_world, ang) or L <= 0 or r_world <= 0:
            continue

        # straight core trimmed by one radius on each end
        half = 0.5 * L
        length_offset = max(half - r_world, 0.0) # trims one radius off each end (total 2*radius)
        dx_axis = math.cos(ang) * length_offset
        dy_axis = math.sin(ang) * length_offset
        x0w, y0w = cx - dx_axis, cy - dy_axis
        x1w, y1w = cx + dx_axis, cy + dy_axis

        # pixel linewidth target (2*radius), clamped -> convert to data units
        lw_nom_px = int(round(2.0 * r_world * ypix_per_unit))
        lw_px = max(1, min(lw_nom_px, max_line_px))
        lw_data = lw_px / max(ypix_per_unit, 1e-9)

        # cull (include half thickness padding)
        p0 = ax.transData.transform((x0w, y0w))
        p1 = ax.transData.transform((x1w, y1w))
        pad = lw_px * 0.5
        xmin = int(min(p0[0], p1[0]) - pad)
        xmax = int(max(p0[0], p1[0]) + pad)
        ymin = int(min(p0[1], p1[1]) - pad)
        ymax = int(max(p0[1], p1[1]) + pad)
        if _bbox_outside((xmin, ymin, xmax, ymax), img_bbox, pad_px):
            continue

        rgb = rgb_colors.get(aid, default_rgb)
        art = segment_pool[s_vis]
        art.set_xdata([x0w, x1w])
        art.set_ydata([y0w, y1w])
        art.set_linewidth(lw_data)  # data-units width via LineWidthData
        art.set_color(rgb)
        art.set_visible(True)
        s_vis += 1
        if s_vis >= len(segment_pool):
            break

    for i in range(s_vis, len(segment_pool)):
        if segment_pool[i].get_visible():
            segment_pool[i].set_visible(False)

    return c_vis, s_vis

def _render_frames(
    frames,
    agents_key,
    fig, canvas, ax, title_obj,
    circle_pool, segment_pool,
    ypix_per_unit, img_bbox, pad_px,
    rgb_colors, default_rgb,
    max_line_px, max_radius_px,
    show_time_title
):
    """Generate a list of PIL Image frames by blitting visible artists."""
    # cache background
    canvas.draw()
    background = canvas.copy_from_bbox(ax.bbox)

    pil_frames = []
    for step in frames:
        _update_pools_for_step(
            step, agents_key, ax, circle_pool, segment_pool,
            ypix_per_unit, img_bbox, pad_px, rgb_colors, default_rgb,
            max_line_px, max_radius_px
        )
        if show_time_title and title_obj is not None:
            title_obj.set_text(f"t={step.get('time', 0):.1f}")

        canvas.restore_region(background)
        if show_time_title and title_obj is not None:
            ax.draw_artist(title_obj)
        for art in circle_pool:
            if art.get_visible():
                ax.draw_artist(art)
        for art in segment_pool:
            if art.get_visible():
                ax.draw_artist(art)
        canvas.blit(ax.bbox)

        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        rgb = rgba[:, :, :3].copy()
        pil_frames.append(Image.fromarray(rgb))
    return pil_frames

def _save_gif(pil_frames, out_path, duration_ms=100, loop=0):
    """Write frames to disk as a GIF (disposal=2 to clear between frames)."""
    if not pil_frames:
        raise ValueError("No frames to save.")
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=2
    )

# ===================== Public API (refactored) =====================

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
    max_radius_px=40,     # max circle radius in pixels (drawn in data units; used for cull calc)
    # phylogeny coloring:
    color_by_phylogeny=True,
    color_seed=None,
    base_s=0.70, base_v=0.95,
    mutate_dh=0.05, mutate_ds=0.03, mutate_dv=0.03,
    # optional fallback color if not coloring by phylogeny
    default_rgb=(0.2, 0.6, 0.9),
):
    """
    Fast Matplotlib renderer with:
      - layer merge (agents + particles + …)
      - culling & size clamping
      - rounded capsules via LineWidthData
      - phylogeny-based colors (mother → daughter hue mutations)
    """
    # ---- prep data/paths/frames
    out_path, env_size, barriers, frames = _prepare_data_and_paths(
        data, config, agents_key, out_dir, filename, skip_frames
    )

    # ---- colors (stable across frames)
    if color_by_phylogeny:
        rgb_colors = build_phylogeny_colors(
            frames, agents_key=agents_key,
            seed=color_seed, base_s=base_s, base_v=base_v,
            dh=mutate_dh, ds=mutate_ds, dv=mutate_dv
        )
    else:
        rgb_colors = {}

    # ---- figure/canvas
    fig, canvas, ax, title_obj, ypix_per_unit, img_bbox = _init_figure(
        env_size, figure_size_inches, dpi, show_time_title
    )
    pad_px = int(round(world_pad * ypix_per_unit))

    # ---- static barriers
    _draw_static_barriers(ax, barriers, ypix_per_unit, max_line_px)

    # ---- pools
    max_circles, max_segments = _precompute_pool_sizes(frames, agents_key)
    circle_pool, segment_pool = _build_artist_pools(ax, max_circles, max_segments)

    # ---- render
    try:
        pil_frames = _render_frames(
            frames, agents_key, fig, canvas, ax, title_obj,
            circle_pool, segment_pool,
            ypix_per_unit, img_bbox, pad_px,
            rgb_colors, default_rgb,
            max_line_px, max_radius_px,
            show_time_title
        )
    finally:
        plt.close(fig)

    # ---- save
    _save_gif(pil_frames, out_path, duration_ms=100, loop=0)
    print(f"GIF saved to {out_path}")
    return out_path
