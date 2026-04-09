import os, math, random
import numpy as np
from math import pi
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import hsv_to_rgb

# ----------------- LineWidthData -----------------
class LineWidthData(Line2D):
    """A Line2D whose linewidth is specified in data units (world units)."""
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop('linewidth', 1.0)
        super().__init__(*args, **kwargs)
        self._lw_data = float(_lw_data)
    def _get_lw(self):
        if self.axes is None:
            return 1.0
        (_, y0), (_, y1) = self.axes.transData.transform([(0, 0), (0, 1)])
        px_per_data = abs(y1 - y0)
        ppd = 72.0 / self.axes.figure.dpi  # points per pixel
        return max(0.1, px_per_data * self._lw_data * ppd)
    def _set_lw(self, lw): self._lw_data = float(lw)
    _linewidth = property(_get_lw, _set_lw)

# --------- tiny utils ---------
def _ensure_gif_filename(path):
    root, ext = os.path.splitext(path)
    return f"{path}.gif" if ext == "" else path

def _finite(*vals): return all(math.isfinite(v) for v in vals)
def _norm_angle(theta): return (theta + pi) % (2*pi) - pi

def _pixels_per_data_y(ax):
    (_, y0), (_, y1) = ax.transData.transform([(0, 0), (0, 1)])
    return abs(y1 - y0)

def _bbox_outside(px_bbox, img_bbox, pad_px):
    ix0, iy0, ix1, iy1 = img_bbox
    x0, y0, x1, y1 = px_bbox
    return (x1 < ix0 - pad_px or x0 > ix1 + pad_px or
            y1 < iy0 - pad_px or y0 > iy1 + pad_px)

def _infer_plot_type(o):
    t = o.get('type')
    if t in ('circle', 'segment'): return t
    if 'radius' in o and 'length' in o: return 'segment'
    if 'radius' in o: return 'circle'
    return None

def _is_plot_entity(o):
    if not isinstance(o, dict): return False
    loc = o.get('location')
    return isinstance(loc, (tuple, list)) and len(loc) == 2 and _infer_plot_type(o) is not None

def _is_entity_map(v): return isinstance(v, dict) and all(_is_plot_entity(x) for x in v.values())

def merge_plot_layers(data, merged_key='agents'):
    out = []
    for step in data:
        step_out = dict(step)
        base = dict(step_out.get(merged_key, {}))
        existing = set(base.keys())
        for k, v in step.items():
            if k == merged_key or not _is_entity_map(v): continue
            for ent_id, ent in v.items():
                out_id = ent_id if ent_id not in existing else f"{k}:{ent_id}"
                if ent.get('type') not in ('circle', 'segment'):
                    t = _infer_plot_type(ent)
                    if t is not None: ent = {**ent, 'type': t}
                base[out_id] = ent
                existing.add(out_id)
            step_out.pop(k, None)
        step_out[merged_key] = base
        out.append(step_out)
    return out

# --------- phylogeny coloring ---------
def _mother_id(agent_id: str):
    return agent_id[:-2] if len(agent_id) >= 2 and agent_id[-2] == '_' and agent_id[-1] in ('0', '1') else None

def _mutate_hsv(h, s, v, rng, dh=0.05, ds=0.03, dv=0.03):
    h = (h + rng.uniform(-dh, dh)) % 1.0
    s = min(1.0, max(0.0, s + rng.uniform(-ds, ds)))
    v = min(1.0, max(0.0, v + rng.uniform(-dv, dv)))
    return h, s, v

def build_phylogeny_colors(frames, agents_key='agents', seed=None, base_s=0.70, base_v=0.95, dh=0.05, ds=0.03, dv=0.03):
    rng, hsv_map = random.Random(seed), {}
    for step in frames:
        for aid in (step.get(agents_key) or {}).keys():
            if aid in hsv_map: continue
            mom = _mother_id(aid)
            if mom in hsv_map:
                hsv_map[aid] = _mutate_hsv(*hsv_map[mom], rng, dh, ds, dv)
            else:
                hsv_map[aid] = (rng.random(), base_s, base_v)
    return {aid: tuple(hsv_to_rgb(hsv)) for aid, hsv in hsv_map.items()}

# ================= CORE RENDERER =================
class GifRenderer:
    def __init__(
        self, env_size, barriers, figure_size_inches, dpi, show_time_title,
        world_pad, max_line_px, xlim=None, ylim=None, flow_regions=None,
        draw_walls=True, adhesion_surface=None, pressure_colorbar=None,
    ):
        self.env_size = float(env_size)
        self.show_time_title = show_time_title
        self.max_line_px = int(max_line_px)

        x0, x1 = xlim if xlim else (0, self.env_size)
        y0, y1 = ylim if ylim else (0, self.env_size)

        # fig/ax/canvas — match figure aspect ratio to data
        data_w = x1 - x0
        data_h = y1 - y0
        base = figure_size_inches[0]
        if data_h > data_w:
            fig_w = base * (data_w / data_h)
            fig_h = base
        else:
            fig_w = base
            fig_h = base * (data_h / data_w)

        # When a colorbar is shown, widen the figure so the colorbar lives
        # in its own column on the right, outside the data axes.
        cbar_extra = 0.18 * fig_w if pressure_colorbar else 0.0

        self.fig = plt.figure(figsize=(fig_w + cbar_extra, fig_h), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        if pressure_colorbar:
            # Reserve a column on the right for the colorbar
            ax_frac = fig_w / (fig_w + cbar_extra)
            self.ax = self.fig.add_axes([0.0, 0.0, ax_frac * 0.95, 1.0])
        else:
            self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_axis_off()
        self.ax.set_autoscale_on(False)

        # establish pixel geometry
        self.canvas.draw()
        w, h = self.fig.canvas.get_width_height()
        self.img_bbox = (0, 0, w - 1, h - 1)
        self.ypu = _pixels_per_data_y(self.ax)
        self.pad_px = int(round(world_pad * self.ypu))
        self.title_obj = self.ax.set_title("") if show_time_title else None

        # draw flow regions, walls, adhesion surface, and barriers once
        if flow_regions:
            self._draw_flow_regions(flow_regions, x0, x1, y0, y1)
        if draw_walls:
            self._draw_walls()
        if adhesion_surface:
            self._draw_adhesion_surface(adhesion_surface)
        self._draw_barriers(barriers)
        if pressure_colorbar:
            self._draw_pressure_colorbar(pressure_colorbar)

        # pools (grow-on-demand)
        self.circle_pool = []
        self.segment_pool = []

        # background for blitting
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def close(self): plt.close(self.fig)

    def _draw_flow_regions(self, flow_regions, x0, x1, y0, y1):
        """Draw translucent grey rectangles for flow/removal regions.

        Each entry can specify any of: x_min, x_max, y_min, y_max.
        Missing bounds extend to the viewport edges.
        """
        from matplotlib.patches import Rectangle
        for region in flow_regions:
            rx0 = region.get('x_min', x0)
            rx1 = region.get('x_max', x1)
            ry0 = region.get('y_min', y0)
            ry1 = region.get('y_max', y1)
            rect = Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0,
                facecolor='gray', alpha=0.2, edgecolor='none', zorder=-1,
            )
            self.ax.add_patch(rect)

    def _draw_walls(self):
        """Draw the four chamber walls as a thin gray rectangle outline."""
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (0, 0), self.env_size, self.env_size,
            facecolor='none', edgecolor='#888', linewidth=1.2,
        )
        self.ax.add_patch(rect)

    def _draw_adhesion_surface(self, surface):
        """Highlight the adhesion surface with a colored line."""
        from matplotlib.lines import Line2D
        s = self.env_size
        color = '#d97b00'  # warm orange
        if surface == 'bottom':
            line = Line2D([0, s], [0, 0], linewidth=3.0, color=color, solid_capstyle='butt')
        elif surface == 'top':
            line = Line2D([0, s], [s, s], linewidth=3.0, color=color, solid_capstyle='butt')
        elif surface == 'left':
            line = Line2D([0, 0], [0, s], linewidth=3.0, color=color, solid_capstyle='butt')
        elif surface == 'right':
            line = Line2D([s, s], [0, s], linewidth=3.0, color=color, solid_capstyle='butt')
        else:
            return
        self.ax.add_line(line)

    def _draw_pressure_colorbar(self, info):
        """Add a gray→red colorbar in its own column to the right of the data axes."""
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        pmax = float(info.get('pmax', 8.0))
        cmap = LinearSegmentedColormap.from_list(
            'pressure', [(0.7, 0.7, 0.7), (0.85, 0.1, 0.1)])
        norm = Normalize(vmin=0.0, vmax=pmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        # Place the colorbar in the reserved right column
        ax_pos = self.ax.get_position()
        cax_left = ax_pos.x1 + 0.02
        cax_width = max(0.02, 1.0 - cax_left - 0.05)
        cax = self.fig.add_axes([cax_left, ax_pos.y0 + 0.10, cax_width * 0.4, ax_pos.height * 0.7])
        cbar = self.fig.colorbar(sm, cax=cax)
        cbar.set_label('pressure', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    def _draw_barriers(self, barriers, color='gray'):
        dpi_fig = self.fig.dpi
        for b in barriers or []:
            (sx, sy), (ex, ey) = b['start'], b['end']
            if not _finite(sx, sy, ex, ey): continue
            t_world = float(b.get('thickness', 1.0))
            lw_px = max(1, min(int(round(t_world * self.ypu)), self.max_line_px))
            lw_pt = lw_px * 72.0 / dpi_fig
            self.ax.add_line(Line2D([sx, ex], [sy, ey], linewidth=lw_pt, color=color, antialiased=False))

    # ------- pools -------
    def _need_circle(self, idx):
        while len(self.circle_pool) <= idx:
            c = Circle((0, 0), 1.0, fill=True, antialiased=False, visible=False)
            self.ax.add_patch(c)
            self.circle_pool.append(c)
        return self.circle_pool[idx]

    def _need_segment(self, idx):
        while len(self.segment_pool) <= idx:
            ln = LineWidthData([0, 0], [0, 0], linewidth=1.0, solid_capstyle='round', antialiased=False, visible=False)
            self.ax.add_line(ln)
            self.segment_pool.append(ln)
        return self.segment_pool[idx]

    # ------- per-frame update -------
    def draw_frame(self, step, agents_key, color_fn, max_radius_px):
        c_vis = s_vis = 0
        layer = step.get(agents_key) or {}

        # Defensive: hide ALL pool slots before drawing to guarantee no stale
        # state from a previous frame can leak through.
        for art in self.circle_pool:
            if art.get_visible(): art.set_visible(False)
        for art in self.segment_pool:
            if art.get_visible(): art.set_visible(False)

        # circles
        for aid, o in layer.items():
            if o.get('type') != 'circle': continue
            cx, cy = o['location']; r = float(o['radius'])
            if not _finite(cx, cy, r) or r <= 0: continue
            px0, py0 = self.ax.transData.transform((cx - r, cy - r))
            px1, py1 = self.ax.transData.transform((cx + r, cy + r))
            if _bbox_outside((int(px0), int(py0), int(px1), int(py1)), self.img_bbox, self.pad_px): continue
            art = self._need_circle(c_vis)
            art.center = (cx, cy); art.set_radius(r)
            rgb = color_fn(aid, o); art.set_facecolor(rgb); art.set_edgecolor(rgb)
            art.set_visible(True)
            c_vis += 1

        # segments — draw full capsule shape matching pymunk geometry
        for aid, o in layer.items():
            if o.get('type') != 'segment': continue
            r = float(o['radius'])
            if r <= 0: continue
            color = color_fn(aid, o)
            lw_data = 2.0 * r  # capsule diameter in data units

            # Build a polyline for this cell. Bending cells emit one from
            # PymunkProcess; others (and freshly-divided daughters that have
            # not yet been seen by PymunkProcess) get a synthesized straight
            # 2-point spine from location/length/angle so they always render
            # through the same code path.
            polyline = o.get('polyline')
            if not polyline or len(polyline) < 2:
                L = float(o.get('length', 0.0) or 0.0)
                ang = _norm_angle(float(o.get('angle', 0.0) or 0.0))
                loc = o.get('location')
                if loc is None or L <= 0 or not _finite(loc[0], loc[1], L, r, ang):
                    continue
                half = 0.5 * L
                dx = math.cos(ang) * half; dy = math.sin(ang) * half
                polyline = [(loc[0] - dx, loc[1] - dy), (loc[0] + dx, loc[1] + dy)]

            for i in range(len(polyline) - 1):
                x0, y0 = polyline[i]
                x1, y1 = polyline[i + 1]
                if not _finite(x0, y0, x1, y1):
                    continue
                art = self._need_segment(s_vis)
                art.set_xdata([x0, x1]); art.set_ydata([y0, y1])
                art._lw_data = lw_data; art.set_color(color)
                art.set_visible(True)
                s_vis += 1

        # title
        if self.show_time_title and self.title_obj is not None:
            self.title_obj.set_text(f"t={step.get('time', 0):.1f}")

        # full redraw
        self.canvas.draw()

        # read back
        w, h = self.fig.canvas.get_width_height()
        rgba = np.frombuffer(self.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
        return Image.fromarray(rgba[:, :, :3].copy())

# ================= PUBLIC API =================
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
    max_radius_px=40,     # kept for API-compat; used in culling path if needed
    xlim=None,            # (x0, x1) viewport override
    ylim=None,            # (y0, y1) viewport override
    flow_regions=None,    # list of dicts with optional x_min/x_max/y_min/y_max
    draw_walls=True,      # draw the chamber wall outline
    adhesion_surface=None,  # 'bottom'/'top'/'left'/'right' to highlight adhesive wall
    # coloring:
    color_by_phylogeny=False,   # << default to uniform color
    color_by_pressure=False,    # color cells by their `pressure` field
    pressure_max=10.0,          # pressure value at which color reaches full red
    color_seed=None,
    base_s=0.70, base_v=0.95,
    mutate_dh=0.05, mutate_ds=0.03, mutate_dv=0.03,
    default_rgb=(0.2, 0.6, 0.9),
    uniform_color=(0.2, 0.6, 0.9),  # <set None to disable uniforming
    particle_color=(0.85, 0.85, 0.55),  # default EPS/particle color (pale yellow)
    frame_duration_ms=100,  # milliseconds per frame in the GIF
):
    """
    Efficient Matplotlib renderer:
      - merges plot-capable layers
      - blitting + grow-on-demand artist pools (no prescan)
      - segments shortened by 2*radius
      - uniform adjustable color by default; optional phylogeny coloring
    """
    # merge layers & downsample frames
    if isinstance(data, (list, tuple)) and data:
        data = merge_plot_layers(data, merged_key=agents_key)
    frames = list(data)[::max(1, int(skip_frames))]
    if not frames: raise ValueError("No frames to render.")

    # paths
    os.makedirs(out_dir, exist_ok=True)
    filename = _ensure_gif_filename(filename)
    out_path = filename if os.path.dirname(filename) else os.path.join(out_dir, filename)
    out_path = os.path.abspath(os.path.expanduser(out_path))

    env_size = float(config['env_size'])
    barriers = config.get('barriers', [])

    # color policy
    _pc = particle_color
    if color_by_pressure:
        # Gray (0.7, 0.7, 0.7) at zero pressure → red (0.85, 0.1, 0.1) at pressure_max
        gray = (0.7, 0.7, 0.7)
        red = (0.85, 0.1, 0.1)
        pmax = max(1e-9, float(pressure_max))
        def _color(aid, ent=None):
            if _pc and aid.startswith(('eps_', 'p_')):
                return _pc
            p = 0.0
            if isinstance(ent, dict):
                p = float(ent.get('pressure', 0.0) or 0.0)
            t = max(0.0, min(1.0, p / pmax))
            return (
                gray[0] + (red[0] - gray[0]) * t,
                gray[1] + (red[1] - gray[1]) * t,
                gray[2] + (red[2] - gray[2]) * t,
            )
    elif color_by_phylogeny:
        rgb_colors = build_phylogeny_colors(
            frames, agents_key=agents_key, seed=color_seed,
            base_s=base_s, base_v=base_v, dh=mutate_dh, ds=mutate_ds, dv=mutate_dv
        )
        def _color(aid, ent=None):
            # Particles (e.g. EPS) get a uniform color
            if _pc and aid.startswith(('eps_', 'p_')):
                return _pc
            return rgb_colors.get(aid, default_rgb)
    else:
        # no phylogeny: use uniform if provided, otherwise fallback default
        def _color(aid, ent=None):
            return uniform_color if uniform_color is not None else default_rgb

    # render
    pressure_colorbar = None  # colorbar disabled
    renderer = GifRenderer(
        env_size, barriers, figure_size_inches, dpi, show_time_title,
        world_pad, max_line_px, xlim=xlim, ylim=ylim,
        flow_regions=flow_regions, draw_walls=draw_walls,
        adhesion_surface=adhesion_surface,
        pressure_colorbar=pressure_colorbar,
    )
    try:
        pil_frames = [renderer.draw_frame(step, agents_key, _color, max_radius_px) for step in frames]
    finally:
        renderer.close()

    # save — quantize to palette to avoid PIL merging "similar" frames
    if not pil_frames: raise ValueError("No frames to save.")
    palette_frames = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for f in pil_frames]
    palette_frames[0].save(
        out_path, save_all=True, append_images=palette_frames[1:],
        duration=frame_duration_ms, loop=0, optimize=False, disposal=2
    )
    print(f"GIF saved to {out_path}")
    return out_path
