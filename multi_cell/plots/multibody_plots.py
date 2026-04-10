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
        draw_walls=True, adhesion_surface=None, field_overlay=None,
        draw_trails=False, trail_alpha=0.7, trail_linewidth=0.6,
        trail_fade_frames=8.0, trail_max_frames=40,
        env_width=None, env_height=None,
        scale_bar=None,
        min_cell_px=0.0,
    ):
        self.env_size = float(env_size)
        # Rectangular chamber support: if env_width / env_height aren't
        # explicitly given, fall back to env_size for both (square chamber).
        self.env_width = float(env_width) if env_width else float(env_size)
        self.env_height = float(env_height) if env_height else float(env_size)
        self.scale_bar_cfg = scale_bar
        self.min_cell_px = float(min_cell_px)
        self.show_time_title = show_time_title
        self.max_line_px = int(max_line_px)
        # Optional concentration-field background overlay
        # field_overlay = {'key': 'fields', 'mol_id': 'glucose', 'vmin', 'vmax', 'cmap', 'alpha', 'colorbar': bool}
        self.field_overlay_cfg = field_overlay
        self.field_overlay_artist = None
        # Per-cell trail rendering with fading alpha. Each draw_frame call
        # appends the cell's current location to a per-id history and
        # rebuilds a LineCollection where the most recent segments have
        # full alpha and older segments fade out exponentially with
        # `trail_fade_frames` as the time constant. History is capped at
        # `trail_max_frames` to bound memory and rendering cost.
        self.draw_trails = bool(draw_trails)
        self.trail_alpha = float(trail_alpha)
        self.trail_linewidth = float(trail_linewidth)
        self.trail_fade_frames = max(1e-6, float(trail_fade_frames))
        self.trail_max_frames = int(trail_max_frames)
        self._trail_history = {}      # cell_id -> list[(x, y)]
        self._trail_collections = {}  # cell_id -> LineCollection

        x0, x1 = xlim if xlim else (0, self.env_size)
        y0, y1 = ylim if ylim else (0, self.env_size)

        # fig/ax/canvas — match figure aspect ratio to data, but if the
        # caller passes an EXPLICIT non-square figure_size_inches use it
        # directly so extreme aspect ratios stay readable.
        data_w = x1 - x0
        data_h = y1 - y0
        if (
            figure_size_inches
            and len(figure_size_inches) >= 2
            and figure_size_inches[0] != figure_size_inches[1]
        ):
            fig_w, fig_h = float(figure_size_inches[0]), float(figure_size_inches[1])
        else:
            base = figure_size_inches[0] if figure_size_inches else 6.0
            if data_h > data_w:
                fig_w = base * (data_w / data_h)
                fig_h = base
            else:
                fig_w = base
                fig_h = base * (data_h / data_w)

        # When a field colorbar is shown, widen the figure so it lives in its
        # own column to the right of the data axes. The fraction can be
        # tuned per-experiment via field_overlay['width_frac'] (default
        # 0.10 — narrow enough to leave room for the chamber but wide
        # enough to fit tick labels).
        show_field_cbar = bool(field_overlay and field_overlay.get('colorbar'))
        cbar_width_frac = float(field_overlay.get('width_frac', 0.10)) if show_field_cbar else 0.0
        cbar_extra = cbar_width_frac * fig_w if show_field_cbar else 0.0

        # Reserve some figure space below the data axes for the scale bar
        # (when one is configured) so the bar sits *below* the chamber rather
        # than over the cells. The scale bar itself is positioned via
        # bbox_to_anchor in axes-fraction coordinates inside _draw_scale_bar.
        bottom_margin = 0.10 if self.scale_bar_cfg else 0.02
        top_margin = 0.08 if show_time_title else 0.02
        ax_height = max(0.05, 1.0 - bottom_margin - top_margin)

        self.fig = plt.figure(figsize=(fig_w + cbar_extra, fig_h), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        if show_field_cbar:
            # Reserve a column on the right for the colorbar
            ax_frac = fig_w / (fig_w + cbar_extra)
            self.ax = self.fig.add_axes(
                [0.02, bottom_margin, ax_frac * 0.93, ax_height]
            )
        else:
            self.ax = self.fig.add_axes([0.02, bottom_margin, 0.96, ax_height])
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
        # Use a monospace font + fixed-width HH:MM:SS format so the title's
        # bounding box stays the exact same width every frame. Otherwise the
        # centered title shifts horizontally as digits and units change,
        # which looks like the title is "jittering".
        if show_time_title:
            self.title_obj = self.ax.set_title(
                "", fontfamily='monospace', fontsize=11, loc='center',
            )
        else:
            self.title_obj = None

        # Field overlay (heatmap) drawn first so cells render on top of it
        if self.field_overlay_cfg:
            self._init_field_overlay(x0, x1, y0, y1)

        # draw flow regions, walls, adhesion surface, and barriers once
        if flow_regions:
            self._draw_flow_regions(flow_regions, x0, x1, y0, y1)
        if draw_walls:
            self._draw_walls()
        if adhesion_surface:
            self._draw_adhesion_surface(adhesion_surface)
        self._draw_barriers(barriers)
        if show_field_cbar:
            self._draw_field_colorbar(field_overlay)
        if self.scale_bar_cfg:
            self._draw_scale_bar(self.scale_bar_cfg)

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

    def _init_field_overlay(self, x0, x1, y0, y1):
        """Set up an imshow artist for a 2D concentration field background.

        Creates the artist eagerly with a (1, 1) placeholder so the colorbar
        can be linked to it via fig.colorbar(artist, ...). The placeholder is
        replaced by set_data on the first frame; the explicit Normalize keeps
        the color scale (and the colorbar's range) locked across frames.
        """
        from matplotlib.colors import Normalize
        cfg = self.field_overlay_cfg
        cmap = cfg.get('cmap', 'YlGn')
        vmin = float(cfg.get('vmin', 0.0))
        vmax = float(cfg.get('vmax', 1.0))
        alpha = float(cfg.get('alpha', 0.85))
        # Locked normalization shared by both heatmap and colorbar.
        self._field_overlay_norm = Normalize(vmin=vmin, vmax=vmax, clip=False)
        self.field_overlay_artist = self.ax.imshow(
            np.zeros((1, 1), dtype=float),
            extent=(x0, x1, y0, y1),
            origin='lower',
            cmap=cmap,
            norm=self._field_overlay_norm,
            alpha=alpha,
            interpolation='nearest',
            zorder=-2,
        )

    def _update_field_overlay(self, step):
        if not self.field_overlay_cfg or self.field_overlay_artist is None:
            return
        cfg = self.field_overlay_cfg
        mol_id = cfg.get('mol_id')
        if not mol_id:
            return
        arr = (step.get(cfg.get('key', 'fields')) or {}).get(mol_id)
        if arr is None:
            return
        self.field_overlay_artist.set_data(np.asarray(arr, dtype=float))

    def _draw_walls(self):
        """Draw the four chamber walls as a thin gray rectangle outline."""
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (0, 0), self.env_width, self.env_height,
            facecolor='none', edgecolor='#888', linewidth=1.2,
        )
        self.ax.add_patch(rect)

    def _draw_scale_bar(self, cfg):
        """Draw a thin scale bar in the figure margin below the data axes.

        The bar's *length* is in data units (so it represents a real
        distance in µm), but its position is in axes-fraction coordinates
        so it sits below the chamber rather than over the cells, and its
        *thickness* is set in matplotlib points so it stays a thin line
        regardless of the chamber size.
        """
        size = float(cfg.get('size', 100.0))
        label = cfg.get('label', f'{int(size)} µm')
        loc = cfg.get('loc', 'lower right')
        color = cfg.get('color', '#222')
        fontsize = int(cfg.get('fontsize', 11))
        linewidth_pt = float(cfg.get('linewidth', 1.6))

        # Convert data-unit bar length to an axes-fraction width.
        x0, x1 = self.ax.get_xlim()
        if x1 == x0:
            return
        bar_frac = abs(size / (x1 - x0))

        # Below the axes — small negative y in axes coords.
        y_bar = -0.04
        y_label = -0.07

        if 'right' in loc:
            bar_x_end = 0.99
            bar_x_start = bar_x_end - bar_frac
            text_x = bar_x_start + bar_frac / 2
        elif 'left' in loc:
            bar_x_start = 0.01
            bar_x_end = bar_x_start + bar_frac
            text_x = bar_x_start + bar_frac / 2
        else:
            bar_x_start = 0.5 - bar_frac / 2
            bar_x_end = 0.5 + bar_frac / 2
            text_x = 0.5

        bar_line = Line2D(
            [bar_x_start, bar_x_end], [y_bar, y_bar],
            transform=self.ax.transAxes,
            color=color,
            linewidth=linewidth_pt,
            solid_capstyle='butt',
            clip_on=False,
        )
        self.ax.add_line(bar_line)
        self.ax.text(
            text_x, y_label, label,
            transform=self.ax.transAxes,
            ha='center', va='top',
            fontsize=fontsize, color=color,
            clip_on=False,
        )

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

    def _draw_field_colorbar(self, field_cfg):
        """Add a colorbar linked directly to the field-overlay imshow artist
        so its range and tick labels stay locked to the artist's Normalize."""
        if self.field_overlay_artist is None:
            return
        label = field_cfg.get('colorbar_label') or field_cfg.get('mol_id') or 'concentration'
        ax_pos = self.ax.get_position()
        cax_left = ax_pos.x1 + 0.015
        cax_width = max(0.02, 1.0 - cax_left - 0.04)
        # Use most of the reserved column for the visible bar — the
        # remainder is just enough to keep tick labels from clipping.
        cax = self.fig.add_axes([cax_left, ax_pos.y0 + 0.10,
                                 cax_width * 0.55, ax_pos.height * 0.7])
        cbar = self.fig.colorbar(self.field_overlay_artist, cax=cax)
        cbar.set_label(label, fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        # Pin the tick locations so matplotlib doesn't recompute them per frame.
        vmin = float(field_cfg.get('vmin', 0.0))
        vmax = float(field_cfg.get('vmax', 1.0))
        ticks = np.linspace(vmin, vmax, 6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f'{t:.2f}' for t in ticks])
        self._field_colorbar = cbar

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

    def _update_trails(self, layer, color_fn):
        """Append each cell's current position to its history and update
        a per-cell LineCollection where the most recent segments are
        opaque and older segments fade out exponentially with time."""
        from matplotlib.collections import LineCollection
        for aid, ent in layer.items():
            loc = ent.get('location') if isinstance(ent, dict) else None
            if loc is None or len(loc) < 2:
                continue
            try:
                x = float(loc[0])
                y = float(loc[1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            history = self._trail_history.get(aid)
            if history is None:
                history = []
                self._trail_history[aid] = history
            history.append((x, y))
            # Cap history to bound memory and rendering cost.
            if len(history) > self.trail_max_frames:
                # Drop the oldest entries.
                del history[: len(history) - self.trail_max_frames]
            n = len(history)
            if n < 2:
                continue
            # Build segments with fading per-segment alpha. Most recent
            # segment (index n-2) has full alpha; older segments decay
            # exponentially with trail_fade_frames as the e-folding time.
            color = color_fn(aid, ent)
            r, g, b = float(color[0]), float(color[1]), float(color[2])
            segments = []
            rgba = []
            for i in range(n - 1):
                age = (n - 2) - i  # 0 for newest, n-2 for oldest
                a = self.trail_alpha * math.exp(-age / self.trail_fade_frames)
                if a < 0.01:
                    continue  # invisible — skip
                segments.append([history[i], history[i + 1]])
                rgba.append((r, g, b, a))
            lc = self._trail_collections.get(aid)
            if lc is None:
                lc = LineCollection(
                    segments, colors=rgba,
                    linewidths=self.trail_linewidth,
                    capstyle='round',
                    antialiased=True,
                    zorder=-1,
                )
                self.ax.add_collection(lc)
                self._trail_collections[aid] = lc
            else:
                lc.set_segments(segments)
                lc.set_colors(rgba)

    # ------- per-frame update -------
    def draw_frame(self, step, agents_key, color_fn, max_radius_px):
        c_vis = s_vis = 0
        layer = step.get(agents_key) or {}

        # Update field background overlay (if configured)
        self._update_field_overlay(step)

        # Append to and refresh persistent path-trail artists.
        if self.draw_trails:
            self._update_trails(layer, color_fn)

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
            # Clamp to a minimum on-screen pixel width so very small cells
            # in a large chamber are still visible. min_cell_px is given
            # in screen pixels; ypu = pixels per data unit.
            if self.min_cell_px > 0 and self.ypu > 0:
                min_lw_data = self.min_cell_px / self.ypu
                if lw_data < min_lw_data:
                    lw_data = min_lw_data

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

        # title — fixed-width HH:MM:SS so the centered title doesn't shift
        # horizontally as the simulation time grows.
        if self.show_time_title and self.title_obj is not None:
            t_now = float(step.get('time', 0) or 0)
            if not math.isfinite(t_now):
                t_now = 0.0
            t_int = int(t_now)
            hh = t_int // 3600
            mm = (t_int % 3600) // 60
            ss = t_int % 60
            self.title_obj.set_text(f"t = {hh:02d}:{mm:02d}:{ss:02d}")

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
    field_overlay=None,     # optional dict {'mol_id', 'vmax', 'cmap', 'alpha'} for heatmap background
    draw_trails=False,      # if True, trace each cell's path across frames
    trail_alpha=0.7,
    trail_linewidth=0.6,
    trail_fade_frames=8.0,  # e-folding age (in frames) for trail alpha decay
    trail_max_frames=40,    # max # of past positions kept per cell
    env_width=None,         # rectangular chamber width (defaults to env_size)
    env_height=None,        # rectangular chamber height (defaults to env_size)
    scale_bar=None,         # dict {size, label, loc, ...} or None
    min_cell_px=0.0,        # min on-screen pixel width for rendered cells
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

    # If a field overlay is requested, pre-scan all frames so the colorbar
    # range is constant across the GIF (otherwise per-frame autoscaling would
    # make the colors drift). The user-supplied vmin/vmax win if present.
    if field_overlay and field_overlay.get('mol_id'):
        key = field_overlay.get('key', 'fields')
        mol_id = field_overlay['mol_id']
        global_min = float('inf')
        global_max = float('-inf')
        for step in frames:
            arr = (step.get(key) or {}).get(mol_id)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                continue
            global_min = min(global_min, float(a.min()))
            global_max = max(global_max, float(a.max()))
        if not math.isfinite(global_min):
            global_min = 0.0
        if not math.isfinite(global_max) or global_max <= global_min:
            global_max = global_min + 1.0
        field_overlay = dict(field_overlay)
        field_overlay.setdefault('vmin', global_min)
        field_overlay.setdefault('vmax', global_max)

    # render
    renderer = GifRenderer(
        env_size, barriers, figure_size_inches, dpi, show_time_title,
        world_pad, max_line_px, xlim=xlim, ylim=ylim,
        flow_regions=flow_regions, draw_walls=draw_walls,
        adhesion_surface=adhesion_surface,
        field_overlay=field_overlay,
        draw_trails=draw_trails,
        trail_alpha=trail_alpha,
        trail_linewidth=trail_linewidth,
        trail_fade_frames=trail_fade_frames,
        trail_max_frames=trail_max_frames,
        env_width=env_width,
        env_height=env_height,
        scale_bar=scale_bar,
        min_cell_px=min_cell_px,
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
