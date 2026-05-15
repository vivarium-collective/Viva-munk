"""Visualization Step subclasses for viva-munk composite docs.

Follows the pbg-superpowers Visualization convention: each subclass
consumes per-step state via wires (like an emitter), accumulates frames
internally, and returns ``{'html': '<rendered>'}`` each update.

``MultibodyVizStep`` delegates per-frame rendering to ``GifRenderer``
from ``multi_cell.plots.multibody_plots`` — the same renderer the
post-run pipeline (``multi_cell.experiments.runner.simulation_to_gif``)
uses for the on-disk GIFs in ``out/``. Frames are accumulated in the
Step instance and re-encoded as an inline animated GIF data URI on
every ``update()``; the GIF assembly uses the same quantize + disposal
pattern as ``simulation_to_gif`` so visuals match 1:1.

Per-composite color modes (pressure / qs_state / inclusion_body) are
config flags on the Step; phylogeny coloring is deferred (needs full
history pre-scan, awkward in a streaming setup).
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")  # headless render — no display required
import numpy as np
from PIL import Image

from pbg_superpowers.visualization import Visualization

from multi_cell.plots.multibody_plots import GifRenderer


# Color modes that don't require history pre-scan — usable in streaming.
_PARTICLE_DEFAULT_RGB = (0.85, 0.85, 0.55)  # pale yellow (eps_/p_ ids)


def _color_uniform(rgb=(0.2, 0.6, 0.9), particle_rgb=_PARTICLE_DEFAULT_RGB):
    """Default per-agent color: solid blue for cells, pale yellow for particles."""
    def _color(aid, ent=None):
        if aid.startswith(('eps_', 'p_')):
            return particle_rgb
        return rgb
    return _color


def _color_by_pressure(pressure_max=10.0, gray=(0.7, 0.7, 0.7),
                       red=(0.85, 0.1, 0.1), particle_rgb=_PARTICLE_DEFAULT_RGB):
    """Gray→red gradient on agent.pressure ∈ [0, pressure_max]."""
    pmax = max(1e-9, float(pressure_max))
    def _color(aid, ent=None):
        if aid.startswith(('eps_', 'p_')):
            return particle_rgb
        p = 0.0
        if isinstance(ent, dict):
            p = float(ent.get('pressure', 0.0) or 0.0)
        t = max(0.0, min(1.0, p / pmax))
        return (
            gray[0] + (red[0] - gray[0]) * t,
            gray[1] + (red[1] - gray[1]) * t,
            gray[2] + (red[2] - gray[2]) * t,
        )
    return _color


def _color_by_qs_state(threshold=0.5,
                       off_rgb=(0.25, 0.55, 0.90),
                       on_rgb=(0.95, 0.20, 0.55),
                       particle_rgb=_PARTICLE_DEFAULT_RGB):
    """Binary OFF/ON on agent.qs_state vs. threshold."""
    def _color(aid, ent=None):
        if aid.startswith(('eps_', 'p_')):
            return particle_rgb
        s = 0.0
        if isinstance(ent, dict):
            s = float(ent.get('qs_state', 0.0) or 0.0)
        return on_rgb if s >= threshold else off_rgb
    return _color


def _color_by_inclusion_body(ib_max=10.0, cmap_name='plasma',
                             particle_rgb=_PARTICLE_DEFAULT_RGB):
    """Colormap on agent.inclusion_body ∈ [0, ib_max]."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=0.0, vmax=ib_max, clip=True)
    def _color(aid, ent=None):
        if aid.startswith(('eps_', 'p_')):
            return particle_rgb
        ib = 0.0
        if isinstance(ent, dict):
            ib = float(ent.get('inclusion_body', 0.0) or 0.0)
        r, g, b, _ = cmap(norm(ib))
        return (r, g, b)
    return _color


class MultibodyVizStep(Visualization):
    """Streaming animated-GIF renderer matching the test-suite GIF style.

    Per ``update(state)``:
      1. Merge cells + particles into one layer, stamp the static field
         onto the frame so the heatmap overlay renders every tick.
      2. ``GifRenderer.draw_frame(...)`` → PIL Image; append to internal
         ``self._frames``.
      3. Quantize all accumulated frames (MEDIANCUT, 256 colors) and
         encode as an animated GIF in-memory (``disposal=2``,
         ``optimize=False`` — same pattern as ``simulation_to_gif``).
      4. Return ``{'html': '<img src="data:image/gif;base64,..."/>'}``.

    Performance: re-encoding the GIF on every tick is O(n²) in step count.
    For dashboard test runs (2–50 steps) this is sub-second; for longer
    runs the ``max_frames`` cap drops oldest frames FIFO to bound memory
    and encode time.
    """

    config_schema = {
        'title':             {'_type': 'string',  '_default': ''},
        'env_width':         {'_type': 'float',   '_default': 100.0},
        'env_height':        {'_type': 'float',   '_default': 100.0},
        'figure_width':      {'_type': 'float',   '_default': 6.0},
        'figure_height':     {'_type': 'float',   '_default': 6.0},
        'dpi':               {'_type': 'integer', '_default': 90},
        'max_line_px':       {'_type': 'integer', '_default': 40},
        'show_time_title':   {'_type': 'boolean', '_default': True},
        # Field overlay (heatmap underneath cells).
        'field_mol_id':      {'_type': 'string',  '_default': ''},
        'field_cmap':        {'_type': 'string',  '_default': 'Greys'},
        'field_alpha':       {'_type': 'float',   '_default': 0.35},
        'field_vmin':        {'_type': 'float',   '_default': 0.0},
        'field_vmax':        {'_type': 'float',   '_default': 0.0},  # 0 → auto from first frame
        # Color mode. ''=uniform, 'pressure', 'qs_state', 'inclusion_body'.
        'color_mode':        {'_type': 'string',  '_default': ''},
        'pressure_max':      {'_type': 'float',   '_default': 10.0},
        'qs_state_threshold':{'_type': 'float',   '_default': 0.5},
        'ib_max':            {'_type': 'float',   '_default': 10.0},
        'ib_cmap':           {'_type': 'string',  '_default': 'plasma'},
        # GIF encoding.
        'frame_duration_ms': {'_type': 'integer', '_default': 100},
        # Memory cap. Older frames dropped FIFO once exceeded.
        'max_frames':        {'_type': 'integer', '_default': 200},
        # If the document's `fields` map is static (set at t=0 and never
        # emitted again), the emitter strips it from per-frame snapshots
        # and the heatmap renders blank. Stamp the most-recently-seen
        # field map onto frames that lack one.
        'stamp_static_field':{'_type': 'boolean', '_default': True},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frames: list[Image.Image] = []
        self._renderer: GifRenderer | None = None
        self._color_fn = None
        self._last_fields: dict | None = None  # for static-field stamping

    # ------------------------------------------------------------------ wiring

    def inputs(self):
        # Concrete types — bare type-name `'any'` trips bigraph_schema's
        # link_path resolution (it expects a Node/dict, not a str).
        return {
            'cells':     'map[pymunk_agent]',
            'particles': 'map[pymunk_agent]',
            'fields':    {'_type': 'map'},
            'time':      'float',
        }

    # ------------------------------------------------------------------ helpers

    def _build_renderer(self, field_overlay_cfg):
        """Lazily instantiate the GifRenderer on the first update()."""
        cfg = self.config or {}
        env_w = float(cfg.get('env_width', 100.0))
        env_h = float(cfg.get('env_height', 100.0))
        env_size = max(env_w, env_h)  # GifRenderer's positional arg
        return GifRenderer(
            env_size=env_size,
            barriers=[],
            figure_size_inches=(
                float(cfg.get('figure_width', 6.0)),
                float(cfg.get('figure_height', 6.0)),
            ),
            dpi=int(cfg.get('dpi', 90)),
            show_time_title=bool(cfg.get('show_time_title', True)),
            world_pad=env_size * 0.05,
            max_line_px=int(cfg.get('max_line_px', 40)),
            xlim=(0.0, env_w),
            ylim=(0.0, env_h),
            field_overlay=field_overlay_cfg,
            env_width=env_w,
            env_height=env_h,
        )

    def _build_color_fn(self):
        cfg = self.config or {}
        mode = (cfg.get('color_mode') or '').strip().lower()
        if mode == 'pressure':
            return _color_by_pressure(pressure_max=float(cfg.get('pressure_max', 10.0)))
        if mode == 'qs_state':
            return _color_by_qs_state(threshold=float(cfg.get('qs_state_threshold', 0.5)))
        if mode == 'inclusion_body':
            return _color_by_inclusion_body(
                ib_max=float(cfg.get('ib_max', 10.0)),
                cmap_name=cfg.get('ib_cmap', 'plasma'),
            )
        return _color_uniform()

    def _build_field_overlay_cfg(self, fields):
        cfg = self.config or {}
        mol_id = cfg.get('field_mol_id') or ''
        if not mol_id:
            return None
        # Compute vmin/vmax. User-supplied wins; otherwise auto from
        # the first frame's field (typically the static initial state).
        vmin = float(cfg.get('field_vmin', 0.0))
        vmax = float(cfg.get('field_vmax', 0.0))
        if vmax <= vmin:
            arr = (fields or {}).get(mol_id)
            if arr is not None:
                a = np.asarray(arr, dtype=float)
                if a.size > 0:
                    vmin = float(a.min())
                    vmax = float(a.max())
                    if vmax <= vmin:
                        vmax = vmin + 1.0
        return {
            'mol_id': mol_id,
            'key': 'fields',
            'vmin': vmin,
            'vmax': vmax,
            'cmap': cfg.get('field_cmap', 'Greys'),
            'alpha': float(cfg.get('field_alpha', 0.35)),
        }

    # ------------------------------------------------------------------ update

    def update(self, state, interval=1.0):
        cfg = self.config or {}
        cells = state.get('cells') or {}
        particles = state.get('particles') or {}
        fields = state.get('fields') or {}
        time_val = state.get('time')

        # Stamp static fields if needed (chemotaxis-style: set at t=0
        # but never re-emitted, so per-tick state has no 'fields').
        if cfg.get('stamp_static_field', True):
            if fields:
                self._last_fields = fields
            elif self._last_fields is not None:
                fields = self._last_fields

        # Lazy renderer init — needs the first frame's fields to derive
        # vmin/vmax for the field overlay.
        if self._renderer is None:
            self._renderer = self._build_renderer(
                self._build_field_overlay_cfg(fields)
            )
            self._color_fn = self._build_color_fn()

        # Build the per-frame dict GifRenderer expects: an `agents`
        # layer (cells + particles merged) plus `fields` and `time`.
        merged_agents = {}
        merged_agents.update(cells)
        merged_agents.update(particles)
        frame = {
            'agents': merged_agents,
            'fields': fields,
            'time': time_val if time_val is not None else 0.0,
        }
        pil = self._renderer.draw_frame(
            frame, 'agents', self._color_fn,
            max_radius_px=40,
        )
        # GifRenderer.draw_frame returns the same PIL Image each call
        # (backed by the persistent canvas); copy so each frame is
        # independent before quantize.
        self._frames.append(pil.copy())

        # Bound memory & encode time.
        max_frames = int(cfg.get('max_frames', 200))
        if len(self._frames) > max_frames:
            self._frames = self._frames[-max_frames:]

        # Encode accumulated frames as animated GIF.
        # Same quantize + disposal pattern as simulation_to_gif.
        palette = [
            f.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            for f in self._frames
        ]
        buf = io.BytesIO()
        palette[0].save(
            buf, format='GIF',
            save_all=True,
            append_images=palette[1:],
            duration=int(cfg.get('frame_duration_ms', 100)),
            loop=0,
            optimize=False,
            disposal=2,
        )
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        html = (
            f'<img src="data:image/gif;base64,{b64}" '
            f'style="max-width:100%;height:auto;display:block;" '
            f'alt="multibody run"/>'
        )
        return {'html': html}


# ---------------------------------------------------------------- composite helpers


def make_viz_stores() -> dict:
    """Return a ``stores`` map entry that holds the viz HTML output.

    Use as the value of a top-level ``stores`` key in a composite document
    paired with :func:`make_multibody_viz_step`.
    """
    return {'_type': 'map[string]', 'viz_html': ''}


def make_multibody_viz_step(
    *,
    title: str,
    env_width: float,
    env_height: float,
    field_mol_id: str = '',
    figure_width: float = 6.0,
    figure_height: float = 6.0,
    color_mode: str = '',
    pressure_max: float = 10.0,
    qs_state_threshold: float = 0.5,
    ib_max: float = 10.0,
    has_particles: bool = True,
    frame_duration_ms: int = 100,
) -> dict:
    """Return a `MultibodyVizStep` Step spec wired against the conventional
    paths (`cells`, `particles`, `fields`, `global_time` → `stores.viz_html`).
    """
    inputs = {
        'cells':  ['cells'],
        'fields': ['fields'],
        'time':   ['global_time'],
    }
    if has_particles:
        inputs['particles'] = ['particles']
    return {
        '_type': 'step',
        'address': 'local:MultibodyVizStep',
        'config': {
            'title': title,
            'env_width': float(env_width),
            'env_height': float(env_height),
            'figure_width': float(figure_width),
            'figure_height': float(figure_height),
            'field_mol_id': field_mol_id,
            'color_mode': color_mode,
            'pressure_max': float(pressure_max),
            'qs_state_threshold': float(qs_state_threshold),
            'ib_max': float(ib_max),
            'frame_duration_ms': int(frame_duration_ms),
        },
        'inputs': inputs,
        'outputs': {
            'html': ['stores', 'viz_html'],
        },
    }


# Per-cell mass-over-time line plot. Lives in its own module so the
# GifRenderer-based MultibodyVizStep stays focused on the streaming-GIF
# concern.
from multi_cell.visualizations.cell_mass_traces import CellMassTraces  # noqa: E402,F401


def make_cell_mass_traces_step(
    *,
    title: str = 'cell mass over time',
    field: str = 'mass',
    show_legend: bool = False,
    xlabel: str = 'time',
    ylabel: str = 'mass',
    figsize_w: float = 8.0,
    figsize_h: float = 4.0,
) -> dict:
    """Return a `CellMassTraces` Step spec wired against the conventional
    paths (`cells`, `global_time` → `stores.viz_cell_mass_html`).

    Pairs with :func:`make_viz_stores` (which already provides a string
    slot for `viz_html`; the cell-mass output uses a sibling slot so the
    multibody GIF and the mass-trace PNG can coexist in the same composite).
    """
    return {
        '_type': 'step',
        'address': 'local:CellMassTraces',
        'config': {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'field': field,
            'show_legend': show_legend,
            'figsize_w': float(figsize_w),
            'figsize_h': float(figsize_h),
        },
        'inputs': {
            'cells': ['cells'],
            'time':  ['global_time'],
        },
        'outputs': {
            'html': ['stores', 'viz_cell_mass_html'],
        },
    }
