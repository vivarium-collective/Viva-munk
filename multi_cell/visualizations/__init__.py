"""Visualization Step subclasses for viva-munk composite docs.

Follows the pbg-superpowers Visualization convention: each subclass
consumes per-step state via wires (like an emitter), accumulates frames
internally if it wants animation, and returns ``{'html': '<rendered>'}``
each step. The composite spec wires the input ports to store paths and
the ``html`` output to a string store.

For now a single ``MultibodyVizStep`` handles all 9 viva-munk composites
— each one configures it with its own env dimensions and (optionally) a
``field_overlay`` for static heatmaps. The spike renders the latest
frame as inline PNG; animated GIF and fancier per-composite styling
(phylogeny colors, pressure heatmap, etc.) can be layered on later.
"""
from __future__ import annotations

import base64
import io
import math

import matplotlib
matplotlib.use("Agg")  # headless render — no display required
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pbg_superpowers.visualization import Visualization


class MultibodyVizStep(Visualization):
    """Render the latest frame of a multibody composite as inline PNG.

    Inputs (all optional, missing keys render empty):
      cells   — map of pymunk_agent dicts (each has location, radius,
                length, angle, type ∈ {circle, segment})
      fields  — map of 2D numpy arrays; if ``field_overlay`` config
                names one, it's drawn as a grayscale heatmap underneath
                the cells
      time    — scalar global time, drawn in the title

    Config:
      title         — figure title
      env_width     — chamber x extent (data units)
      env_height    — chamber y extent (data units)
      field_overlay — optional dict {'mol_id': str, 'cmap': str, 'alpha': float}
      figure_size   — [width_in, height_in]; default 6×6
      dpi           — default 90
    """

    config_schema = {
        'title':         {'_type': 'string', '_default': ''},
        'env_width':     {'_type': 'float',  '_default': 100.0},
        'env_height':    {'_type': 'float',  '_default': 100.0},
        'figure_width':  {'_type': 'float',  '_default': 6.0},
        'figure_height': {'_type': 'float',  '_default': 6.0},
        'dpi':           {'_type': 'integer', '_default': 90},
        'field_mol_id':  {'_type': 'string', '_default': ''},
        'field_cmap':    {'_type': 'string', '_default': 'Greys'},
        'field_alpha':   {'_type': 'float',  '_default': 0.35},
        'cell_color':    {'_type': 'string', '_default': '#3399cc'},
    }

    def inputs(self):
        # Concrete types — using bare type-name `'any'` trips
        # bigraph_schema's link_path resolution (the str doesn't have
        # an `_link_path` slot to mutate).
        return {
            'cells':     'map[pymunk_agent]',
            'particles': 'map[pymunk_agent]',  # circle particles (EPS, etc.)
            'fields':    {'_type': 'map'},  # map of arrays; value type left open
            'time':      'float',
        }

    def update(self, state, interval=1.0):
        cells = state.get('cells') or {}
        particles = state.get('particles') or {}
        fields = state.get('fields') or {}
        time_val = state.get('time')

        cfg = self.config or {}
        env_w = float(cfg.get('env_width', 100.0))
        env_h = float(cfg.get('env_height', 100.0))
        title = cfg.get('title') or ''
        fig_w = float(cfg.get('figure_width', 6.0))
        fig_h = float(cfg.get('figure_height', 6.0))
        dpi = int(cfg.get('dpi', 90))
        mol_id = cfg.get('field_mol_id') or ''
        cell_color = cfg.get('cell_color') or '#3399cc'

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, env_w)
        ax.set_ylim(0, env_h)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Field overlay (optional)
        if mol_id and mol_id in fields:
            arr = np.asarray(fields[mol_id])
            if arr.ndim == 2 and arr.size > 0:
                ax.imshow(
                    arr.T if arr.shape[0] > arr.shape[1] else arr,
                    extent=(0, env_w, 0, env_h),
                    origin='lower',
                    cmap=cfg.get('field_cmap', 'Greys'),
                    alpha=float(cfg.get('field_alpha', 0.35)),
                    aspect='auto',
                )

        # Cells: circles drawn as patches; segments as line collections
        segments = []
        widths = []
        particle_color = '#dddd88'  # pale yellow

        def _add_agent(aid, agent, color):
            ctype = agent.get('type')
            loc = agent.get('location')
            r = agent.get('radius')
            if loc is None or r is None:
                return
            try:
                x, y = float(loc[0]), float(loc[1])
                r = float(r)
            except (TypeError, IndexError):
                return
            if r <= 0:
                return
            if ctype == 'circle':
                ax.add_patch(plt.Circle((x, y), r, color=color, alpha=0.85))
                return
            # segment: 2-point spine, length × angle
            length = float(agent.get('length') or 0.0)
            angle = float(agent.get('angle') or 0.0)
            half = 0.5 * length
            dx, dy = math.cos(angle) * half, math.sin(angle) * half
            segments.append([(x - dx, y - dy), (x + dx, y + dy)])
            widths.append(2.0 * r)

        for aid, cell in cells.items():
            if isinstance(cell, dict):
                _add_agent(aid, cell, cell_color)
        for pid, p in particles.items():
            if isinstance(p, dict):
                _add_agent(pid, p, particle_color)

        if segments:
            # Linewidths in points; rough scale: data → pts via the y-axis ratio
            y_pixels = fig.dpi * fig_h
            y_data = env_h
            data_per_pt = y_data / max(y_pixels, 1.0) * 72.0 / fig.dpi
            lws = [max(0.5, w / max(data_per_pt, 1e-6)) for w in widths]
            lc = LineCollection(segments, linewidths=lws, colors=cell_color, alpha=0.85, capstyle='round')
            ax.add_collection(lc)

        # Title with optional time
        title_parts = []
        if title:
            title_parts.append(title)
        if time_val is not None:
            try:
                title_parts.append(f"t = {float(time_val):.2f}")
            except (TypeError, ValueError):
                pass
        if title_parts:
            ax.set_title(' — '.join(title_parts), fontsize=10)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')

        html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%;height:auto;display:block;" '
            f'alt="multibody frame"/>'
        )
        return {'html': html}


def make_viz_stores() -> dict:
    """Return a `stores` map entry that holds the viz HTML output.

    Use as the value of a top-level `stores` key in a composite document
    paired with ``make_multibody_viz_step()``.
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
    cell_color: str = '#3399cc',
    has_particles: bool = True,
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
            'cell_color': cell_color,
        },
        'inputs': inputs,
        'outputs': {
            'html': ['stores', 'viz_html'],
        },
    }
