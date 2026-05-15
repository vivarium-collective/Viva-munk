"""CellMassTraces — per-cell mass-over-time line plot.

Stateful Visualization Step. For each tick where a cell appears in the
``cells`` map, records (time, mass) in that cell's history. On every
``update()`` re-renders the full figure: one colored line per cell ever
seen, with an endpoint marker at the cell's last sampled time. When a
cell is removed from the simulation (no longer in ``cells``), its trace
simply ends at the last point we have for it — so the user can see
lineage start/end events on the chart.

Wire example::

    "viz_cell_mass": {
        "_type": "step",
        "address": "local:CellMassTraces",
        "config": {"field": "mass", "show_legend": False},
        "inputs": {
            "cells": ["cells"],
            "time":  ["global_time"],
        },
        "outputs": {"html": ["viz_cell_mass_html"]},
    }
"""
from __future__ import annotations
import base64
import colorsys
import hashlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pbg_superpowers.visualization import Visualization


def _color_by_id(cell_id) -> tuple[float, float, float]:
    """Deterministic RGB triple from a cell id (golden-ratio HSV hop)."""
    h = int(hashlib.sha1(str(cell_id).encode("utf-8")).hexdigest(), 16)
    # Hue jittered by the golden ratio for max visual separation.
    hue = (h % 1000) / 1000.0
    sat = 0.65
    val = 0.85
    return colorsys.hsv_to_rgb(hue, sat, val)


class CellMassTraces(Visualization):
    """Per-cell mass-over-time line plot. Configurable cell field (default 'mass')."""

    config_schema = {
        "title": {"_type": "string", "_default": "cell mass over time"},
        "xlabel": {"_type": "string", "_default": "time"},
        "ylabel": {"_type": "string", "_default": "mass"},
        "field": {"_type": "string", "_default": "mass"},
        "show_legend": {"_type": "boolean", "_default": False},
        "linewidth": {"_type": "float", "_default": 1.2},
        "alpha": {"_type": "float", "_default": 0.85},
        "figsize_w": {"_type": "float", "_default": 8.0},
        "figsize_h": {"_type": "float", "_default": 4.0},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # cell_id → list[(time, value)]
        self._history: dict[str, list[tuple[float, float]]] = {}
        self._step = 0

    def inputs(self):
        return {
            "cells": "map[pymunk_agent]",
            "time": "float",
        }

    def update(self, state):
        cells = state.get("cells") or {}
        t = state.get("time")
        if t is None:
            t = float(self._step)
        else:
            t = float(t)
        self._step += 1

        field = (self.config or {}).get("field") or "mass"
        for cell_id, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            v = cell.get(field)
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            self._history.setdefault(cell_id, []).append((t, v))
        return {"html": self._render()}

    def _render(self) -> str:
        if not self._history:
            return '<p style="color:#888;padding:8px">No cell data yet</p>'

        cfg = self.config or {}
        title = cfg.get("title", "cell mass over time")
        xlabel = cfg.get("xlabel", "time")
        ylabel = cfg.get("ylabel", "mass")
        show_legend = bool(cfg.get("show_legend", False))
        linewidth = float(cfg.get("linewidth", 1.2))
        alpha = float(cfg.get("alpha", 0.85))
        figsize = (float(cfg.get("figsize_w", 8.0)),
                   float(cfg.get("figsize_h", 4.0)))

        fig, ax = plt.subplots(figsize=figsize)
        for cell_id, points in sorted(self._history.items(), key=lambda kv: str(kv[0])):
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            color = _color_by_id(cell_id)
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha,
                    label=str(cell_id))
            # Endpoint marker — makes lineage termination (cell removal)
            # visible as a dot at the trace end.
            ax.plot(xs[-1], ys[-1], "o", color=color,
                    markersize=4, alpha=alpha)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25, linestyle=":")

        if show_legend and len(self._history) <= 25:
            ax.legend(
                fontsize=7, loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
            )

        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001 — tight_layout occasionally complains
            pass

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        buf.close()
        return (
            '<div style="text-align:center;padding:8px">'
            f'<img alt="{title}" src="data:image/png;base64,{b64}" '
            'style="max-width:100%;height:auto;border:1px solid #e5e7eb;'
            'border-radius:4px" />'
            "</div>"
        )

    @classmethod
    def demo(cls):
        return {
            "cells": {
                "c0": {"id": "c0", "mass": 1.0},
                "c1": {"id": "c1", "mass": 0.8},
            },
            "time": 0.0,
        }

    @classmethod
    def is_visualization(cls) -> bool:
        return True


CellMassTraces.__pb_kind__ = "visualization"
CellMassTraces.__pb_aliases__ = ["CellMassTraces"]
