"""
Cell ↔ field exchange step.

Connects pymunk_agent cells to a map of 2D concentration fields. For each
cell, it samples the field values at the cell's bin (writing them onto
cell['local']) and applies the cell's per-molecule exchange amounts to the
corresponding field bin as a concentration delta. After applying, each
cell's exchange dict is reset to zero.

Adapted from spatio_flux.processes.particles.ParticleExchange — the
differences are:
  - cells are typed as `pymunk_agent` (we read `location` not `position`)
  - the agents map is named `cells`
  - cell['local'] / cell['exchange'] are dicts the user can populate from
    other processes (e.g. uptake/secretion kinetics)
"""
import numpy as np

from process_bigraph import Process


def _bin_index(value, lo, hi, n_bins):
    """Map a physical coordinate to a bin index, clamped to [0, n_bins-1]."""
    if hi <= lo:
        return 0
    frac = (float(value) - lo) / (hi - lo)
    idx = int(frac * n_bins)
    if idx < 0:
        return 0
    if idx >= n_bins:
        return n_bins - 1
    return idx


def _sample_fields(fields, x_bin, y_bin):
    """Sample each field's value at (y_bin, x_bin) — arrays are (ny, nx)."""
    out = {}
    for mol_id, field in fields.items():
        arr = np.asarray(field)
        if arr.size == 0:
            out[mol_id] = 0.0
            continue
        if arr.ndim == 2:
            xi = int(np.clip(x_bin, 0, arr.shape[1] - 1))
            yi = int(np.clip(y_bin, 0, arr.shape[0] - 1))
            out[mol_id] = float(arr[yi, xi])
        elif arr.ndim == 1:
            xi = int(np.clip(x_bin, 0, arr.shape[0] - 1))
            out[mol_id] = float(arr[xi])
        else:
            arr2 = np.squeeze(arr)
            if arr2.ndim == 2:
                xi = int(np.clip(x_bin, 0, arr2.shape[1] - 1))
                yi = int(np.clip(y_bin, 0, arr2.shape[0] - 1))
                out[mol_id] = float(arr2[yi, xi])
            elif arr2.ndim == 1:
                xi = int(np.clip(x_bin, 0, arr2.shape[0] - 1))
                out[mol_id] = float(arr2[xi])
            else:
                raise ValueError(
                    f"Unsupported field shape {arr.shape} for {mol_id}"
                )
    return out


class CellFieldExchange(Process):
    """Per-cell sampling and deposition into a 2D concentration field map.

    Runs as a Process (not a Step) so it ticks every interval like the rest
    of the document, guaranteeing cell.local is refreshed every step and the
    field is debited every step. Mirrors spatio-flux's ParticleExchange but
    keyed to multi_cell's `pymunk_agent` cells using `location`.
    """

    config_schema = {
        # IMPORTANT: n_bins is (nx, ny) for parity with DiffusionAdvection.
        # Field arrays are still (ny, nx) == (rows, cols).
        'n_bins': 'tuple[integer{1},integer{1}]',
        # Physical bounds: (xmax, ymax). Cells live in [0, xmax] x [0, ymax].
        'bounds': 'tuple[float{1.0},float{1.0}]',
        # Vertical extent of each bin so amounts -> concentrations: ΔC = Δamt / Vbin.
        'depth': {'_type': 'float', '_default': 1.0},
        # Which agents-map key to operate on.
        'agents_key': {'_type': 'string', '_default': 'cells'},
    }

    def initialize(self, config):
        nx, ny = tuple(config['n_bins'])
        xmax, ymax = tuple(config['bounds'])
        nx = int(nx)
        ny = int(ny)
        xmax = float(xmax)
        ymax = float(ymax)
        depth = float(config.get('depth', 1.0))

        if nx <= 0 or ny <= 0:
            raise ValueError(f"n_bins must be positive, got {(nx, ny)}")
        if xmax <= 0.0 or ymax <= 0.0:
            raise ValueError(f"bounds must be positive, got {(xmax, ymax)}")
        if depth <= 0.0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.nx = nx
        self.ny = ny
        self.xmax = xmax
        self.ymax = ymax
        self.bin_volume = (xmax / nx) * (ymax / ny) * depth

    def inputs(self):
        return {
            'agents': 'map[pymunk_agent]',
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'array',
                    '_shape': self.config['n_bins'],
                    '_data': 'float',
                },
            },
        }

    def outputs(self):
        return {
            'agents': 'map[pymunk_agent]',
            'fields': {
                '_type': 'map',
                '_value': {
                    '_type': 'positive_array',
                    '_shape': self.config['n_bins'],
                    '_data': 'float',
                },
            },
        }

    def update(self, state, interval):
        agents = state.get('agents', {}) or {}
        fields = state.get('fields', {}) or {}
        if not agents or not fields:
            return {'agents': {}, 'fields': {}}

        vol = self.bin_volume

        # Pre-allocate per-field delta arrays (concentration deltas)
        field_updates = {
            mol_id: np.zeros_like(np.asarray(arr, dtype=float), dtype=float)
            for mol_id, arr in fields.items()
        }

        agent_updates = {}
        for aid, cell in agents.items():
            loc = cell.get('location')
            if loc is None:
                continue
            x_bin = _bin_index(loc[0], 0.0, self.xmax, self.nx)
            y_bin = _bin_index(loc[1], 0.0, self.ymax, self.ny)

            # Sample field values at this bin and write to cell.local
            sampled = _sample_fields(fields, x_bin, y_bin)

            # Deposit cell's exchange amounts as ΔC = Δamt / Vbin
            exchange = cell.get('exchange') or {}
            for mol_id, delta_amt in exchange.items():
                if mol_id in field_updates:
                    field_updates[mol_id][y_bin, x_bin] += float(delta_amt) / vol

            # Reset exchange to zero, set local to fresh sample
            agent_updates[aid] = {
                'local': sampled,
                'exchange': {mol_id: 0.0 for mol_id in (exchange.keys() or fields.keys())},
            }

        return {
            'agents': agent_updates,
            'fields': field_updates,
        }


def make_cell_field_exchange_process(
    n_bins, bounds, depth=1.0, agents_key='cells', fields_key='fields',
    interval=30.0,
):
    """Build a Process spec for embedding in a document."""
    return {
        '_type': 'process',
        'address': 'local:CellFieldExchange',
        'config': {
            'n_bins': tuple(n_bins),
            'bounds': tuple(bounds),
            'depth': float(depth),
            'agents_key': agents_key,
        },
        'interval': float(interval),
        'inputs': {
            'agents': [agents_key],
            'fields': [fields_key],
        },
        'outputs': {
            'agents': [agents_key],
            'fields': [fields_key],
        },
    }
