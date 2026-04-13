"""
Uniform first-order decay of one or more 2D concentration fields.

Each tick, every bin of each listed field is attenuated by
``exp(-k_i · dt)`` using a per-species rate ``k_i`` (1/s). This models
bulk abiotic degradation (e.g. hydrolysis of AHLs in aqueous media,
photodegradation, uptake by a homogeneous background community) that
is independent of the cells.

Having a non-zero bulk decay lets a signaling field reach a finite
spatial steady state even under closed boundaries and prevents AI from
accumulating globally in an open-boundary chamber long enough to flood
sparse regions at the cluster activation threshold.
"""
import math

import numpy as np
from process_bigraph import Process


class FieldDecay(Process):
    config_schema = {
        # Per-species decay rate (1/s). Species not listed are not decayed.
        'decay_rates': 'map[float]',
    }

    def inputs(self):
        return {'fields': 'map[array]'}

    def outputs(self):
        return {'fields': 'map[array]'}

    def update(self, state, interval):
        dt = float(interval)
        rates = dict(self.config.get('decay_rates', {}))
        fields = state.get('fields', {}) or {}
        out = {}
        for mol_id, arr in fields.items():
            k = float(rates.get(mol_id, 0.0) or 0.0)
            if k <= 0.0:
                continue
            a = np.asarray(arr, dtype=float)
            # Analytic decay: new = c · exp(-k·dt). We emit the delta
            # so CellFieldExchange / DiffusionAdvection's accumulating
            # schema adds it rather than replacing.
            factor = math.exp(-k * dt)
            delta = a * (factor - 1.0)
            out[mol_id] = delta
        return {'fields': out}


def make_field_decay_process(decay_rates, fields_key='fields', interval=30.0):
    """Build a FieldDecay process spec keyed off a top-level fields map."""
    return {
        '_type': 'process',
        'address': 'local:FieldDecay',
        'config': {'decay_rates': dict(decay_rates)},
        'interval': float(interval),
        'inputs': {'fields': [fields_key]},
        'outputs': {'fields': [fields_key]},
    }
