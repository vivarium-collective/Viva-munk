"""
Pressure — a Step that estimates per-cell mechanical pressure from cell-cell
and cell-wall overlaps.

For each cell pair, the step computes how much the cells' bounding circles
overlap (using effective_radius = length/2 + radius for capsules) and adds
that overlap depth to each cell's pressure. The same is done for the four
chamber walls. Overlap depth is proportional to the contact spring force in
pymunk's collision resolver, so this is a much closer proxy for actual
mechanical pressure than a simple neighbor count.

Pressure is written to each cell's `pressure` field; downstream processes
(e.g. GrowDivide) can read it.
"""
import math

from process_bigraph import Step


class Pressure(Step):
    config_schema = {
        'agents_key':      {'_type': 'string', '_default': 'cells'},
        # Cutoff slack (extra distance beyond bounding circles) for contributing pairs
        'contact_slack':   {'_type': 'float',  '_default': 0.5},
        # Multiplier applied to summed overlap depth to scale pressure
        'pressure_scale':  {'_type': 'float',  '_default': 1.0},
        # Walls of the chamber also contribute to pressure
        'env_size':        {'_type': 'float',  '_default': 0.0},  # 0 disables wall pressure
        'wall_weight':     {'_type': 'float',  '_default': 1.0},  # multiplier on wall overlap
    }

    def inputs(self):
        return {
            'agents': 'map[pymunk_agent]',
        }

    def outputs(self):
        return {
            'agents': 'map[pymunk_agent]',
        }

    def update(self, state):
        import numpy as np
        agents = state.get('agents', {}) or {}
        if not agents:
            return {'agents': {}}

        slack = float(self.config['contact_slack'])
        scale = float(self.config['pressure_scale'])
        env_size = float(self.config.get('env_size', 0.0) or 0.0)
        wall_weight = float(self.config.get('wall_weight', 1.0))

        # Vectorized snapshot of cell geometry
        ids = []
        cx, cy, rs, halfs, angs = [], [], [], [], []
        for aid, agent in agents.items():
            loc = agent.get('location')
            if loc is None:
                continue
            ids.append(aid)
            cx.append(float(loc[0]))
            cy.append(float(loc[1]))
            rs.append(float(agent.get('radius', 0.5) or 0.5))
            halfs.append(0.5 * float(agent.get('length', 0.0) or 0.0))
            angs.append(float(agent.get('angle', 0.0) or 0.0))

        n = len(ids)
        if n == 0:
            return {'agents': {}}

        cx = np.array(cx); cy = np.array(cy)
        rs = np.array(rs); halfs = np.array(halfs); angs = np.array(angs)

        # Effective radius for the bounding-circle check: cap radius + a fraction
        # of half-length. This is a fast proxy that approximates capsule contact
        # without doing a full segment-to-segment distance every pair.
        eff_r = rs + 0.5 * halfs

        # All pairs vectorized
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        contact = eff_r[:, None] + eff_r[None, :] + slack
        overlap = contact - d
        # Mask self-contacts and non-overlapping pairs
        np.fill_diagonal(overlap, 0.0)
        overlap = np.where(overlap > 0.0, overlap, 0.0)
        pair_pressure = overlap.sum(axis=1)

        pressures = pair_pressure * scale

        # Wall contribution: how much the bounding circle (around the cell center)
        # crosses each chamber wall
        if env_size > 0:
            for d_wall in (cx - eff_r,
                           env_size - cx - eff_r,
                           cy - eff_r,
                           env_size - cy - eff_r):
                contrib = np.where(d_wall < slack, wall_weight * (slack - d_wall), 0.0)
                pressures = pressures + contrib

        out = {aid: {'pressure': float(p)} for aid, p in zip(ids, pressures)}
        return {'agents': out}


def make_pressure_process(agents_key='cells', contact_slack=0.5,
                          pressure_scale=1.0, env_size=0.0, wall_weight=1.0):
    """Create a Pressure step spec to embed in a document."""
    return {
        '_type': 'step',
        'address': 'local:Pressure',
        'config': {
            'agents_key': agents_key,
            'contact_slack': contact_slack,
            'pressure_scale': pressure_scale,
            'env_size': env_size,
            'wall_weight': wall_weight,
        },
        'inputs': {
            'agents': [agents_key],
        },
        'outputs': {
            'agents': [agents_key],
        },
    }
