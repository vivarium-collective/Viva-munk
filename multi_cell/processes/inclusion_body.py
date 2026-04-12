"""Inclusion-body growth + asymmetric segregation process.

This module exposes two processes:

``InclusionBody`` — a per-cell process that only grows the IB scalar.
Embedded on each agent via ``add_inclusion_body_to_agents``.

``IBColony`` — a single *top-level* process that walks every cell each
tick and applies (a) exponential cell growth on mass/length, (b)
division at a mass threshold with asymmetric IB segregation, and (c)
IB-mass accumulation (constant formation + exponential term).

``IBColony`` exists as a workaround: the standard per-cell
``GrowDivide`` process emits ``_add`` daughters carrying embedded
process specs, and the current ``Composite`` does not re-instantiate
those specs as live processes (their ``'instance'`` slot stays unset).
A top-level process avoids that gap, so cells continue to grow, divide,
and accumulate IB across many generations.

Reference: https://github.com/vivarium-collective/inclusion-body
"""
import math

from process_bigraph import Process

from multi_cell.processes.multibody import daughter_locations


def _seed_bending_polyline(loc, angle, L):
    """Return a two-point straight polyline for a fresh bending-cell daughter.

    The physics process rebuilds the polyline each tick from the actual
    multi-segment body, but seeding a straight line here avoids a one-
    frame glitch where the daughter renders as a zero-length dot.
    """
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    hx, hy = cos_a * (L / 2), sin_a * (L / 2)
    return [
        (float(loc[0] - hx), float(loc[1] - hy)),
        (float(loc[0] + hx), float(loc[1] + hy)),
    ]


class InclusionBody(Process):
    """Per-cell IB accumulator (formation + exponential growth)."""
    config_schema = {
        'agents_key':     {'_type': 'string', '_default': 'cells'},
        'formation_rate': {'_type': 'float',  '_default': 0.001},
        'growth_rate':    {'_type': 'float',  '_default': 0.0002},
    }

    def inputs(self):
        return {'agent_id': 'string', 'agents': 'map[pymunk_agent]'}

    def outputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def update(self, state, interval):
        agent_id = state['agent_id']
        agent = state['agents'].get(agent_id, {})
        if not agent:
            return {'agents': {}}
        dt = float(interval)
        ib = float(agent.get('inclusion_body', 0.0) or 0.0)
        f_rate = float(self.config['formation_rate'])
        g_rate = float(self.config['growth_rate'])
        d_ib = f_rate * dt + g_rate * ib * dt
        if d_ib <= 0.0:
            return {'agents': {}}
        return {'agents': {agent_id: {
            'type': agent.get('type', 'segment'),
            'inclusion_body': d_ib,
        }}}


def make_inclusion_body_process(agents_key='cells', interval=30.0, config=None):
    cfg = {'agents_key': agents_key}
    if config:
        cfg.update(config)
    return {
        '_type': 'process',
        'address': 'local:InclusionBody',
        'config': cfg,
        'interval': float(interval),
        'inputs': {
            'agent_id': ['id'],
            'agents': ['..', '..', agents_key],
        },
        'outputs': {
            'agents': ['..', '..', agents_key],
        },
    }


def add_inclusion_body_to_agents(
    initial_state, agents_key='cells', interval=30.0, config=None,
):
    agents = initial_state.get(agents_key, {})
    for agent in agents.values():
        agent.setdefault('inclusion_body', 0.0)
        agent['inclusion_body_proc'] = make_inclusion_body_process(
            agents_key=agents_key, interval=interval, config=config,
        )
    return initial_state


# =====================================================================
# IBColony — top-level growth + division + IB for all cells
# =====================================================================

class IBColony(Process):
    """Top-level growth + division + inclusion-body process.

    Units
    -----
    inclusion_body : cell pole aggregate diameter, **nanometers**
    ib_max_nm      : plateau aggregate size, typically 100–1000 nm
                     (e.g. asparaginase ≈ 150 nm; hGH ≈ 800 nm at 4 h)
    formation_rate : **nm/s** nucleation seed (active while IB << IB_max)
    growth_rate    : **1/s** exponential expansion on existing IB

    IB dynamics (logistic with nucleation seed):
        dIB/dt = (formation_rate + growth_rate · IB) · (1 − IB / IB_max)

    Growth inhibition: IB formation imposes a metabolic + proteostatic
    burden, so effective cell growth rate drops as IB burden rises:

        μ_eff = μ_baseline · max(μ_floor, 1 − burden_coef · IB / IB_max)

    At division, the entire IB is transferred to daughter 0 (old-pole
    lineage); daughter 1 is rejuvenated. Because growth is inhibited by
    IB, the old-pole lineage measurably out-lags the IB-free new-pole
    daughter — reproducing the asymmetric single-cell effect reported
    in the literature.
    """
    config_schema = {
        'agents_key':        {'_type': 'string', '_default': 'cells'},
        'growth_rate':       {'_type': 'float',  '_default': 0.000289},  # ln(2)/2400s, µ_baseline
        'threshold':         {'_type': 'float',  '_default': 0.08},      # division mass
        # IB size dynamics (nm)
        'ib_formation_rate': {'_type': 'float',  '_default': 0.05},      # nm/s
        'ib_growth_rate':    {'_type': 'float',  '_default': 0.0005},    # 1/s
        'ib_max_nm':         {'_type': 'float',  '_default': 800.0},     # plateau (hGH-like)
        # Growth inhibition by IB burden
        'ib_burden_coef':    {'_type': 'float',  '_default': 0.6},       # 0..1
        'growth_rate_floor': {'_type': 'float',  '_default': 0.15},      # fraction of baseline
        # Mechanical-pressure inhibition (reads agent['pressure'] set by Pressure step)
        'pressure_k':        {'_type': 'float',  '_default': 0.0},       # 0 disables
    }

    def inputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def outputs(self):
        return {'agents': 'map[pymunk_agent]'}

    def update(self, state, interval):
        agents = state.get('agents') or {}
        if not agents:
            return {'agents': {}}
        dt = float(interval)
        mu0 = float(self.config['growth_rate'])
        thr = float(self.config['threshold'])
        ib_f = float(self.config['ib_formation_rate'])
        ib_g = float(self.config['ib_growth_rate'])
        ib_max = max(1e-6, float(self.config['ib_max_nm']))
        burden_coef = float(self.config['ib_burden_coef'])
        mu_floor = float(self.config['growth_rate_floor'])
        pressure_k = float(self.config['pressure_k'])

        upd = {}
        adds = {}
        removes = []
        for aid, agent in agents.items():
            m = float(agent.get('mass', 0.0) or 0.0)
            L = float(agent.get('length', 0.0) or 0.0)
            r = float(agent.get('radius', 0.0) or 0.0)
            if m <= 0.0:
                continue

            ib = float(agent.get('inclusion_body', 0.0) or 0.0)
            ib_frac = min(1.0, ib / ib_max)

            # Growth inhibition — cells with heavy IB burden divide slower.
            mu_eff = mu0 * max(mu_floor, 1.0 - burden_coef * ib_frac)
            # Optional mechanical-pressure gating.
            if pressure_k > 0.0:
                p = float(agent.get('pressure', 0.0) or 0.0)
                if p > 0.0:
                    mu_eff *= math.exp(-p / pressure_k)
            dm = m * mu_eff * dt
            m_new = m + dm
            if L > 0.0:
                dL = L * (m_new / m) - L
                upd.setdefault(aid, {})['length'] = dL
            upd.setdefault(aid, {})['mass'] = dm
            upd[aid]['type'] = agent.get('type', 'segment')

            # IB logistic growth toward plateau size (ib_max_nm).
            d_ib = (ib_f + ib_g * ib) * max(0.0, 1.0 - ib_frac) * dt
            if d_ib > 0.0:
                upd[aid]['inclusion_body'] = d_ib

            # Division
            if m_new >= thr and L > 0.0 and r > 0.0:
                angle = float(agent.get('angle', 0.0))
                L_d = L * (m_new / m) * 0.5
                half_mass = 0.5 * m_new
                ib_mother = ib + max(0.0, d_ib)
                loc1, loc2 = daughter_locations(
                    agent, gap=r * 0.1, daughter_length=L_d, daughter_radius=r,
                )
                vx, vy = agent.get('velocity', (0.0, 0.0))
                eps = r * 0.01
                dvx = eps * math.cos(angle)
                dvy = eps * math.sin(angle)
                inherit_keys = ['elasticity', 'friction', 'angle', 'radius', 'type']
                base = {k: agent[k] for k in inherit_keys if k in agent}
                d1 = dict(base, **{
                    'type': 'segment', 'mass': half_mass, 'length': L_d, 'radius': r,
                    'angle': angle,
                    'location': (float(loc1[0]), float(loc1[1])),
                    'velocity': (float(vx + dvx), float(vy + dvy)),
                    'inclusion_body': ib_mother,  # old-pole: keeps all aggregate
                })
                d2 = dict(base, **{
                    'type': 'segment', 'mass': half_mass, 'length': L_d, 'radius': r,
                    'angle': angle,
                    'location': (float(loc2[0]), float(loc2[1])),
                    'velocity': (float(vx - dvx), float(vy - dvy)),
                    'inclusion_body': 0.0,  # new-pole: rejuvenated
                })
                if agent.get('polyline'):
                    d1['polyline'] = _seed_bending_polyline(loc1, angle, L_d)
                    d2['polyline'] = _seed_bending_polyline(loc2, angle, L_d)
                d1_id = f"{aid}_0"
                d2_id = f"{aid}_1"
                d1['id'] = d1_id
                d2['id'] = d2_id
                adds[d1_id] = d1
                adds[d2_id] = d2
                removes.append(aid)
                # Cancel the growth/IB update for the mother — she's gone.
                upd.pop(aid, None)

        out = dict(upd)
        if adds:
            out['_add'] = adds
        if removes:
            out['_remove'] = removes
        return {'agents': out}


def make_ib_colony_process(
    agents_key='cells', interval=30.0, config=None,
):
    cfg = {'agents_key': agents_key}
    if config:
        cfg.update(config)
    return {
        '_type': 'process',
        'address': 'local:IBColony',
        'config': cfg,
        'interval': float(interval),
        'inputs': {'agents': [agents_key]},
        'outputs': {'agents': [agents_key]},
    }
