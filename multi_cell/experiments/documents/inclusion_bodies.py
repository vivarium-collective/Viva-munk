"""inclusion_bodies — IB aggregation with asymmetric segregation,
growth inhibition, and soft bending bodies.

A single cell is seeded in an open chamber and builds a colony over 8 h
post-induction. Each cell accumulates an inclusion-body (IB) aggregate
(size tracked in nm, logistic growth toward ib_max_nm). IB burden
inhibits growth: µ_eff = µ · max(floor, 1 − burden · IB / IB_max),
multiplied by a mechanical-pressure term exp(−p / pressure_k). At
division, the full IB is transferred to the old-pole daughter; the
new-pole daughter is rejuvenated and measurably out-grows its sibling.

Cells are modeled as multi-segment bending capsules (pivot joints +
damped rotary springs) so they flex under mechanical load from
neighbors and walls as the colony packs.

Ported from the vivarium 1.0 inclusion-body composite:
    https://github.com/vivarium-collective/inclusion-body
"""
import math

from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import build_microbe, make_rng
from multi_cell.processes.inclusion_body import make_ib_colony_process
from multi_cell.processes.pressure import make_pressure_process


def inclusion_bodies_document(config=None):
    config = config or {}
    env_size = config.get('env_size', 40)
    interval = config.get('interval', 30.0)
    growth_rate = config.get('growth_rate', 0.000289)  # ln(2)/2400 ≈ 40 min
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
    n_cells = config.get('n_cells', 1)
    division_threshold = config.get(
        'division_threshold',
        density * (2 * cell_radius) * (cell_length * 2.0),
    )

    # Bending-body config
    n_bending_segments = config.get('n_bending_segments', 4)
    bending_stiffness = config.get('bending_stiffness', 14.0)
    bending_damping = config.get('bending_damping', 5.0)
    jitter = config.get('jitter_per_second', 0.0)

    # Pressure feedback
    pressure_k = config.get('pressure_k', 2.5)
    contact_slack = config.get('contact_slack', 0.2)
    pressure_scale = config.get('pressure_scale', 1.0)

    # Seed cells in the center of the chamber via build_microbe so the
    # multi-segment bending body is built correctly by PymunkProcess.
    rng = make_rng(config.get('seed', 7))
    cells = {}
    for i in range(n_cells):
        cx = env_size / 2.0 + rng.uniform(-1.0, 1.0)
        cy = env_size / 2.0 + rng.uniform(-1.0, 1.0)
        aid, cell = build_microbe(
            rng, env_size,
            agent_id=f'cell_{i}',
            x=cx, y=cy,
            angle=rng.uniform(0, math.pi),
            length=cell_length,
            radius=cell_radius,
            density=density,
            velocity=(0.0, 0.0),
            speed_range=(0.0, 0.0),
        )
        cell.setdefault('inclusion_body', 0.0)
        cells[aid] = cell

    ib_colony = make_ib_colony_process(
        agents_key='cells',
        interval=interval,
        config={
            'growth_rate': growth_rate,
            'threshold': division_threshold,
            'ib_formation_rate': config.get('ib_formation_rate', 0.05),
            'ib_growth_rate':    config.get('ib_growth_rate', 0.0005),
            'ib_max_nm':         config.get('ib_max_nm', 800.0),
            'ib_burden_coef':    config.get('ib_burden_coef', 0.6),
            'growth_rate_floor': config.get('growth_rate_floor', 0.15),
            'pressure_k':        pressure_k,
        },
    )

    return {
        'cells': cells,
        'particles': {},
        'ib_colony': ib_colony,
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'jitter_per_second': jitter,
                'n_bending_segments': n_bending_segments,
                'bending_stiffness': bending_stiffness,
                'bending_damping': bending_damping,
            },
            'interval': interval,
            'inputs': {
                'bending_cells': ['cells'],
                'circle_particles': ['particles'],
            },
            'outputs': {
                'bending_cells': ['cells'],
                'circle_particles': ['particles'],
            },
        },
        'pressure': make_pressure_process(
            agents_key='cells',
            contact_slack=contact_slack,
            pressure_scale=pressure_scale,
            env_size=env_size,
            wall_weight=2.0,
        ),
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'particles': ['particles'],
            'time': ['global_time'],
        }),
    }
