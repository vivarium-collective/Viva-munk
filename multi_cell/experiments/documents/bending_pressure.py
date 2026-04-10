"""bending_pressure — multi-segment bending cells with pressure inhibition."""
import math

from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import build_microbe, make_rng
from multi_cell.processes.grow_divide import add_grow_divide_to_agents
from multi_cell.processes.pressure import make_pressure_process


def bending_pressure_document(config=None):
    """Bending cells with pressure-inhibited growth.

    Each cell is built as a multi-segment compound body (pivot joints + damped
    rotary springs) so it can flex under load. A Pressure step computes a
    per-cell pressure proxy from neighbor and wall contacts, and GrowDivide
    scales each cell's growth rate by exp(-pressure / pressure_k). Cells under
    high mechanical pressure grow more slowly AND visibly bend.
    """
    config = config or {}
    env_size = config.get('env_size', 30)
    interval = config.get('interval', 30.0)
    growth_rate = config.get('growth_rate', 0.000578)  # ~20 min unstressed doubling
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
    n_cells = config.get('n_cells', 1)
    contact_slack = config.get('contact_slack', 0.2)
    pressure_scale = config.get('pressure_scale', 1.0)
    pressure_k = config.get('pressure_k', 1.5)
    n_segments = config.get('n_bending_segments', 4)
    stiffness = config.get('bending_stiffness', 15.0)
    damping = config.get('bending_damping', 5.0)
    jitter = config.get('jitter_per_second', 0.0)
    division_threshold = config.get('division_threshold', None)
    if division_threshold is None:
        division_threshold = density * (2 * cell_radius) * (cell_length * 2.0)

    rng = make_rng(11)
    cells = {}
    cx_mid = env_size / 2.0
    cy_mid = env_size / 2.0
    for i in range(n_cells):
        cx = cx_mid + rng.uniform(-1.0, 1.0)
        cy = cy_mid + rng.uniform(-1.0, 1.0)
        aid, cell = build_microbe(
            rng, env_size,
            agent_id=f'cell_{i}',
            x=cx, y=cy,
            angle=rng.uniform(0, math.pi),
            length=cell_length,
            radius=cell_radius,
            density=density,
            velocity=(0, 0),
            speed_range=(0, 0),
        )
        cells[aid] = cell

    initial_state = {'cells': cells, 'particles': {}}

    add_grow_divide_to_agents(
        initial_state,
        agents_key='cells',
        config={
            'agents_key': 'cells',
            'rate': growth_rate,
            'threshold': division_threshold,
            'mutate': True,
            'pressure_k': pressure_k,
        },
    )

    return {
        'cells': initial_state['cells'],
        'particles': initial_state['particles'],
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'gravity': 0.0,
                'elasticity': 0.0,
                'jitter_per_second': jitter,
                'n_bending_segments': n_segments,
                'bending_stiffness': stiffness,
                'bending_damping': damping,
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
