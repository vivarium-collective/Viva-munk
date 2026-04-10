"""attachment — cells with adhesin molecules pinning to the bottom surface."""
import math

from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import build_microbe, make_rng
from multi_cell.processes.grow_divide import add_grow_divide_to_agents
from multi_cell.processes.secrete_eps import add_secrete_eps_to_agents


def attachment_document(config=None):
    """Cells growing in an environment with an adhesive bottom surface.

    Each cell starts with a default adhesin count. When a cell touches the
    adhesive surface and has adhesins above the threshold, it pins in place
    via a PivotJoint to the static world body. Daughters split adhesins
    evenly, so the colony eventually saturates and produces unattached cells.
    """
    config = config or {}
    env_size = config.get('env_size', 30)
    interval = config.get('interval', 30.0)
    growth_rate = config.get('growth_rate', 0.000289)
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
    n_cells = config.get('n_cells', 4)
    initial_adhesins = config.get('initial_adhesins', 8.0)
    adhesion_threshold = config.get('adhesion_threshold', 0.5)
    division_threshold = config.get('division_threshold', None)
    if division_threshold is None:
        division_threshold = density * (2 * cell_radius) * (cell_length * 2.0)

    # Place initial cells in a small cluster near the middle of the x-axis,
    # 25% above the floor.
    rng = make_rng(7)
    cells = {}
    cy_low = env_size * 0.20
    cx_mid = env_size / 2.0
    cluster_width = max(2.0, n_cells * cell_radius * 4)
    for i in range(n_cells):
        # Spread cells around the x-midpoint within cluster_width
        cx = cx_mid + (i - (n_cells - 1) / 2.0) * (cluster_width / n_cells) + rng.uniform(-0.3, 0.3)
        cy = cy_low + rng.uniform(-0.5, 1.0)
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
            adhesins=initial_adhesins,
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
        },
    )

    # Attached cells secrete EPS particles
    add_secrete_eps_to_agents(
        initial_state,
        agents_key='cells',
        particles_key='particles',
        config={
            'secretion_rate': config.get('secretion_rate', 0.02),
            'eps_radius': config.get('eps_radius', 0.15),
            'requires_attached': True,
        },
        interval=interval,
    )

    return {
        'cells': initial_state['cells'],
        'particles': initial_state['particles'],
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'gravity': 0.0,  # no gravity — only adhesion holds cells down
                'elasticity': 0.0,
                'jitter_per_second': 0.0008,  # gentle diffusive drift
                'adhesion_enabled': True,
                'adhesion_surface': 'bottom',
                'adhesion_threshold': adhesion_threshold,
                'adhesion_distance': cell_radius,
            },
            'interval': interval,
            'inputs': {
                'segment_cells': ['cells'],
                'circle_particles': ['particles'],
            },
            'outputs': {
                'segment_cells': ['cells'],
                'circle_particles': ['particles'],
            },
        },
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'particles': ['particles'],
            'time': ['global_time'],
        }),
    }
