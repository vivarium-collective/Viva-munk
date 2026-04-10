"""biofilm — cells in a chamber seeded with passive particles ("with_particles")."""
from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import make_initial_state
from multi_cell.processes.grow_divide import add_grow_divide_to_agents


def biofilm_document(config=None):
    """Cells growing in an environment seeded with passive particles of varying sizes."""
    config = config or {}
    env_size = config.get('env_size', 30)
    interval = config.get('interval', 30.0)
    growth_rate = config.get('growth_rate', 0.000289)
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
    n_cells = config.get('n_cells', 3)
    division_threshold = config.get('division_threshold', None)
    if division_threshold is None:
        division_threshold = density * (2 * cell_radius) * (cell_length * 2.0)

    eps_radius = config.get('eps_radius', 0.15)
    n_initial_particles = config.get('n_initial_particles', 0)

    initial_state = make_initial_state(
        n_microbes=n_cells,
        n_particles=n_initial_particles,
        env_size=env_size,
        particle_radius_range=(eps_radius, eps_radius * 20),
        microbe_length_range=(cell_length, cell_length),
        microbe_radius_range=(cell_radius, cell_radius),
        microbe_mass_density=density,
    )

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

    return {
        'cells': initial_state['cells'],
        'particles': initial_state['particles'],
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'gravity': 0,
                'elasticity': 0.0,
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
