"""biofilm — cells in a chamber seeded with passive particles ("with_particles")."""
from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import make_initial_state
from multi_cell.processes.grow_divide import add_adder_grow_divide_to_agents


def biofilm_document(config=None):
    """Cells growing in an environment seeded with passive particles of varying sizes."""
    config = config or {}
    env_size = config.get('env_size', 30)
    interval = config.get('interval', 30.0)
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
    n_cells = config.get('n_cells', 3)

    eps_radius = config.get('eps_radius', 0.15)
    n_initial_particles = config.get('n_initial_particles', 0)
    # Particle size distribution. Default 'log_uniform' gives a scale-free
    # spread of radii — equal density per decade — so the chamber gets a few
    # large boulders mixed in with many small grains.
    particle_radius_min = config.get('particle_radius_min', eps_radius)
    particle_radius_max = config.get('particle_radius_max', eps_radius * 30)
    particle_radius_dist = config.get('particle_radius_dist', 'log_uniform')

    initial_state = make_initial_state(
        n_microbes=n_cells,
        n_particles=n_initial_particles,
        env_size=env_size,
        particle_radius_range=(particle_radius_min, particle_radius_max),
        particle_radius_dist=particle_radius_dist,
        microbe_length_range=(cell_length, cell_length),
        microbe_radius_range=(cell_radius, cell_radius),
        microbe_mass_density=density,
    )

    add_adder_grow_divide_to_agents(
        initial_state,
        agents_key='cells',
        config={
            'agents_key': 'cells',
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
