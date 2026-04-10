"""daughter_machine — single cell in a chamber with an absorbing right wall."""
from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import make_initial_state
from multi_cell.processes.grow_divide import add_grow_divide_to_agents
from multi_cell.processes.remove_crossing import make_remove_crossing_process


def daughter_machine_document(config=None):
    """A single cell growing in an environment with an absorbing right wall.

    Cells that drift past the right boundary are removed by RemoveCrossing,
    so the colony tends to grow leftward as daughters are pushed out.
    """
    config = config or {}
    env_size = config.get('env_size', 30)
    interval = config.get('interval', 30.0)
    growth_rate = config.get('growth_rate', 0.000289)  # ln(2)/2400 ~ 40 min doubling
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
    flow_x = config.get('flow_x', env_size * 0.85)  # remove past 85% of width
    division_threshold = config.get('division_threshold', None)
    if division_threshold is None:
        division_threshold = density * (2 * cell_radius) * (cell_length * 2.0)

    initial_state = make_initial_state(
        n_microbes=1,
        n_particles=0,
        env_size=env_size,
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
                'elasticity': 0.1,
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
        'remove_crossing': make_remove_crossing_process(
            x_max=flow_x,
            agents_key='cells',
        ),
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'particles': ['particles'],
            'time': ['global_time'],
        }),
    }
