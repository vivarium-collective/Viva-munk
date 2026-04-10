"""mother_machine — narrow dead-end channels with a flow channel above."""
import math

from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import build_microbe, make_rng
from multi_cell.processes.grow_divide import add_adder_grow_divide_to_agents
from multi_cell.processes.remove_crossing import make_remove_crossing_process


def mother_machine_document(config=None):
    """Cells growing in a mother machine with narrow dead-end channels and a flow channel.

    Default dimensions use E. coli proportions (units = micrometers):
      - Cell width (diameter): ~1.0 um
      - Cell length at birth: ~2.0 um
      - Cell length at division: ~4.0 um
      - Channel width: ~1.5 um (just wider than one cell)
    """
    config = config or {}

    # E. coli defaults (all in micrometers)
    cell_radius = config.get('cell_radius', 0.5)          # half-width of capsule
    cell_length = config.get('cell_length', 2.0)           # birth length
    density = config.get('density', 0.02)

    # Channel geometry
    channel_width = config.get('channel_width', 1.5)       # just wider than cell diameter
    spacer_thickness = config.get('spacer_thickness', 0.3)
    channel_height = config.get('channel_height', 20.0)     # dead-end channel depth
    flow_channel_y = config.get('flow_channel_y', channel_height)  # y above which cells are removed
    n_channels = config.get('n_channels', 6)

    env_size = config.get('env_size', None)
    if env_size is None:
        width = n_channels * (channel_width + spacer_thickness) + spacer_thickness + 2.0
        height = channel_height + 5.0  # room above channels for flow
        env_size = max(width, height)
    env_size = float(env_size)

    interval = config.get('interval', 30.0)

    # Build barriers: vertical walls creating narrow channels
    barriers = []
    x = spacer_thickness
    for i in range(n_channels + 1):
        barriers.append({
            'start': (x, 0),
            'end': (x, channel_height),
            'thickness': spacer_thickness,
        })
        x += channel_width + spacer_thickness

    # Place one cell at the bottom of each channel, oriented vertically
    rng = make_rng(42)
    cells = {}
    x = spacer_thickness + spacer_thickness / 2  # start after first wall
    for i in range(n_channels):
        channel_center_x = x + channel_width / 2
        cell_y = cell_length / 2 + cell_radius + 0.5
        aid, cell = build_microbe(
            rng, env_size,
            agent_id=f'cell_{i}',
            x=channel_center_x, y=cell_y,
            angle=math.pi / 2,
            length=cell_length,
            radius=cell_radius,
            density=density,
            velocity=(0, 0),
            speed_range=(0, 0),
        )
        cells[aid] = cell
        x += channel_width + spacer_thickness

    initial_state = {'cells': cells, 'particles': {}}

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
                'elasticity': 0.1,
                'barriers': barriers,
                'wall_thickness': spacer_thickness,
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
            crossing_y=flow_channel_y,
            agents_key='cells',
        ),
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'particles': ['particles'],
            'time': ['global_time'],
        }),
    }
