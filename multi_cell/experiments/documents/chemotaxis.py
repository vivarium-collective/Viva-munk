"""chemotaxis — non-growing cells run/tumble up a static ligand gradient."""
import math

import numpy as np
from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import build_microbe, make_rng
from multi_cell.processes.chemotaxis import add_chemotaxis_to_agents
from multi_cell.processes.cell_field_exchange import make_cell_field_exchange_process


def chemotaxis_document(config=None):
    """A dozen non-growing cells perform run/tumble chemotaxis up a 2D
    Gaussian ligand gradient.

    The ligand field is set up at t=0 as a static Gaussian peak at the
    center of the chamber and is NOT diffused (no DiffusionAdvection
    process), so it acts as a fixed attractant landscape.

    Each tick:
      1. CellFieldExchange samples each cell's local concentration into
         ``cell.local`` (cells don't write any uptake amounts back, so
         the field stays unchanged).
      2. Chemotaxis reads ``cell.local[ligand_key]``, compares it to the
         cell's previous reading, and modulates the tumble probability:
         ``tumble_rate = baseline * exp(-sensitivity * dC)``. So going up
         the gradient lowers the rate (longer runs) and going down raises
         it (more tumbles).
      3. On a "run" tick the process emits a forward thrust; on a tumble
         it emits a small thrust plus a random angular torque.
      4. PymunkProcess reads ``cell.thrust`` and ``cell.torque`` from the
         framework state and applies them as real forces to the cell body
         in its substep loop.

    Cells do NOT grow or divide here.
    """
    config = config or {}
    # Long, narrow chamber. env_width is the long axis (the gradient
    # direction); env_height is the short axis. Cells are tiny relative
    # to env_width, so they have a real distance to swim.
    env_width = float(config.get('env_width', 1000.0))    # µm
    env_height = float(config.get('env_height', 60.0))    # µm
    n_bins = config.get('n_bins', (200, 12))              # 5 µm × 5 µm bins
    interval = float(config.get('interval', 0.1))         # s — matches tumble duration
    n_cells = int(config.get('n_cells', 12))
    cell_radius = float(config.get('cell_radius', 0.5))   # µm (half-width)
    cell_length = float(config.get('cell_length', 2.0))   # µm
    density = float(config.get('density', 0.02))
    ligand_peak = float(config.get('ligand_peak', 5.0))   # µM at the high wall
    gradient_type = config.get('gradient_type', 'linear_x')
    ligand_sigma = float(config.get('ligand_sigma', env_width * 0.18))
    # Chemotaxis tuning — direct passthrough to the Chemotaxis process.
    run_speed = float(config.get('run_speed', 20.0))
    baseline_tumble_rate = float(config.get('baseline_tumble_rate', 1.0))
    sensitivity = float(config.get('sensitivity', 2.0))
    tau_memory = float(config.get('tau_memory', 3.0))
    tumble_rate_min = float(config.get('tumble_rate_min', 0.1))
    tumble_rate_max = float(config.get('tumble_rate_max', 5.0))
    tumble_duration = float(config.get('tumble_duration', 0.1))
    tumble_angle_sigma = float(config.get('tumble_angle_sigma', 1.187))
    # Pymunk damping. We override body.velocity each substep based on
    # motile_speed, so damping_per_second = 1.0 (no damping) gives the
    # cells exactly the prescribed speed without bleed-off.
    damping_per_second = float(config.get('damping_per_second', 1.0))
    angular_damping_per_second = float(config.get('angular_damping_per_second', 1.0))

    nx, ny = int(n_bins[0]), int(n_bins[1])
    xs = np.linspace(0.0, env_width, nx)
    ys = np.linspace(0.0, env_height, ny)
    X, Y = np.meshgrid(xs, ys)
    if gradient_type == 'gaussian':
        cx0 = env_width / 2.0
        cy0 = env_height / 2.0
        ligand_field = float(ligand_peak) * np.exp(
            -((X - cx0) ** 2 + (Y - cy0) ** 2) / (2.0 * ligand_sigma ** 2)
        )
    elif gradient_type == 'exponential_x':
        # High at x=0, exponential decay along +x. ligand_decay_length
        # controls how steeply the gradient falls. Cells that stray far
        # from the high wall barely sense any ligand.
        decay_length = float(config.get('ligand_decay_length', env_width * 0.18))
        ligand_field = float(ligand_peak) * np.exp(-X / max(decay_length, 1e-6))
    else:  # 'linear_x' — high at x=0, zero at x=env_width
        ligand_field = float(ligand_peak) * (1.0 - X / float(env_width))
        ligand_field = np.clip(ligand_field, 0.0, None)

    # Place cells on the LOW-concentration side (right edge) so they have
    # to swim left up the gradient. Spread vertically across the chamber.
    rng = make_rng(101)
    cells = {}
    seed_x = env_width * 0.92
    for i in range(n_cells):
        cx = seed_x + rng.uniform(-cell_radius, cell_radius)
        cy = (i + 0.5) * env_height / n_cells + rng.uniform(-1.0, 1.0)
        # Random initial body orientation
        angle = rng.uniform(-math.pi, math.pi)
        aid, cell = build_microbe(
            rng, env_width,
            agent_id=f'cell_{i}',
            x=cx, y=cy,
            angle=angle,
            length=cell_length,
            radius=cell_radius,
            density=density,
            velocity=(0, 0),
            speed_range=(0, 0),
        )
        cells[aid] = cell

    initial_state = {'cells': cells}

    # Per-cell chemotaxis process. Runs at `interval` (= tumble_duration
    # by default) so the discrete tumble timer has the right granularity.
    add_chemotaxis_to_agents(
        initial_state,
        agents_key='cells',
        ligand_key='glucose',
        interval=interval,
        config={
            'run_speed': run_speed,
            'baseline_tumble_rate': baseline_tumble_rate,
            'sensitivity': sensitivity,
            'tau_memory': tau_memory,
            'tumble_rate_min': tumble_rate_min,
            'tumble_rate_max': tumble_rate_max,
            'tumble_duration': tumble_duration,
            'tumble_angle_sigma': tumble_angle_sigma,
        },
    )

    return {
        'cells': initial_state['cells'],
        'fields': {'glucose': ligand_field},
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_width,
                'env_height': env_height,
                # No damping — we own body.velocity directly each substep
                # via the chemotaxis motile_speed pathway, and damping
                # would just bleed off our prescribed run velocity.
                'damping_per_second': damping_per_second,
                'angular_damping_per_second': angular_damping_per_second,
                'jitter_per_second': 0.0,
                'substeps': 4,
            },
            'interval': interval,
            'inputs': {
                'segment_cells': ['cells'],
            },
            'outputs': {
                'segment_cells': ['cells'],
            },
        },
        'cell_field_exchange': make_cell_field_exchange_process(
            n_bins=(nx, ny),
            bounds=(env_width, env_height),
            depth=1.0,
            agents_key='cells',
            fields_key='fields',
            interval=interval,
        ),
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'fields': ['fields'],
            'time': ['global_time'],
        }),
    }
