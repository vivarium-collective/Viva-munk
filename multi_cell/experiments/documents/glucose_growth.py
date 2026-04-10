"""glucose_growth — cells eat a 2D glucose field that diffuses and depletes."""
import math

import numpy as np
from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import build_microbe, make_rng
from multi_cell.processes.grow_divide import add_grow_divide_to_agents
from multi_cell.processes.cell_field_exchange import make_cell_field_exchange_process


def glucose_growth_document(config=None):
    """Cells grow on a glucose field, eat glucose, and stop growing when it runs out.

    A 2D glucose concentration field is initialized uniformly across the
    chamber. DiffusionAdvection diffuses it. CellFieldExchange runs as a
    Process every interval — it samples the local concentration onto each
    cell (so GrowDivide can read it) and applies the cell's exchange amounts
    back into the field bin (subtracting uptake). GrowDivide gates its rate
    by Monod S/(Km+S) on local glucose, so cells stop growing once their
    local patch is depleted.
    """
    config = config or {}
    # Spatial / temporal scales: env in μm, time in s, glucose in mM,
    # diffusion in μm²/s. Cell length 2 μm + radius 0.5 μm fits inside
    # a 3 μm bin (24 / 8 = 3).
    env_size = config.get('env_size', 24)            # μm
    n_bins = config.get('n_bins', (8, 8))             # 3 μm per bin
    interval = config.get('interval', 30.0)           # s
    growth_rate = config.get('growth_rate', 0.0015)   # 1/s
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
    n_cells = config.get('n_cells', 3)
    glucose_init = config.get('glucose_init', 5.0)    # mM
    glucose_km = config.get('glucose_km', 0.5)        # mM (Monod K_s)
    glucose_diffusion = config.get('glucose_diffusion', 0.05)  # μm²/s
    nutrient_yield = config.get('nutrient_yield', 0.025)  # cell mass per mM glucose
    division_threshold = config.get('division_threshold', None)
    if division_threshold is None:
        division_threshold = density * (2 * cell_radius) * (cell_length * 2.0)

    rng = make_rng(13)
    cells = {}
    cx_mid = env_size / 2.0
    cy_mid = env_size / 2.0
    for i in range(n_cells):
        cx = cx_mid + rng.uniform(-3.0, 3.0)
        cy = cy_mid + rng.uniform(-3.0, 3.0)
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

    initial_state = {'cells': cells}
    add_grow_divide_to_agents(
        initial_state,
        agents_key='cells',
        config={
            'agents_key': 'cells',
            'rate': growth_rate,
            'threshold': division_threshold,
            'mutate': False,
            'pressure_k': 1e9,  # disable pressure inhibition for this experiment
            'nutrient_key': 'glucose',
            'nutrient_km': glucose_km,
            'nutrient_yield': nutrient_yield,
        },
    )

    nx, ny = int(n_bins[0]), int(n_bins[1])
    glucose_field = np.full((ny, nx), float(glucose_init), dtype=float)

    return {
        'cells': initial_state['cells'],
        'fields': {'glucose': glucose_field},
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'gravity': 0.0,
                'elasticity': 0.0,
            },
            'interval': interval,
            'inputs': {
                'segment_cells': ['cells'],
            },
            'outputs': {
                'segment_cells': ['cells'],
            },
        },
        'diffusion': {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': (nx, ny),
                'bounds': (float(env_size), float(env_size)),
                'default_diffusion_rate': glucose_diffusion,
                'diffusion_coeffs': {'glucose': glucose_diffusion},
                'advection_coeffs': {'glucose': (0.0, 0.0)},
            },
            'interval': interval,
            'inputs': {'fields': ['fields']},
            'outputs': {'fields': ['fields']},
        },
        'cell_field_exchange': make_cell_field_exchange_process(
            n_bins=(nx, ny),
            bounds=(float(env_size), float(env_size)),
            # Shallow chamber: small bin volume → bigger ΔC per uptake amount,
            # so cells deplete their bin much faster.
            depth=config.get('depth', 0.1),
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
