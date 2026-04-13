"""quorum_sensing — static population of circular cells couples through a
diffusible autoinducer field and turns ON as the local concentration
crosses the Hill threshold.
"""
import math

import numpy as np
from process_bigraph.emitter import emitter_from_wires

from multi_cell.processes.multibody import make_rng
from multi_cell.processes.cell_field_exchange import make_cell_field_exchange_process
from multi_cell.processes.quorum_sensing import add_quorum_sensing_to_agents
from multi_cell.processes.field_decay import make_field_decay_process


def _build_circle_cell(agent_id, x, y, radius, density):
    mass = density * math.pi * radius * radius
    return {
        'id': agent_id,
        'type': 'circle',
        'mass': float(mass),
        'radius': float(radius),
        'location': (float(x), float(y)),
        'velocity': (0.0, 0.0),
        'elasticity': 0.0,
        'qs_state': 0.0,
    }


def quorum_sensing_document(config=None):
    """A colony of non-growing circular cells couples through a diffusible
    autoinducer field.

    Each tick CellFieldExchange samples the local AI concentration onto
    each cell and applies the cell's exchange amounts back to the field
    bin. QuorumSensing reads ``cell.local[ai_key]``, writes the Hill
    activation level into ``cell.qs_state``, and accumulates secretion
    into ``cell.exchange[ai_key]``. DiffusionAdvection spreads the AI
    across the chamber. Cells are colored by their qs_state, so the
    switch-like transition from OFF (low) to ON (high) is visible as
    the population reaches quorum.
    """
    config = config or {}
    env_size = float(config.get('env_size', 40.0))
    n_bins = config.get('n_bins', (20, 20))
    interval = float(config.get('interval', 30.0))
    n_cells = int(config.get('n_cells', 40))
    cell_radius = float(config.get('cell_radius', 0.8))
    density = float(config.get('density', 0.02))

    ai_init = float(config.get('ai_init', 0.0))
    ai_diffusion = float(config.get('ai_diffusion', 1.0))

    hill_k = float(config.get('hill_k', 1.0))
    hill_n = float(config.get('hill_n', 2.5))
    basal_secretion = float(config.get('basal_secretion', 0.05))
    max_secretion = float(config.get('max_secretion', 2.0))
    degradation_rate = float(config.get('degradation_rate', 0.0))
    # Bulk (space-uniform) decay of the AI field. Without this, secreted
    # AI accumulates everywhere to a chamber-wide level set only by
    # boundary loss — enough to trigger sparse cells even when their
    # local source is tiny. A non-zero bulk decay lets sparse regions
    # stay near 0 while dense clusters still build up a local excess.
    bulk_decay_rate = float(config.get('bulk_decay_rate', 0.0))

    # Spatially heterogeneous placement: a mixture of gaussian clusters
    # spread across the chamber (each with its own center, width, and
    # count) plus a sparse scatter of loners. Clusters with small sigma
    # and many cells create dense hotspots that cross quorum locally;
    # the sparse background stays below threshold so the contrast
    # between crowded and lonely cells is visible in qs_state.
    rng = make_rng(7)
    cells = {}
    placed = []
    gap = 0.3
    min_sep_sq = (2 * cell_radius + gap) ** 2

    def _try_place(x, y):
        if not (cell_radius + 0.5 <= x <= env_size - cell_radius - 0.5):
            return False
        if not (cell_radius + 0.5 <= y <= env_size - cell_radius - 0.5):
            return False
        for (px, py) in placed:
            if (x - px) ** 2 + (y - py) ** 2 < min_sep_sq:
                return False
        aid = f'cell_{len(placed)}'
        placed.append((x, y))
        cells[aid] = _build_circle_cell(aid, x, y, cell_radius, density)
        return True

    clusters = config.get('clusters')
    if clusters is None:
        # Default landscape: a mix of one very-dense hotspot, a couple of
        # medium clusters, a loose cluster, and a sparse background.
        s = env_size
        clusters = [
            {'cx': 0.30 * s, 'cy': 0.30 * s, 'sigma': 3.0, 'count': 100},  # dense
            {'cx': 0.72 * s, 'cy': 0.28 * s, 'sigma': 5.0, 'count': 70},   # medium
            {'cx': 0.25 * s, 'cy': 0.75 * s, 'sigma': 6.0, 'count': 60},   # medium
            {'cx': 0.72 * s, 'cy': 0.72 * s, 'sigma': 9.0, 'count': 50},   # loose
            # Uniform background scatter across the whole chamber
            {'cx': 0.5 * s, 'cy': 0.5 * s, 'sigma': None, 'count': 40},
        ]

    max_attempts_per_cell = 40
    for clu in clusters:
        cx = float(clu.get('cx', env_size / 2.0))
        cy = float(clu.get('cy', env_size / 2.0))
        sigma = clu.get('sigma')
        count = int(clu.get('count', 0))
        for _ in range(count):
            for _try in range(max_attempts_per_cell):
                if sigma is None:
                    x = rng.uniform(
                        cell_radius + 0.5, env_size - cell_radius - 0.5,
                    )
                    y = rng.uniform(
                        cell_radius + 0.5, env_size - cell_radius - 0.5,
                    )
                else:
                    x = cx + rng.gauss(0.0, float(sigma))
                    y = cy + rng.gauss(0.0, float(sigma))
                if _try_place(x, y):
                    break

    if n_cells and len(placed) < n_cells:
        # Top up with background scatter if the caller asked for a
        # specific total and the clusters under-filled (can happen when
        # sigma is tight and cells jam).
        for _ in range(n_cells - len(placed)):
            for _try in range(max_attempts_per_cell):
                x = rng.uniform(cell_radius + 0.5, env_size - cell_radius - 0.5)
                y = rng.uniform(cell_radius + 0.5, env_size - cell_radius - 0.5)
                if _try_place(x, y):
                    break

    initial_state = {'cells': cells}

    nx, ny = int(n_bins[0]), int(n_bins[1])
    depth = float(config.get('depth', 1.0))
    bin_volume = (env_size / nx) * (env_size / ny) * depth

    add_quorum_sensing_to_agents(
        initial_state,
        agents_key='cells',
        ai_key='ai',
        interval=interval,
        config={
            'hill_k': hill_k,
            'hill_n': hill_n,
            'basal_secretion': basal_secretion,
            'max_secretion': max_secretion,
            'degradation_rate': degradation_rate,
            'bin_volume': bin_volume,
        },
    )

    ai_field = np.full((ny, nx), ai_init, dtype=float)

    return {
        'cells': initial_state['cells'],
        'fields': {'ai': ai_field},
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'damping_per_second': 0.1,
                'jitter_per_second': 0.0,
            },
            'interval': interval,
            'inputs': {'segment_cells': ['cells']},
            'outputs': {'segment_cells': ['cells']},
        },
        'diffusion': {
            '_type': 'process',
            'address': 'local:DiffusionAdvection',
            'config': {
                'n_bins': (nx, ny),
                'bounds': (env_size, env_size),
                'default_diffusion_rate': ai_diffusion,
                'diffusion_coeffs': {'ai': ai_diffusion},
                'advection_coeffs': {'ai': (0.0, 0.0)},
                # Boundary conditions: choose per-experiment via config.
                # 'neumann' seals the chamber (needs nonzero degradation
                # to reach steady state); 'open' (dirichlet_ghost, value=0)
                # treats the walls as a large bulk medium into which AI
                # diffuses out — physically realistic for a microcolony
                # embedded in a larger reservoir.
                'boundary_conditions': config.get('boundary_conditions', {
                    'default': {
                        'x': {'type': 'dirichlet_ghost', 'value': 0.0},
                        'y': {'type': 'dirichlet_ghost', 'value': 0.0},
                    },
                }),
            },
            'interval': interval,
            'inputs': {'fields': ['fields']},
            'outputs': {'fields': ['fields']},
        },
        'cell_field_exchange': make_cell_field_exchange_process(
            n_bins=(nx, ny),
            bounds=(env_size, env_size),
            depth=depth,
            agents_key='cells',
            fields_key='fields',
            interval=interval,
        ),
        **({
            'field_decay': make_field_decay_process(
                decay_rates={'ai': bulk_decay_rate},
                fields_key='fields',
                interval=interval,
            ),
        } if bulk_decay_rate > 0.0 else {}),
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'fields': ['fields'],
            'time': ['global_time'],
        }),
    }
