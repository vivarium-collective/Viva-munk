"""Experiment registry — single source of truth for what experiments exist
and how they're configured. Each entry is a dict with keys:

    document     — function (config: dict|None) -> composite document dict
    time         — total simulation time in seconds
    config       — experiment-specific configuration kwargs
    description  — human-readable text shown in the HTML report
"""
from multi_cell.experiments.documents import (
    daughter_machine_document,
    mother_machine_document,
    biofilm_document,
    bending_pressure_document,
    glucose_growth_document,
    attachment_document,
)


EXPERIMENT_REGISTRY = {
    'daughter_machine': {
        'document': daughter_machine_document,
        'time': 28800.0,  # 8 hours = ~12 generations
        'config': {
            'env_size': 40,
        },
        'description': 'A single E. coli-scale microbe grows and divides in an open chamber with an absorbing right wall. Daughters drift toward the right boundary and are removed when they cross it, keeping the population growing without bound.',
    },
    'mother_machine': {
        'document': mother_machine_document,
        'time': 14400.0,  # 4 hours ~ 6 generations
        'config': {
            'n_channels': 32,
            'channel_height': 25.0,
            'flow_channel_y': 22.0,
        },
        'description': 'E. coli-scale cells seeded at the bottom of narrow dead-end channels (~1.5 um wide). Cells grow vertically and divide; daughters are pushed upward. Cells crossing the flow channel (top) are removed from the simulation.',
    },
    'with_particles': {
        'document': biofilm_document,
        'time': 14400.0,  # 4 hours
        'config': {
            'env_size': 60,
            'n_cells': 5,
            'n_initial_particles': 300,
        },
        'description': 'Multiple E. coli-scale cells grow and divide in an environment seeded with passive particles of varying sizes. The cells push and rearrange the particles as they grow.',
    },
    'bending_pressure': {
        'document': bending_pressure_document,
        'time': 21600.0,  # 6 hours
        'config': {
            'env_size': 30,
            'n_cells': 1,
            'contact_slack': 0.2,
            'pressure_scale': 1.0,
            # pressure_k raised so the colony grows to a comfortable size under
            # the (now-correct) inheritance of pressure_k across divisions.
            'pressure_k': 5.0,
            'n_bending_segments': 4,
            'bending_stiffness': 14.0,   # a bit stiffer — less bending
            'bending_damping': 5.0,
            'color_by_pressure': True,
            'pressure_max_visual': 8.0,
        },
        'description': 'A single cell grows into a colony of multi-segment bending capsules. A Pressure step computes per-cell mechanical pressure from neighbor and wall contacts; GrowDivide inhibits growth as rate * exp(-pressure / pressure_k). Cells under high mechanical pressure grow more slowly AND visibly bend, so the colony shows both compositional inhibition (red interior) and physical deformation.',
    },
    'glucose_growth': {
        'document': glucose_growth_document,
        'time': 7200.0,  # 2 hours
        'config': {
            'env_size': 48,            # μm
            'n_bins': (16, 16),        # 3 μm per bin (each cell fits in one bin)
            'n_cells': 3,
            'glucose_init': 5.0,       # mM
            'glucose_km': 0.5,         # mM (Monod K_s)
            'glucose_diffusion': 0.05, # μm²/s
            'nutrient_yield': 0.025,
            'growth_rate': 0.0015,
            'field_overlay': {
                'mol_id': 'glucose',
                'vmin': 0.0, 'vmax': 5.0,
                'cmap': 'YlGn',          # pale yellow → deep green
                'alpha': 0.7,
                'colorbar': True,
                'colorbar_label': 'glucose (mM)',
            },
        },
        'description': "Cells grow on a 2D glucose field. A DiffusionAdvection process spreads the glucose; CellFieldExchange runs every tick to sample the local concentration onto each cell and apply the cell's consumption back into the field bin. GrowDivide gates rate by Monod kinetics rate × glucose / (K_s + glucose), so cells slow and stop growing as their local glucose runs out. Units: μm, mM, s; bins are 3 μm so a single cell fits inside one grid site. The background heatmap shows glucose, with a colorbar on the right.",
    },
    'attachment': {
        'document': attachment_document,
        'time': 14400.0,  # 4 hours
        'config': {
            'env_size': 50,
            'n_cells': 4,
            'initial_adhesins': 8.0,
            'adhesion_threshold': 0.5,
        },
        'description': 'Cells start with adhesin molecules that let them attach to the bottom surface. When a cell touches the surface and carries enough adhesins, a PivotJoint pins it in place. Adhesins split between daughters at division, so descendants of an attached lineage gradually exhaust the pool and the youngest cells eventually fail to attach.',
    },
}
