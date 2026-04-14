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
    chemotaxis_document,
    inclusion_bodies_document,
    quorum_sensing_document,
)


EXPERIMENT_REGISTRY = {
    'daughter_machine': {
        'document': daughter_machine_document,
        'time': 28800.0,  # 8 hours = ~12 generations
        'config': {
            'env_size': 40,
            'scale_bar': {
                'size': 10.0,
                'label': '10 µm',
                'loc': 'lower right',
                'fontsize': 11,
            },
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
            'scale_bar': {
                'size': 5.0,
                'label': '5 µm',
                'loc': 'lower right',
                'fontsize': 11,
            },
        },
        'description': 'E. coli-scale cells seeded at the bottom of narrow dead-end channels (~1.5 um wide). Cells grow vertically and divide; daughters are pushed upward. Cells crossing the flow channel (top) are removed from the simulation.',
    },
    'with_particles': {
        'document': biofilm_document,
        'time': 14400.0,  # 4 hours
        'config': {
            'env_size': 60,
            'n_cells': 5,
            'n_initial_particles': 400,
            # Scale-free particle sizes: log-uniform from 0.1 to 5.0 μm
            # gives ~5 octaves of size, so each octave of radii has roughly
            # equal density and the chamber holds a few large boulders
            # mixed with many small grains.
            'particle_radius_min': 0.1,
            'particle_radius_max': 5.0,
            'particle_radius_dist': 'log_uniform',
            'scale_bar': {
                'size': 10.0,
                'label': '10 µm',
                'loc': 'lower right',
                'fontsize': 11,
            },
        },
        'description': 'Multiple E. coli-scale cells grow and divide in an environment seeded with passive particles whose sizes follow a scale-free (log-uniform) distribution. A few large boulders share the chamber with many small grains; cells push and rearrange them as they grow.',
    },
    'bending_pressure': {
        'document': bending_pressure_document,
        'time': 28800.0,  # 8 hours
        'config': {
            'env_size': 30,
            'n_cells': 1,
            'contact_slack': 0.2,
            'pressure_scale': 1.0,
            # Lower pressure_k → stronger inhibition (rate · exp(-p / k)).
            # Tuned down from 5.0 so the colony stops expanding once cells
            # reach high contact pressure, instead of slowly creeping past
            # what the chamber should hold.
            'pressure_k': 2.5,
            'n_bending_segments': 4,
            'bending_stiffness': 14.0,   # a bit stiffer — less bending
            'bending_damping': 5.0,
            'color_by_pressure': True,
            'pressure_max_visual': 8.0,
            'scale_bar': {
                'size': 5.0,
                'label': '5 µm',
                'loc': 'lower right',
                'fontsize': 11,
            },
        },
        'description': 'A single cell grows into a colony of multi-segment bending capsules. A Pressure step computes per-cell mechanical pressure from neighbor and wall contacts; GrowDivide inhibits growth as rate * exp(-pressure / pressure_k). Cells under high mechanical pressure grow more slowly AND visibly bend, so the colony shows both compositional inhibition (red interior) and physical deformation.',
    },
    'glucose_growth': {
        'document': glucose_growth_document,
        'time': 28800.0,  # 8 hours — long enough for glucose to deplete and growth to halt
        'config': {
            'env_size': 48,            # μm
            'n_bins': (16, 16),        # 3 μm per bin (each cell fits in one bin)
            'n_cells': 3,
            'glucose_init': 5.0,       # mM
            'glucose_km': 0.5,         # mM (Monod K_s)
            'glucose_diffusion': 0.05, # μm²/s
            'nutrient_yield': 0.025,
            'field_overlay': {
                'mol_id': 'glucose',
                'vmin': 0.0, 'vmax': 5.0,
                'cmap': 'YlGn',          # pale yellow → deep green
                'alpha': 0.7,
                'colorbar': True,
                'colorbar_label': 'glucose (mM)',
            },
            'scale_bar': {
                'size': 10.0,
                'label': '10 µm',
                'loc': 'lower right',
                'fontsize': 11,
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
            'scale_bar': {
                'size': 10.0,
                'label': '10 µm',
                'loc': 'lower right',
                'fontsize': 11,
            },
        },
        'description': 'Cells start with adhesin molecules that let them attach to the bottom surface. When a cell touches the surface and carries enough adhesins, a PivotJoint pins it in place. Adhesins split between daughters at division, so descendants of an attached lineage gradually exhaust the pool and the youngest cells eventually fail to attach.',
    },
    'chemotaxis': {
        'document': chemotaxis_document,
        'time': 3600.0,  # 60 minutes
        # interval=0.1s over 3600s = 36k emitter ticks; thin to ~3.6k rows.
        # The stored `step` column preserves the real composite tick number.
        'emitter_subsample': 10,
        'config': {
            'env_width': 1500.0,        # µm — long axis
            'env_height': 250.0,        # µm — short axis (6:1 aspect)
            'n_bins': (300, 50),        # 5 µm per bin
            'n_cells': 12,
            'cell_radius': 0.5,         # real bacterial size
            'cell_length': 2.0,
            'interval': 0.1,            # = tumble_duration → 1-tick tumbles
            # Long-tail exponential decay so cells can sense the gradient
            # from anywhere in the chamber, not just near the wall.
            #   c(x) = peak · exp(-x / L)
            #   peak = 20 µM, L = 600 µm  → at x=1500 c ≈ 1.6 µM
            'gradient_type': 'exponential_x',
            'ligand_peak': 20.0,        # µM at the high (left) wall
            'ligand_decay_length': 600.0,  # µm — 1/e distance
            # Run/tumble behavior — matches the spec:
            #   v = 20 µm/s, λ₀ = 1.0/s, k = 2.0 s/µM, τ_mem = 3.0 s,
            #   tumble = 0.1 s, turn ~ Normal(0, 68°)
            'run_speed': 20.0,
            'baseline_tumble_rate': 1.0,
            'sensitivity': 2.0,
            'tau_memory': 3.0,
            'tumble_rate_min': 0.1,
            'tumble_rate_max': 5.0,
            'tumble_duration': 0.1,
            'tumble_angle_sigma': 1.187,
            # Pymunk: damping=1.0 means no damping. We own body.velocity
            # directly via motile_speed each substep.
            'damping_per_second': 1.0,
            'angular_damping_per_second': 1.0,
            # Renderer viewport + figure shape: match the rectangular
            # chamber so the gif isn't squished into a square or
            # collapsed to a thin strip.
            'xlim': (0.0, 1500.0),
            'ylim': (0.0, 250.0),
            'figure_size_inches': (14.0, 3.0),
            # Trace each cell's path across frames so the swimming
            # trajectories unfold visibly as the gif plays.
            'draw_trails': True,
            'trail_alpha': 0.85,
            'trail_linewidth': 0.7,
            # Fade trails so the most recent few frames stand out and
            # older positions dissolve into the background. trail_fade_frames
            # is the e-folding age (in frames) for alpha decay; trail_max_frames
            # caps the history length so faint old segments are eventually dropped.
            'trail_fade_frames': 5.0,
            'trail_max_frames': 30,
            # Bump tiny cells up to a visible minimum on-screen width.
            'min_cell_px': 4.0,
            # 200 µm scale bar in the lower-right corner.
            'scale_bar': {
                'size': 200.0,
                'label': '200 µm',
                'loc': 'lower right',
                'fontsize': 12,
            },
            # The ligand field is static — no point paying the per-tick
            # emit cost. It's recoverable from composite_config if a future
            # change wants to re-enable a one-shot background overlay.
        },
        'description': 'A dozen tiny non-growing cells perform memory-based run/tumble chemotaxis in a long, narrow chamber (1500 × 250 µm) with a static exponential ligand gradient (peak 20 µM at x=0, 1/e decay length 600 µm) so cells can sense the gradient even from the far right wall. Each cell maintains a one-variable smoothed memory c_memory of its local concentration (exponential moving average, τ = 3 s) and computes dc/dt_smoothed = (c_now − c_memory)/τ. Its tumble rate is λ = λ₀ · exp(−k · dc/dt_smoothed), clamped to [0.1, 5.0]/s, with λ₀ = 1.0/s and k = 2.0 s/µM. RUN: cell swims at exactly 20 µm/s in its current heading. TUMBLE: cell stops for 0.1 s and then turns by Normal(0°, 68°) relative to its current heading. PymunkProcess sets body.velocity directly each substep so the swimming speed is exact, and seeded cells swim left up the gradient.',
    },
    'inclusion_bodies': {
        'document': inclusion_bodies_document,
        'time': 36000.0,  # 10 hours post-induction
        'config': {
            'env_size': 40,
            'n_cells': 1,
            'interval': 30.0,
            # hGH-like slow-growing IB: nucleates and grows continuously,
            # reaching ~800 nm after 4 h and plateauing thereafter.
            # (For asparaginase-like: set ib_max_nm=150, ib_formation_rate=0.2,
            # ib_growth_rate=0.001 — saturates almost immediately.)
            'ib_formation_rate': 0.05,     # nm/s nucleation seed
            'ib_growth_rate':    0.0005,   # 1/s on existing IB
            'ib_max_nm':         800.0,    # plateau size
            'ib_burden_coef':    0.6,      # up to 60% growth slowdown at IB_max
            'growth_rate_floor': 0.15,     # never below 15% of baseline µ
            # Bending-body physics: cells are soft multi-segment capsules.
            'n_bending_segments': 4,
            'bending_stiffness':  14.0,
            'bending_damping':    5.0,
            # Mechanical-pressure feedback: rate · exp(-p / pressure_k).
            'pressure_k':        2.5,
            'contact_slack':     0.2,
            'pressure_scale':    1.0,
            'color_by_inclusion_body': True,
            'inclusion_body_max_visual': 800.0,
            'inclusion_body_cmap': 'plasma',
            'inclusion_body_colorbar_label': 'IB size (nm)',
            'inclusion_body_colorbar_width_frac': 0.14,
            'scale_bar': {
                'size': 10.0,
                'label': '10 µm',
                'loc': 'lower right',
                'fontsize': 11,
            },
        },
        'description': 'Inclusion bodies (IBs) are dense aggregates of misfolded protein that form at the cell pole during heterologous protein expression in E. coli. This experiment grows a colony for 10 h while each cell accumulates an IB (size tracked in nanometers, logistic growth toward an 800 nm plateau — hGH-like slow-growing regime). Aggregation imposes a metabolic burden that slows growth proportionally to IB size. At division the full IB goes to one daughter (old-pole lineage) while the other starts clean (new-pole), so IB-free daughters visibly out-grow their IB-laden siblings. Cells are multi-segment bending capsules, and a Pressure process adds a second mechanical slowdown as the colony packs. Cells are colored by IB size (plasma colormap, 0–800 nm).',
    },
    'quorum_sensing': {
        'document': quorum_sensing_document,
        # 15-minute window. With D = 2 µm²/s the AI field takes
        # several hundred seconds to spread across the chamber, so the
        # off phase, the per-cluster ignition, and the build-up of the
        # density-structured steady-state AI field all fit inside it.
        'time': 900.0,
        'config': {
            # Larger chamber hosts multiple distinct colonies at varied
            # densities and leaves empty regions between them so the AI
            # field has a non-trivial spatial structure.
            'env_size': 100.0,
            # Very fine lattice: dx = 0.625 µm, just above a bacterial
            # radius, so each cell samples roughly its own bin and local
            # gradients resolve around individual cells.
            'n_bins': (160, 160),
            'interval': 1.0,
            # Spatially heterogeneous placement: a mix of dense hotspots,
            # medium clusters, a loose cluster, and a sparse scatter
            # across the whole chamber. Each entry is a Gaussian
            # (cx, cy, sigma, count); sigma=None means uniform over the
            # full chamber (the sparse background).
            'clusters': [
                {'cx': 30.0, 'cy': 30.0, 'sigma': 3.5, 'count': 180},  # dense
                {'cx': 72.0, 'cy': 30.0, 'sigma': 6.0, 'count': 110},  # medium
                {'cx': 30.0, 'cy': 72.0, 'sigma': 7.5, 'count':  90},  # medium-loose
                {'cx': 72.0, 'cy': 72.0, 'sigma': 12.0, 'count': 70},  # loose
                {'cx': 50.0, 'cy': 50.0, 'sigma': None, 'count': 60},  # uniform background
            ],
            # Total target is the sum of counts above (~510); used only
            # to top up if the clusters under-fill due to overlap jam.
            'n_cells': 510,
            # Realistic bacterial size: ~0.5 µm.
            'cell_radius': 0.5,
            # Autoinducer kinetics in realistic units (nM). LuxR-type
            # receivers switch around a few nM AHL.
            # Kinetics chosen to make activation density-dependent.
            # With bulk decay k_bulk = 0.5/s and diffusion D = 2 µm²/s
            # the spatial coupling length λ = √(D / k_bulk) ≈ 2 µm.
            # An isolated cell's own AI escapes through diffusion
            # faster than it can accumulate (D/dx² ≫ k_bulk) so
            # single cells stay well below K; cells packed inside a
            # cluster share each other's plumes across λ and build a
            # local field that crosses K, igniting the cluster. The
            # positive feedback (max = 20 vs basal = 3) keeps the
            # ignited patch locked on.
            'hill_k': 7.5,           # nM — switching threshold
            'hill_n': 4.0,           # Hill coefficient — sharp switch
            'basal_secretion': 3.0,  # nM / s at the cell bin
            'max_secretion': 20.0,   # ~7× positive feedback when induced
            # AHL diffusion: ~500 µm²/s in dilute water; much slower in
            # dense biofilm / EPS matrices. We push into the slow end
            # (2 µm²/s) so each colony stays quasi-isolated on the
            # simulation timescale — diffusion length √(D·t) after
            # 30 min is only ~60 µm, and the transverse timescale
            # between clusters (~40 µm apart) is L²/D ≈ 800 s. That
            # preserves the local density contrast instead of
            # homogenizing the chamber.
            'ai_diffusion': 2.0,     # µm² / s
            'ai_init': 0.0,
            # Open (dirichlet_ghost = 0) boundaries: the chamber is a
            # microcolony inside a much larger AI-free reservoir, so AI
            # that reaches the wall is lost to bulk. This is physically
            # meaningful and avoids the neumann pathology of unbounded
            # accumulation under constant secretion.
            'boundary_conditions': {
                'default': {
                    'x': {'type': 'dirichlet_ghost', 'value': 0.0},
                    'y': {'type': 'dirichlet_ghost', 'value': 0.0},
                },
            },
            # Uniform bulk decay of AI across the whole field
            # (FieldDecay process). Sets the spatial coupling length
            # λ = √(D / k_bulk). With D = 2 µm²/s, k_bulk = 0.3/s
            # gives λ ≈ 2.6 µm. Bulk (rather than cell-local) decay
            # applies everywhere, so a dense cluster can't flood
            # empty regions with AI above threshold — sparse cells
            # stay OFF.
            'bulk_decay_rate': 0.5,
            'degradation_rate': 0.0,
            'depth': 1.0,
            'color_by_qs_state': True,
            'qs_state_colorbar_label': 'QS state',
            'qs_state_colorbar_width_frac': 0.10,
            'qs_state_threshold': 0.5,
            'field_overlay': {
                'mol_id': 'ai',
                # PowerNorm compresses the high end so low-concentration
                # areas between colonies remain visible instead of
                # mapping to near-white. gamma=0.4 brings faint regions
                # well into the middle of the colormap.
                'norm': 'power',
                'gamma': 0.4,
                'vmin': 0.0, 'vmax': 20.0,
                'cmap': 'Purples',
                'alpha': 0.65,
                'colorbar': True,
                'colorbar_label': 'autoinducer AI (nM)',
                'width_frac': 0.08,
            },
            'scale_bar': {
                'size': 10.0,
                'label': '10 µm',
                'loc': 'lower right',
                'fontsize': 11,
            },
        },
        'description': 'A 100 µm chamber seeded with ~500 non-growing circular bacteria (0.5 µm radius, E. coli / Vibrio scale) at spatially heterogeneous density: one dense hotspot, two medium clusters, a loose cluster, and a sparse uniform scatter across the whole chamber. A diffusible autoinducer (AHL-like) field lives on a fine 160×160 lattice (dx = 0.625 µm) and is updated every 1 s. Each cell secretes AI at a small basal rate (3 nM/s at its own bin) and, when the local field sampled by CellFieldExchange crosses the Hill threshold K = 7.5 nM (n = 4), ramps secretion up ~7× via positive feedback. A FieldDecay process removes AI uniformly from every bin at k = 0.5/s (bulk-abiotic hydrolysis / background lactonase), and within each cell-bin QuorumSensing applies analytic first-order decay (stable for any k·dt). Together with AI diffusion D = 2 µm²/s these set a spatial coupling length λ = √(D/k) ≈ 2 µm: cells packed tighter than λ share AI and ignite each other past quorum, whereas isolated cells (nearest-neighbor ≫ λ) see only their own plume — which decays faster than it can build above K — and stay OFF. Result: dense clusters light up while the sparse scatter stays dark. Cells are colored by a two-state legend (blue = OFF, magenta = ON at s ≥ 0.5); a purple heatmap shows the AI concentration (nM) with PowerNorm (γ = 0.4) so faint inter-colony signal is visible alongside the bright cluster peaks.',
    },
}
