"""Per-experiment runner: builds the document, runs the sim, captures the
GIF/viz/JSON, and returns a result dict that the report can render."""
import json
import os
import time

from bigraph_viz import plot_bigraph
from process_bigraph import Composite, gather_emitter_results

from multi_cell import core_import
from multi_cell.experiments.registry import EXPERIMENT_REGISTRY
from multi_cell.plots.multibody_plots import simulation_to_gif


# Single core per process. Each parallel worker re-imports this module and
# gets its own clean PymunkCore (no shared mutable state across workers).
PYMUNK_CORE = core_import()


def _splice_process_configs(serialized, state):
    """Walk the serialized state and graft process config dicts from the live sim state.

    `core.serialize` returns an empty dict for fields typed as opaque Node
    (like a process's `config`), even when the live state holds the populated
    dict. This walker copies the actual config values across so the JSON
    viewer shows useful content.
    """
    if not isinstance(serialized, dict) or not isinstance(state, dict):
        return
    if 'address' in serialized and 'config' in serialized:
        live_cfg = state.get('config') if isinstance(state, dict) else None
        if isinstance(live_cfg, dict) and live_cfg:
            serialized['config'] = dict(live_cfg)
    for key, sub in serialized.items():
        if isinstance(sub, dict):
            sub_state = state.get(key) if isinstance(state, dict) else None
            if isinstance(sub_state, dict):
                _splice_process_configs(sub, sub_state)


def _serialize_state(core, sim):
    """Serialize sim state to a JSON-safe dict (or None on failure)."""
    try:
        serialized = core.serialize(sim.schema, sim.state)
        _splice_process_configs(serialized, sim.state)
        # Replace inf/nan with JSON-safe values
        serialized = json.loads(
            json.dumps(serialized, default=lambda x: None if x != x else str(x))
            .replace('Infinity', 'null')
            .replace('-Infinity', 'null')
            .replace('NaN', 'null')
        )
        return serialized
    except Exception as e:
        print(f'  state JSON failed: {e}')
        return None


def _save_bigraph_viz(core, sim, name, output_dir):
    """Save a bigraph composition figure for `name`. Returns a path or None."""
    plot_state = {k: v for k, v in sim.state.items() if k not in ['global_time', 'emitter']}
    plot_schema = {k: v for k, v in sim.schema.items() if k not in ['global_time', 'emitter']}
    viz_path = os.path.join(output_dir, f'{name}_viz')
    try:
        # Only show one cell and one particle for readability
        if 'cells' in plot_state and plot_state['cells']:
            first_key = next(iter(plot_state['cells']))
            plot_state['cells'] = {first_key: plot_state['cells'][first_key]}
        if 'particles' in plot_state and plot_state['particles']:
            first_key = next(iter(plot_state['particles']))
            plot_state['particles'] = {first_key: plot_state['particles'][first_key]}
        plot_bigraph(
            state=plot_state, schema=plot_schema, core=core,
            out_dir=output_dir, filename=f'{name}_viz', dpi='200',
            collapse_redundant_processes=True,
            show_values=False,
        )
        return viz_path
    except Exception as e:
        print(f'  bigraph viz failed: {e}')
        return None


def _derive_gif_options(document, config, env_size):
    """Pull viewport, flow regions, and adhesion-surface hints out of the
    experiment's config and document so the renderer can show them."""
    multibody_cfg = document.get('multibody', {}).get('config', {})

    gif_config = {'env_size': env_size}
    if 'barriers' in multibody_cfg:
        gif_config['barriers'] = multibody_cfg['barriers']

    # Auto-derive viewport from channel geometry if not specified
    xlim = config.get('xlim', None)
    ylim = config.get('ylim', None)
    if xlim is None and 'n_channels' in config:
        n_ch = config['n_channels']
        cw = config.get('channel_width', 1.5)
        st = config.get('spacer_thickness', 0.3)
        total_w = n_ch * (cw + st) + st
        xlim = (-0.5, total_w + 0.5)
    if ylim is None and 'channel_height' in config:
        ylim = (-0.5, config['channel_height'] + 2.0)

    # Auto-derive flow regions from any RemoveCrossing step in the document
    flow_regions = []
    for val in document.values():
        if isinstance(val, dict) and val.get('address') == 'local:RemoveCrossing':
            rc_cfg = val.get('config', {})
            region = {}
            if rc_cfg.get('x_max') is not None:
                region['x_min'] = rc_cfg['x_max']
            if rc_cfg.get('x_min') is not None:
                region['x_max'] = rc_cfg['x_min']
            if rc_cfg.get('y_max') is not None:
                region['y_min'] = rc_cfg['y_max']
            if rc_cfg.get('y_min') is not None:
                region['y_max'] = rc_cfg['y_min']
            if rc_cfg.get('crossing_y') is not None and 'y_min' not in region:
                region['y_min'] = rc_cfg['crossing_y']
            if region:
                flow_regions.append(region)

    # Highlight the adhesion surface if PymunkProcess has adhesion enabled
    adhesion_surface = None
    if multibody_cfg.get('adhesion_enabled'):
        adhesion_surface = multibody_cfg.get('adhesion_surface', 'bottom')

    return gif_config, xlim, ylim, flow_regions, adhesion_surface


def run_experiment(name, output_dir='out', entry=None):
    """Run one experiment by name and return its result dict.

    Pass `entry` to bypass the global registry (useful for ad-hoc runs).
    """
    if entry is None:
        entry = EXPERIMENT_REGISTRY[name]
    doc_fn = entry['document']
    total_time = entry.get('time', 100.0)
    config = entry.get('config', {})
    env_size = config.get('env_size', 600)

    print(f'\n=== {name} ===')
    os.makedirs(output_dir, exist_ok=True)

    core = PYMUNK_CORE
    document = doc_fn(config)
    sim = Composite({'state': document}, core=core)

    viz_path = _save_bigraph_viz(core, sim, name, output_dir)
    serialized = _serialize_state(core, sim)

    t0 = time.time()
    sim.run(total_time)
    elapsed = time.time() - t0
    results = gather_emitter_results(sim)[('emitter',)]
    print(f'Completed in {elapsed:.1f}s — {len(results)} steps')

    last = results[-1]
    n_cells = len(last.get('agents', {}))
    n_particles = len(last.get('particles', {}))
    print(f'Final state: {n_cells} cells, {n_particles} particles')

    # Generate GIF — target ~150 frames
    gif_config, xlim, ylim, flow_regions, adhesion_surface = _derive_gif_options(
        document, config, env_size,
    )
    target_frames = 150
    skip = max(1, len(results) // target_frames)

    color_by_pressure = bool(config.get('color_by_pressure', False))
    color_by_inclusion_body = bool(config.get('color_by_inclusion_body', False))
    color_fn = None
    cell_colorbar = None
    if color_by_inclusion_body:
        import matplotlib.cm as _cm
        from matplotlib.colors import Normalize as _Normalize
        ib_max = float(config.get('inclusion_body_max_visual', 10.0))
        cmap_name = config.get('inclusion_body_cmap', 'plasma')
        cmap = _cm.get_cmap(cmap_name)
        norm = _Normalize(vmin=0.0, vmax=ib_max, clip=True)
        particle_rgb = (0.85, 0.85, 0.55)
        def color_fn(aid, ent=None):
            if aid.startswith(('eps_', 'p_')):
                return particle_rgb
            ib = 0.0
            if isinstance(ent, dict):
                ib = float(ent.get('inclusion_body', 0.0) or 0.0)
            r, g, b, _ = cmap(norm(ib))
            return (r, g, b)
        cell_colorbar = {
            'vmin': 0.0,
            'vmax': ib_max,
            'cmap': cmap_name,
            'label': config.get('inclusion_body_colorbar_label', 'inclusion-body size'),
            'width_frac': float(config.get('inclusion_body_colorbar_width_frac', 0.14)),
        }
    field_overlay = config.get('field_overlay', None)
    figure_size_inches = config.get('figure_size_inches', (6, 6))
    draw_trails = bool(config.get('draw_trails', False))
    trail_alpha = float(config.get('trail_alpha', 0.7))
    trail_linewidth = float(config.get('trail_linewidth', 0.6))
    trail_fade_frames = float(config.get('trail_fade_frames', 8.0))
    trail_max_frames = int(config.get('trail_max_frames', 40))
    env_width = config.get('env_width', None)
    env_height = config.get('env_height', None)
    scale_bar = config.get('scale_bar', None)
    min_cell_px = float(config.get('min_cell_px', 0.0))
    gif_path = simulation_to_gif(
        results,
        filename=name,
        config=gif_config,
        out_dir=output_dir,
        color_by_phylogeny=not (color_by_pressure or color_fn is not None),
        color_by_pressure=color_by_pressure and color_fn is None,
        color_fn=color_fn,
        pressure_max=float(config.get('pressure_max_visual', 8.0)),
        skip_frames=skip,
        frame_duration_ms=50,
        show_time_title=True,
        world_pad=env_size * 0.05,
        dpi=150,
        xlim=xlim,
        ylim=ylim,
        flow_regions=flow_regions or None,
        adhesion_surface=adhesion_surface,
        field_overlay=field_overlay,
        figure_size_inches=figure_size_inches,
        draw_trails=draw_trails,
        trail_alpha=trail_alpha,
        trail_linewidth=trail_linewidth,
        trail_fade_frames=trail_fade_frames,
        trail_max_frames=trail_max_frames,
        env_width=env_width,
        env_height=env_height,
        scale_bar=scale_bar,
        min_cell_px=min_cell_px,
        cell_colorbar=cell_colorbar,
    )
    print(f'GIF: {gif_path}')

    return {
        'name': name,
        'gif_path': gif_path,
        'viz_path': f'{viz_path}.png' if viz_path else None,
        'state_json': serialized,
        'elapsed': elapsed,
        'n_steps': len(results),
        'n_cells': n_cells,
        'n_particles': n_particles,
        'total_time': total_time,
        'description': entry.get('description', ''),
    }
