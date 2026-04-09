"""
Test suite for multi-cell experiments.

Usage:
    python -m multi_cell.experiments.test_suite --tests single_cell_growth --output out/
"""
import argparse
import json
import os
import time

from bigraph_viz import plot_bigraph
from process_bigraph import Composite, gather_emitter_results
from process_bigraph.emitter import emitter_from_wires

from multi_cell import core_import
from multi_cell.processes.multibody import make_initial_state, get_mother_machine_config, build_microbe, make_rng
from multi_cell.processes.grow_divide import add_grow_divide_to_agents, make_grow_divide_process
from multi_cell.processes.remove_crossing import make_remove_crossing_process
from multi_cell.processes.secrete_eps import add_secrete_eps_to_agents
from multi_cell.plots.multibody_plots import simulation_to_gif


PYMUNK_CORE = core_import()


# ---------------------------------------------------------------------
# Experiment document generators
# ---------------------------------------------------------------------

def single_cell_growth_document(config=None):
    """A single cell growing in an empty environment. E. coli proportions by default."""
    config = config or {}
    env_size = config.get('env_size', 30)
    interval = config.get('interval', 30.0)
    growth_rate = config.get('growth_rate', 0.000289)  # ln(2)/2400 ~ 40 min doubling
    cell_radius = config.get('cell_radius', 0.5)
    cell_length = config.get('cell_length', 2.0)
    density = config.get('density', 0.02)
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

    document = {
        'cells': initial_state['cells'],
        'particles': initial_state['particles'],
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'gravity': 0,
                'elasticity': 0.1,
            },
            'interval': interval,
            'inputs': {
                'agents': ['cells'],
                'particles': ['particles'],
            },
            'outputs': {
                'agents': ['cells'],
                'particles': ['particles'],
            },
        },
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'particles': ['particles'],
            'time': ['global_time'],
        }),
    }
    return document


def mother_machine_document(config=None):
    """Cells growing in a mother machine with narrow dead-end channels and a flow channel.

    Default dimensions use E. coli proportions (units = micrometers):
      - Cell width (diameter): ~1.0 um
      - Cell length at birth: ~2.0 um
      - Cell length at division: ~4.0 um
      - Channel width: ~1.5 um (just wider than one cell)
    """
    config = config or {}
    import math

    # E. coli defaults (all in micrometers)
    cell_radius = config.get('cell_radius', 0.5)          # half-width of capsule
    cell_length = config.get('cell_length', 2.0)           # birth length
    division_threshold_mass = config.get('division_threshold', None)

    # Derive division threshold from geometry if not set:
    # at division, length ~ 2x birth length, mass ~ density * 2r * L
    density = config.get('density', 0.02)
    if division_threshold_mass is None:
        division_length = cell_length * 2.0
        division_threshold_mass = density * (2 * cell_radius) * division_length

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
    growth_rate = config.get('growth_rate', 0.000289)  # ln(2)/2400 ~ 40 min doubling

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

    add_grow_divide_to_agents(
        initial_state,
        agents_key='cells',
        config={
            'agents_key': 'cells',
            'rate': growth_rate,
            'threshold': division_threshold_mass,
            'mutate': True,
        },
    )

    document = {
        'cells': initial_state['cells'],
        'particles': initial_state['particles'],
        'multibody': {
            '_type': 'process',
            'address': 'local:PymunkProcess',
            'config': {
                'env_size': env_size,
                'gravity': 0,
                'elasticity': 0.1,
                'barriers': barriers,
                'wall_thickness': spacer_thickness,
            },
            'interval': interval,
            'inputs': {
                'agents': ['cells'],
                'particles': ['particles'],
            },
            'outputs': {
                'agents': ['cells'],
                'particles': ['particles'],
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
    return document


def biofilm_document(config=None):
    """Cells growing and secreting EPS particles that accumulate into a biofilm."""
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

    secretion_rate = config.get('secretion_rate', 0.005)
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

    add_secrete_eps_to_agents(
        initial_state,
        agents_key='cells',
        particles_key='particles',
        config={
            'secretion_rate': secretion_rate,
            'eps_radius': eps_radius,
        },
        interval=interval,
    )

    document = {
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
                'agents': ['cells'],
                'particles': ['particles'],
            },
            'outputs': {
                'agents': ['cells'],
                'particles': ['particles'],
            },
        },
        'emitter': emitter_from_wires({
            'agents': ['cells'],
            'particles': ['particles'],
            'time': ['global_time'],
        }),
    }
    return document


# ---------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------

EXPERIMENT_REGISTRY = {
    'single_cell_growth': {
        'document': single_cell_growth_document,
        'time': 14400.0,  # 4 hours = ~6 generations
        'config': {
            'env_size': 30,
        },
        'description': 'A single E. coli-scale microbe grows with ~40 min doubling time and divides when its length doubles. Daughters inherit mutated growth parameters, colored by phylogeny.',
    },
    'mother_machine': {
        'document': mother_machine_document,
        'time': 14400.0,  # 4 hours ~ 6 generations
        'config': {
            'n_channels': 20,
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
            'n_initial_particles': 200,
            'secretion_rate': 0.01,
        },
        'description': 'Multiple E. coli-scale cells grow and divide in an environment seeded with particles of varying sizes. Cells also secrete small EPS particles that accumulate around the colony.',
    },
}


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_experiment(name, output_dir='out'):
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

    # Save bigraph visualization of initial composition
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
    except Exception as e:
        print(f'  bigraph viz failed: {e}')
        viz_path = None

    # Save serialized state JSON
    state_json_path = os.path.join(output_dir, f'{name}_state.json')
    try:
        serialized = core.serialize(sim.schema, sim.state)
        # Replace inf/nan with JSON-safe values
        serialized = json.loads(json.dumps(serialized, default=lambda x: None if x != x else str(x)).replace('Infinity', 'null').replace('-Infinity', 'null').replace('NaN', 'null'))
        with open(state_json_path, 'w') as f:
            json.dump(serialized, f, indent=2)
    except Exception as e:
        print(f'  state JSON failed: {e}')
        serialized = None

    t0 = time.time()
    sim.run(total_time)
    elapsed = time.time() - t0
    results = gather_emitter_results(sim)[('emitter',)]
    print(f'Completed in {elapsed:.1f}s — {len(results)} steps')

    # Report final cell count
    last = results[-1]
    n_cells = len(last.get('agents', {}))
    n_particles = len(last.get('particles', {}))
    print(f'Final state: {n_cells} cells, {n_particles} particles')

    # Generate GIF — target ~100-200 frames
    gif_config = {'env_size': env_size}
    # Pass barriers through if present in the multibody process config
    multibody_cfg = document.get('multibody', {}).get('config', {})
    if 'barriers' in multibody_cfg:
        gif_config['barriers'] = multibody_cfg['barriers']
    target_frames = 150
    skip = max(1, len(results) // target_frames)
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

    gif_path = simulation_to_gif(
        results,
        filename=name,
        config=gif_config,
        out_dir=output_dir,
        color_by_phylogeny=True,
        skip_frames=skip,
        frame_duration_ms=50,
        show_time_title=True,
        world_pad=env_size * 0.05,
        dpi=150,
        xlim=xlim,
        ylim=ylim,
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


def generate_html_report(experiment_results, output_dir='out'):
    """Generate an HTML report with GIFs, bigraph viz, and interactive JSON viewer."""
    html_path = os.path.join(output_dir, 'report.html')

    sections = []
    for r in experiment_results:
        gif_rel = os.path.relpath(r['gif_path'], output_dir)
        safe_id = r['name'].replace(' ', '_')

        # Bigraph viz image
        viz_html = ''
        if r.get('viz_path') and os.path.exists(r['viz_path']):
            viz_rel = os.path.relpath(r['viz_path'], output_dir)
            viz_html = f"""
      <h3>Composition</h3>
      <img src="{viz_rel}" alt="{r['name']} bigraph" class="viz-img" />"""

        # JSON viewer
        json_html = ''
        if r.get('state_json'):
            blob = json.dumps(r['state_json'], ensure_ascii=False, default=str)
            json_html = f"""
      <h3>Initial State</h3>
      <div class="json-viewer" data-test="{safe_id}">
        <div class="json-toolbar">
          <input class="json-search" placeholder="Search keys..." />
          <button type="button" class="json-reset">Top-level</button>
          <span class="json-status"></span>
        </div>
        <div class="json-layout">
          <div class="json-nav"></div>
          <div class="json-main">
            <div class="json-path"></div>
            <div class="json-value"></div>
          </div>
        </div>
        <script type="application/json" id="json-data-{safe_id}">
{blob}
        </script>
      </div>"""

        sections.append(f"""
    <section>
      <h2>{r['name'].replace('_', ' ').title()}</h2>
      <p>{r['description']}</p>
      <table>
        <tr><td>Simulation time</td><td>{r['total_time']:.1f}s ({r['total_time']/3600:.1f} hours)</td></tr>
        <tr><td>Steps emitted</td><td>{r['n_steps']}</td></tr>
        <tr><td>Final cells</td><td>{r['n_cells']}</td></tr>
        <tr><td>Final particles</td><td>{r['n_particles']}</td></tr>
        <tr><td>Wall-clock time</td><td>{r['elapsed']:.1f}s</td></tr>
      </table>
      <h3>Simulation</h3>
      <img src="{gif_rel}" alt="{r['name']}" />{viz_html}{json_html}
    </section>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>multi-cell experiments</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; background: #fafafa; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
  h3 {{ margin-top: 1.2rem; margin-bottom: 0.4rem; }}
  section {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0; }}
  h2 {{ margin-top: 0; }}
  table {{ border-collapse: collapse; margin: .8rem 0; }}
  td {{ padding: .25rem .8rem; border: 1px solid #eee; }}
  td:first-child {{ font-weight: 600; }}
  img {{ max-width: 100%; margin-top: .5rem; border: 1px solid #ddd; border-radius: 4px; }}
  .viz-img {{ max-width: 80%; display: block; margin: 0.5rem auto; }}
  .json-viewer {{ border: 1px solid #ddd; background: #fff; border-radius: 8px; padding: 10px; margin: 10px 0; }}
  .json-toolbar {{ display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }}
  .json-toolbar input {{ flex: 1; padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px; }}
  .json-toolbar button {{ padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px; background: #f5f5f5; cursor: pointer; }}
  .json-toolbar button:hover {{ background: #eee; }}
  .json-status {{ font-size: 12px; color: #555; }}
  .json-layout {{ display: grid; grid-template-columns: 280px 1fr; gap: 10px; height: 400px; }}
  .json-nav {{ overflow: auto; border-right: 1px solid #eee; padding-right: 8px; }}
  .json-main {{ overflow: auto; padding-left: 6px; }}
  .json-item {{ padding: 5px 8px; border-radius: 6px; cursor: pointer; font-family: ui-monospace, monospace; font-size: 12px; }}
  .json-item:hover {{ background: #f3f3f3; }}
  .json-item.active {{ background: #e9eefc; }}
  .json-path {{ font-family: ui-monospace, monospace; font-size: 12px; margin-bottom: 8px; color: #333; }}
  .json-value pre {{ background: #f8f8f8; border: 1px solid #eee; border-radius: 8px; padding: 10px; overflow: auto; font-size: 12px; }}
  .json-pill {{ display: inline-block; padding: 2px 8px; border: 1px solid #ddd; border-radius: 999px; font-size: 12px; margin-right: 6px; background: #fafafa; }}
</style>
</head>
<body>
<h1>multi-cell experiments</h1>
{''.join(sections)}
{_json_viewer_js()}
</body>
</html>"""

    with open(html_path, 'w') as f:
        f.write(html)
    print(f'Report: {html_path}')
    return html_path


def _json_viewer_js():
    """Inline JS for interactive JSON navigation."""
    return r"""<script>
(function(){
  function isObj(x){ return x && typeof x === "object" && !Array.isArray(x); }
  function getAt(root, path){
    let c = root;
    for (const p of path){ if (c==null) return undefined; c = Array.isArray(c) ? c[Number(p)] : c[p]; }
    return c;
  }
  function renderVal(el, v){
    el.innerHTML = "";
    const pre = document.createElement("pre");
    if (v === null || typeof v !== "object"){ pre.textContent = JSON.stringify(v, null, 2); }
    else if (Array.isArray(v)){
      el.innerHTML = `<span class="json-pill">array [${v.length}]</span>`;
      pre.textContent = JSON.stringify(v, null, 2);
    } else {
      const keys = Object.keys(v);
      el.innerHTML = `<span class="json-pill">object {${keys.length} keys}</span>`;
      pre.textContent = JSON.stringify(v, null, 2);
    }
    el.appendChild(pre);
  }
  document.querySelectorAll(".json-viewer").forEach(viewer => {
    const testId = viewer.dataset.test;
    const dataEl = document.getElementById("json-data-" + testId);
    if (!dataEl) return;
    const root = JSON.parse(dataEl.textContent);
    const nav = viewer.querySelector(".json-nav");
    const pathEl = viewer.querySelector(".json-path");
    const valEl = viewer.querySelector(".json-value");
    const search = viewer.querySelector(".json-search");
    const resetBtn = viewer.querySelector(".json-reset");
    const status = viewer.querySelector(".json-status");

    function showKeys(obj, basePath){
      nav.innerHTML = "";
      const keys = Object.keys(obj);
      status.textContent = keys.length + " keys";
      keys.forEach(k => {
        const item = document.createElement("div");
        item.className = "json-item";
        const v = obj[k];
        const hint = v === null ? "null" : Array.isArray(v) ? `[${v.length}]` : typeof v === "object" ? "{...}" : JSON.stringify(v).slice(0,30);
        item.textContent = k + "  " + hint;
        item.onclick = () => {
          nav.querySelectorAll(".json-item").forEach(x => x.classList.remove("active"));
          item.classList.add("active");
          const path = basePath.concat([k]);
          pathEl.textContent = path.join(".");
          const val = getAt(root, path);
          if (isObj(val) && Object.keys(val).length > 0){
            showKeys(val, path);
          } else {
            renderVal(valEl, val);
          }
        };
        nav.appendChild(item);
      });
    }
    function reset(){ pathEl.textContent = ""; valEl.innerHTML = ""; showKeys(root, []); }
    resetBtn.onclick = reset;
    search.oninput = () => {
      const q = search.value.toLowerCase().trim();
      if (!q){ reset(); return; }
      nav.querySelectorAll(".json-item").forEach(el => {
        el.style.display = el.textContent.toLowerCase().includes(q) ? "" : "none";
      });
    };
    reset();
  });
})();
</script>"""


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='multi-cell test suite')
    parser.add_argument(
        '--tests', nargs='+',
        default=list(EXPERIMENT_REGISTRY.keys()),
        choices=list(EXPERIMENT_REGISTRY.keys()),
        help='Which experiments to run',
    )
    parser.add_argument('--output', default='out', help='Output directory')
    parser.add_argument('--open', action='store_true', default=True, help='Open report in browser')
    args = parser.parse_args()

    all_results = []
    for name in args.tests:
        result = run_experiment(name, output_dir=args.output)
        all_results.append(result)

    html_path = generate_html_report(all_results, output_dir=args.output)

    if args.open:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(html_path)}')


if __name__ == '__main__':
    main()
