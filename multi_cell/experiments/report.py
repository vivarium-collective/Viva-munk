"""HTML report generation for the experiment runner."""
import json
import os
import socket
import subprocess
from datetime import datetime, timezone


def _gather_metadata():
    """Collect provenance info: timestamp, host, git commit."""
    meta = {
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
    }
    if os.environ.get('GITHUB_ACTIONS'):
        runner = os.environ.get('RUNNER_NAME', 'GitHub Actions')
        repo = os.environ.get('GITHUB_REPOSITORY', '')
        meta['generated_on'] = f'GitHub Actions ({runner})' + (f' — {repo}' if repo else '')
    else:
        meta['generated_on'] = socket.gethostname()
    sha = os.environ.get('GITHUB_SHA')
    if not sha:
        try:
            sha = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            ).decode().strip()
        except Exception:
            sha = None
    meta['commit'] = sha
    repo_url = os.environ.get('GITHUB_SERVER_URL', 'https://github.com')
    repo_name = os.environ.get('GITHUB_REPOSITORY', 'vivarium-collective/Viva-munk')
    meta['commit_url'] = f'{repo_url}/{repo_name}/commit/{sha}' if sha else None
    return meta


def _section_html(r, output_dir):
    """Render one experiment as an HTML <section>."""
    gif_rel = os.path.relpath(r['gif_path'], output_dir)
    safe_id = r['name'].replace(' ', '_')

    viz_html = ''
    if r.get('viz_path') and os.path.exists(r['viz_path']):
        viz_rel = os.path.relpath(r['viz_path'], output_dir)
        viz_html = f"""
      <h3>Composition</h3>
      <img src="{viz_rel}" alt="{r['name']} bigraph" class="viz-img" />"""

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

    sim_id = r.get('simulation_id')
    sim_id_row = ''
    if sim_id:
        tip = (
            'Primary key of this run in out/history.db on the machine that '
            'generated the report. Click to copy. See "How to reproduce" at '
            'the end of this report for how to access the underlying data.'
        )
        sim_id_row = (
            f'<tr><td>Simulation ID</td>'
            f'<td><code class="sim-id" title="{tip}">{sim_id}</code></td>'
            f'</tr>'
        )

    cached = bool(r.get('cached'))
    badge = (
        '<span class="badge badge-cached" title="Rebuilt from existing rows in history.db — no simulation re-run">cached</span>'
        if cached else
        '<span class="badge badge-fresh" title="Simulation was executed fresh for this report">fresh</span>'
    )

    return f"""
    <section id="{safe_id}">
      <h2>{r['name'].replace('_', ' ').title()} {badge}</h2>
      <p>{r['description']}</p>
      <table>
        {sim_id_row}
        <tr><td>Simulation time</td><td>{r['total_time']:.1f}s ({r['total_time']/3600:.1f} hours)</td></tr>
        <tr><td>Steps emitted</td><td>{r['n_steps']}</td></tr>
        <tr><td>Final cells</td><td>{r['n_cells']}</td></tr>
        <tr><td>Final particles</td><td>{r['n_particles']}</td></tr>
        <tr><td>Wall-clock time</td><td>{r['elapsed']:.1f}s</td></tr>
      </table>
      <h3>Simulation</h3>
      <img src="{gif_rel}" alt="{r['name']}" />{viz_html}{json_html}
    </section>"""


def _summary_html(experiment_results):
    """Render the end-of-report summary: per-experiment + total runtimes."""
    if not experiment_results:
        return ''

    rows = []
    total_wall = 0.0
    total_sim = 0.0
    for r in experiment_results:
        wall = float(r.get('elapsed') or 0.0)
        sim_t = float(r.get('total_time') or 0.0)
        total_wall += wall
        total_sim += sim_t
        cached = bool(r.get('cached'))
        badge = (
            '<span class="badge badge-cached">cached</span>'
            if cached else
            '<span class="badge badge-fresh">fresh</span>'
        )
        short_id = (r.get('simulation_id') or '')[:8] or '—'
        name_href = r['name'].replace(' ', '_')
        if wall > 0:
            wall_tip = ' title="wall-clock time of the original simulation run (recorded in history.db)"' if cached else ''
            wall_cell = f'<span{wall_tip}>{wall:.1f}s</span>'
        else:
            wall_cell = '—'
        rows.append(
            f'<tr>'
            f'<td><a href="#{name_href}">{r["name"].replace("_", " ").title()}</a></td>'
            f'<td>{badge}</td>'
            f'<td><code>{short_id}</code></td>'
            f'<td>{r.get("n_steps", "—")}</td>'
            f'<td>{sim_t/3600:.2f} h</td>'
            f'<td>{wall_cell}</td>'
            f'</tr>'
        )

    fresh_count = sum(1 for r in experiment_results if not r.get('cached'))
    cached_count = len(experiment_results) - fresh_count

    footer = (
        f'<tr><td colspan="2">Totals</td>'
        f'<td colspan="2">{len(experiment_results)} experiments '
        f'({fresh_count} fresh, {cached_count} cached)</td>'
        f'<td>{total_sim/3600:.2f} h sim</td>'
        f'<td>{total_wall:.1f}s wall</td></tr>'
    )

    return f"""
<section class="summary" id="summary">
  <h2>Run Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Experiment</th>
        <th>Source</th>
        <th>Sim ID</th>
        <th>Steps</th>
        <th>Sim time</th>
        <th>Wall-clock</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
    <tfoot>{footer}</tfoot>
  </table>
  <p style="margin-top:0.8rem; font-size:12px; color:#666;">
    <strong>fresh</strong> = simulation executed for this report.
    <strong>cached</strong> = rebuilt from existing rows in <code>history.db</code>; no simulation re-run. Wall-clock for cached rows shows the original run's recorded duration.
  </p>
</section>"""


def _reproduce_html(meta):
    """End-of-report block with short instructions for re-running the
    experiments and accessing the recorded data on another machine."""
    commit = (meta.get('commit') or '')[:8] or 'main'
    return f"""
<section class="reproduce" id="reproduce">
  <h2>How to reproduce</h2>
  <p>The simulation IDs shown above identify runs in a SQLite database on the machine that generated this report (<code>out/history.db</code>). Viewers reading this report online will not have that file — to get the underlying data you need to re-run the experiments yourself.</p>
  <h3>Re-run locally</h3>
  <pre class="snippet">git clone https://github.com/vivarium-collective/Viva-munk
cd Viva-munk
git checkout {commit}
pip install -e .
python -m multi_cell.experiments.test_suite</pre>
  <p style="font-size:12px;color:#666;">This records every run into <code>out/history.db</code> and regenerates an HTML report at <code>out/report.html</code>.</p>
  <h3>Access a run's data in Python</h3>
  <pre class="snippet">from process_bigraph.emitter import load_history, load_simulation_metadata

# Replace with any simulation_id from the summary table above.
sim_id  = '&lt;simulation_id&gt;'
history = load_history('out/history.db', sim_id)          # list of per-step dicts
meta    = load_simulation_metadata('out/history.db', sim_id)  # config + provenance</pre>
  <h3>Re-render a GIF from recorded history (no re-run)</h3>
  <pre class="snippet">python -m multi_cell.experiments.replay &lt;simulation_id&gt;</pre>
</section>"""


def generate_html_report(experiment_results, output_dir='out'):
    """Generate an HTML report with GIFs, bigraph viz, and an interactive
    JSON viewer for each experiment in `experiment_results`."""
    html_path = os.path.join(output_dir, 'report.html')
    meta = _gather_metadata()

    sections = [_section_html(r, output_dir) for r in experiment_results]

    nav_links = '\n'.join(
        f'  <a href="#{r["name"].replace(" ", "_")}">{r["name"].replace("_", " ").title()}</a>'
        for r in experiment_results
    )
    nav_html = f"""<nav class="experiment-nav">
  <span class="nav-title">Experiments:</span>
{nav_links}
  <a href="#summary">Summary</a>
  <a href="#reproduce">How to reproduce</a>
</nav>"""

    commit_html = ''
    if meta.get('commit'):
        commit_html = (
            f'<div><strong>Commit:</strong> '
            f'<a href="{meta["commit_url"]}"><code>{meta["commit"][:8]}</code></a></div>'
        )

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
  .meta {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0.8rem 1.2rem; margin: 1rem 0; font-size: 13px; color: #555; }}
  .meta div {{ display: inline-block; margin-right: 1.5rem; }}
  .meta code {{ background: #f0f0f0; padding: 1px 6px; border-radius: 4px; }}
  .meta a {{ color: #0366d6; text-decoration: none; }}
  .experiment-nav {{
    position: sticky; top: 0; z-index: 10;
    background: #fff; border: 1px solid #ddd; border-radius: 8px;
    padding: 0.6rem 1rem; margin: 1rem 0;
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
  }}
  .experiment-nav .nav-title {{ font-weight: 600; margin-right: 0.4rem; color: #555; }}
  .experiment-nav a {{
    padding: 3px 10px; border: 1px solid #ddd; border-radius: 999px;
    background: #fafafa; color: #0366d6; text-decoration: none; font-size: 13px;
  }}
  .experiment-nav a:hover {{ background: #eef3fa; border-color: #b8c7dc; }}
  section {{ scroll-margin-top: 4rem; }}
  .sim-id {{ background: #f0f0f0; padding: 2px 8px; border-radius: 4px; cursor: copy; user-select: all; font-size: 12px; }}
  .sim-id:hover {{ background: #e4ecf7; }}
  .sim-id.copied {{ background: #d4edda; }}
  .replay-hint {{ margin-top: 6px; font-size: 11px; color: #555; }}
  .replay-hint code {{ background: #f8f8f8; padding: 1px 6px; border-radius: 3px; }}
  .replay-hint strong {{ display: block; margin-top: 6px; color: #333; font-size: 11px; }}
  .snippet {{ background: #f8f8f8; border: 1px solid #eee; border-radius: 4px; padding: 6px 10px; margin: 3px 0 6px 0; font-family: ui-monospace, monospace; font-size: 11px; overflow-x: auto; user-select: all; }}
  .badge {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge-fresh  {{ background: #e8f4ff; color: #0b61a4; border: 1px solid #b8d8f0; }}
  .badge-cached {{ background: #fff6d6; color: #8a6d00; border: 1px solid #ead58a; }}
  .summary {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem 1.5rem; margin-top: 2rem; }}
  .summary h2 {{ margin-top: 0; }}
  .summary table {{ width: 100%; }}
  .summary td, .summary th {{ padding: 0.3rem 0.6rem; border: 1px solid #eee; text-align: left; }}
  .summary th {{ background: #f8f8f8; }}
  .summary tfoot td {{ font-weight: 600; background: #fafafa; }}
  .reproduce {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem 1.5rem; margin-top: 1.5rem; }}
  .reproduce h2 {{ margin-top: 0; }}
  .reproduce h3 {{ margin-top: 1.2rem; margin-bottom: 0.3rem; font-size: 14px; }}
  .reproduce pre.snippet {{ background: #f6f8fa; border: 1px solid #e4e4e4; border-radius: 6px; padding: 10px 12px; margin: 0; font-family: ui-monospace, monospace; font-size: 12px; overflow-x: auto; }}
  .reproduce p {{ font-size: 13px; color: #444; margin: 0.4rem 0; }}
</style>
</head>
<body>
<h1>multi-cell experiments</h1>
<div class="meta">
  <div><strong>Generated:</strong> {meta['generated_at']}</div>
  <div><strong>On:</strong> {meta['generated_on']}</div>
  {commit_html}
</div>
{nav_html}
{''.join(sections)}
{_summary_html(experiment_results)}
{_reproduce_html(meta)}
{_json_viewer_js()}
<script>
document.querySelectorAll('.sim-id').forEach(el => {{
  el.addEventListener('click', async () => {{
    try {{
      await navigator.clipboard.writeText(el.textContent);
      el.classList.add('copied');
      setTimeout(() => el.classList.remove('copied'), 900);
    }} catch (e) {{}}
  }});
}});
</script>
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
