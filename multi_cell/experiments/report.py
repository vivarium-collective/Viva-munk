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

    return f"""
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
    </section>"""


def generate_html_report(experiment_results, output_dir='out'):
    """Generate an HTML report with GIFs, bigraph viz, and an interactive
    JSON viewer for each experiment in `experiment_results`."""
    html_path = os.path.join(output_dir, 'report.html')
    meta = _gather_metadata()

    sections = [_section_html(r, output_dir) for r in experiment_results]

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
</style>
</head>
<body>
<h1>multi-cell experiments</h1>
<div class="meta">
  <div><strong>Generated:</strong> {meta['generated_at']}</div>
  <div><strong>On:</strong> {meta['generated_on']}</div>
  {commit_html}
</div>
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
