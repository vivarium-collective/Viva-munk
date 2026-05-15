# Studies Phase 1.5 — v3-native Study Handlers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken `/api/study-*` URL aliases (which point at v2-layout-coupled investigation handlers) with v3-native Study handlers that read/write the `studies/<name>/` layout directly, so the Study Detail view's action buttons work end-to-end.

**Architecture:** Phase 1's Task 2.2 aliased `/api/study-*` onto existing investigation handlers, betting they were layout-agnostic. They aren't — they expect `composites/<variant>.yaml` sidecars, a `sim_name` column, and v2 spec shapes. This plan: (1) fix one genuine schema drift (`sim_name`), (2) extract the *already-v3-native* run-and-persist core out of `_post_composite_test_run`, (3) write small v3-native handlers for run-baseline / run-variant / variant-add / variant-delete / run-delete / runs-clear / comparison-add that operate on `studies/<name>/study.yaml` + `studies/<name>/runs.db` directly, and (4) drop the dead aliases. The handlers written fresh in Phase 1 (set-objective, rename, export, the Study Detail *page*) already work — they're the template.

**Tech Stack:** Python 3.11+, pytest, pyyaml, sqlite3, subprocess. `vivarium-dashboard` on branch `studies-phase-1`. Tests run via `~/code/viva-munk/.venv/bin/python -m pytest`.

**The v3 Study layout (the contract this plan targets):**
```
studies/<name>/
├── study.yaml      # schema_version: 3 — name, status, objective, description,
│                   #   baseline:{composite,params}, variants:[], runs:[],
│                   #   visualizations:[], comparisons:[], conclusion, parent_studies:[]
├── runs.db         # runs_meta + history (SQLiteEmitter); per-Study, accumulating
├── composites/     # (unused for pure-param variants — kept for future process_overrides)
└── viz/            # rendered viz HTML
```

**The 7 broken aliased routes this plan replaces** (all currently 404/500/400 against a `studies/` study):
`/api/study-run-baseline`, `/api/study-run-variant`, `/api/study-variant-add`, `/api/study-variant-delete`, `/api/study-run-delete`, `/api/study-runs-clear`, `/api/study-comparison-add`. Plus `/api/study/<name>` (detail API) — fixed indirectly by Task 1.

**Key reuse:** `_post_composite_test_run` already resolves a generator composite, runs it in a subprocess, persists `runs_meta` + `history` to a runs.db, and renders viz. Task 2 extracts that core; the study-run handlers call it pointed at the Study's runs.db.

**File structure:**
- `vivarium_dashboard/lib/composite_runs.py` — runs.db schema (Task 1).
- `vivarium_dashboard/server.py` — extracted `_run_composite_subprocess` core (Task 2) + 7 new `_post_study_*` handlers (Tasks 3-7) + dead-alias cleanup (Task 8).
- `vivarium_dashboard/static/study-detail.js` — wire `btn-delete-variant` (Task 8).
- `tests/test_study_runs.py` (new), `tests/test_study_variants.py` (new), `tests/test_composite_runs.py` (extend).

---

## Task 1: runs.db schema consistency — add `sim_name`

**Why:** `_get_investigation_detail` (aliased as `/api/study/<name>`) does `SELECT run_id, sim_name, ... FROM runs_meta`. The base `connect()` schema has no `sim_name` — only `_post_composite_test_run` ALTERs it in. So a Study's runs.db (created by `copy_run_to_new_db`, which uses the base schema) crashes the detail query with `no such column: sim_name`.

**Files:**
- Modify: `vivarium_dashboard/lib/composite_runs.py` (the `_SCHEMA_RUNS_META` constant + `connect()`)
- Test: `tests/test_composite_runs.py` (extend)

- [ ] **Step 1: Write the failing test.** Append to `tests/test_composite_runs.py`:

```python
def test_connect_runs_meta_has_sim_name(tmp_path):
    """Fresh runs.db must have sim_name — _get_investigation_detail SELECTs it."""
    from vivarium_dashboard.lib.composite_runs import connect
    conn = connect(tmp_path / "fresh.db")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs_meta)")}
    conn.close()
    assert "sim_name" in cols


def test_connect_adds_sim_name_to_legacy_db(tmp_path):
    """connect() ALTERs sim_name into a pre-existing runs_meta that lacks it."""
    import sqlite3
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(str(db))
    raw.executescript('''
        CREATE TABLE runs_meta (
            run_id TEXT PRIMARY KEY, spec_id TEXT NOT NULL, label TEXT,
            params_json TEXT, started_at REAL NOT NULL, completed_at REAL,
            n_steps INTEGER, status TEXT NOT NULL
        );
    ''')
    raw.execute(
        "INSERT INTO runs_meta VALUES (?,?,?,?,?,?,?,?)",
        ("r1", "pkg.foo", "lbl", "{}", 1.0, 2.0, 5, "completed"),
    )
    raw.commit()
    raw.close()

    from vivarium_dashboard.lib.composite_runs import connect
    conn = connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs_meta)")}
    # Existing row survives the migration.
    n = conn.execute("SELECT COUNT(*) FROM runs_meta").fetchone()[0]
    conn.close()
    assert "sim_name" in cols
    assert n == 1
```

- [ ] **Step 2: Run tests to verify they fail.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_composite_runs.py -v -k sim_name`
Expected: 2 failures — `sim_name` not in columns.

- [ ] **Step 3: Add `sim_name` to the base schema + migrate legacy DBs.** In `vivarium_dashboard/lib/composite_runs.py`, update the schema constant:

```python
_SCHEMA_RUNS_META = """
CREATE TABLE IF NOT EXISTS runs_meta (
    run_id        TEXT PRIMARY KEY,
    spec_id       TEXT NOT NULL,
    label         TEXT,
    params_json   TEXT,
    started_at    REAL NOT NULL,
    completed_at  REAL,
    n_steps       INTEGER,
    status        TEXT NOT NULL,
    sim_name      TEXT
);
"""
```

And in `connect()`, after the `CREATE TABLE` executes, ALTER the column into pre-existing DBs that predate it:

```python
def connect(db_file: str | Path) -> sqlite3.Connection:
    """Open the runs DB and ensure the metadata schema exists."""
    db_file = Path(db_file)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA_RUNS_META)
    conn.execute(_INDEX_RUNS_META)
    # Legacy DBs created before sim_name was in the base schema: ALTER it in.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(runs_meta)")}
    if "sim_name" not in cols:
        conn.execute("ALTER TABLE runs_meta ADD COLUMN sim_name TEXT")
    conn.commit()
    return conn
```

- [ ] **Step 4: Run tests to verify they pass.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_composite_runs.py -v`
Expected: all pass (the 2 new + all prior).

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/lib/composite_runs.py tests/test_composite_runs.py
git commit -m "add sim_name to runs.db base schema; ALTER it into legacy DBs"
```

---

## Task 2: Extract `_run_composite_subprocess` shared core

**Why:** `_post_composite_test_run` already does the v3-native work — resolve a generator composite, run it in a subprocess, persist `runs_meta`+`history` to a runs.db, render viz. The study-run handlers (Tasks 3-4) need exactly this, just pointed at a Study's runs.db. Extract the reusable core so both call it.

**Files:**
- Modify: `vivarium_dashboard/server.py` (extract `_run_composite_subprocess`; refactor `_post_composite_test_run` to call it)
- Test: `tests/test_composite_explorer_api.py` (verify no behavior change — the existing test-run tests must still pass)

- [ ] **Step 1: Read `_post_composite_test_run` end to end.** It spans roughly `server.py:5392` to `server.py:5560`. Identify the three phases: (a) resolve `spec_id` + `overrides` → `state` dict, (b) wire emitter + build subprocess script + run + parse, (c) `save_metadata` / `complete_metadata` bookkeeping.

- [ ] **Step 2: Write a characterization test FIRST (locks current behavior).** Append to `tests/test_composite_explorer_api.py`:

```python
def test_run_composite_subprocess_persists_and_returns(tmp_path, monkeypatch):
    """_run_composite_subprocess runs a generator composite into a given db_file
    and returns {simulation_id, results, viz_html, steps}."""
    import vivarium_dashboard.server as srv
    # Use the fixtures workspace's package so build_core + a generator exist.
    # (Mirror whatever _fixtures_workspace the other tests in this file use.)
    # This test asserts the SHAPE of the return + that runs.db got rows.
    # If the fixtures harness makes a full subprocess run impractical here,
    # SKIP this step and rely on the e2e smoke test in Task 8 instead — but
    # the refactor in Step 3 must be behavior-preserving regardless.
    pytest.skip("covered by existing test-run tests + Task 8 e2e; see Step 3")
```

(The point of Step 2 is the discipline reminder: the Step 3 refactor must be behavior-preserving. The *existing* `test_composite_explorer_api.py` test-run tests are the real safety net — they must stay green.)

- [ ] **Step 3: Extract the helper.** Add a module-level function in `server.py` (near the other module-level helpers like `_study_dir`). Move phase (b)+(c) of `_post_composite_test_run` into it:

```python
def _run_composite_subprocess(*, pkg, state, steps, db_file, run_id, spec_id,
                              label, sim_name=None, timeout=120):
    """Run a resolved composite `state` for `steps` steps in a subprocess,
    persisting runs_meta + history (via an injected SQLiteEmitter) to `db_file`.

    Shared by _post_composite_test_run (scratchpad db) and the study-run
    handlers (per-Study db). Does NOT clear prior rows — callers decide.

    Returns (response_dict, status_code). response_dict always has
    "simulation_id"; on success also "results", "viz_html", "steps".
    """
    from vivarium_dashboard.lib import composite_runs as cr

    state = cr.inject_sqlite_emitter(state, run_id=run_id, db_file=db_file)

    py = sys.executable
    script = textwrap.dedent(f"""
        import json, sys, traceback
        try:
            from {pkg}.core import build_core
            from process_bigraph import Composite, gather_emitter_results
            from process_bigraph.emitter import SQLiteEmitter
            core = build_core()
            core.register_link('SQLiteEmitter', SQLiteEmitter)
            composite = Composite({{'state': __import__('json').loads({json.dumps(json.dumps(state, default=_json_default))})}}, core=core)
            composite.run({steps})
            results = gather_emitter_results(composite)
            out = {{}}
            for path_tuple, entries in results.items():
                key = '.'.join(str(p) for p in path_tuple)
                out[key] = entries
            viz_html = {{}}
            try:
                from pbg_superpowers.visualization import render_results
                rendered = render_results(composite)
                for path_tuple, payload in rendered.items():
                    key = '.'.join(str(p) for p in path_tuple)
                    viz_html[key] = payload
            except Exception:
                viz_html = {{}}
            print('@@@RESULTS@@@')
            print(json.dumps({{'results': out, 'viz_html': viz_html}}, default=str))
        except Exception:
            print('@@@ERROR@@@')
            print(traceback.format_exc())
    """)

    conn = cr.connect(db_file)
    try:
        try:
            cr.save_metadata(conn, spec_id=spec_id, run_id=run_id,
                             params={}, label=label, started_at=time.time())
            if sim_name is not None:
                conn.execute("UPDATE runs_meta SET sim_name=? WHERE run_id=?",
                             (sim_name, run_id))
                conn.commit()
        except sqlite3.IntegrityError:
            return {"simulation_id": run_id,
                    "error": "duplicate run_id (rare timing collision) — retry"}, 500

        try:
            result = subprocess.run([py, "-c", script], cwd=WORKSPACE,
                                    capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                if exc.process is not None:
                    exc.process.kill()
                    exc.process.communicate(timeout=2)
            except Exception:
                pass
            cr.complete_metadata(conn, run_id=run_id, n_steps=0, status="failed")
            return {"simulation_id": run_id, "error": "run timed out"}, 504

        out = result.stdout
        if "@@@ERROR@@@" in out:
            cr.complete_metadata(conn, run_id=run_id, n_steps=0, status="failed")
            tb = out.split("@@@ERROR@@@", 1)[1].strip()
            return {"simulation_id": run_id, "error": "run failed",
                    "traceback": tb}, 502

        try:
            payload = json.loads(out.split("@@@RESULTS@@@", 1)[1].strip())
        except (IndexError, json.JSONDecodeError):
            cr.complete_metadata(conn, run_id=run_id, n_steps=0, status="failed")
            return {"simulation_id": run_id, "error": "could not parse run output",
                    "stdout": out, "stderr": result.stderr}, 502

        if isinstance(payload, dict) and "results" in payload:
            results = payload.get("results") or {}
            viz_html = payload.get("viz_html") or {}
        else:
            results = payload
            viz_html = {}

        cr.complete_metadata(conn, run_id=run_id, n_steps=steps, status="completed")
        return {"simulation_id": run_id, "results": results,
                "viz_html": viz_html, "steps": steps}, 200
    finally:
        conn.close()
```

Then refactor `_post_composite_test_run` so phase (b)+(c) is replaced by a call:

```python
        # ... after `state` is resolved and the scratchpad-clear block ...
        db_file = str(WORKSPACE / ".pbg" / "composite-runs.db")
        run_id = cr.generate_run_id(spec_id, overrides)
        # (scratchpad-clear block stays here, unchanged)
        response, code = _run_composite_subprocess(
            pkg=pkg, state=state, steps=steps, db_file=db_file,
            run_id=run_id, spec_id=spec_id, label=label, timeout=120,
        )
        return self._json(response, code)
```

NOTE: `_post_composite_test_run` previously passed `params=overrides` to `save_metadata`. The extracted helper passes `params={}` for simplicity — if the existing test-run tests assert on `params_json`, keep an `overrides` param on `_run_composite_subprocess` and thread it through to `save_metadata`. Check `tests/test_composite_explorer_api.py` for such an assertion before finalizing; if present, add `overrides=None` kwarg to the helper signature and use it in the `save_metadata` call.

- [ ] **Step 4: Run the existing test-run + composite-explorer tests.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_composite_explorer_api.py -v`
Expected: all pass — the refactor is behavior-preserving.

- [ ] **Step 5: End-to-end check — the scratchpad test-run still works.**

```bash
cd ~/code/viva-munk
lsof -ti tcp:8765 | xargs -r kill 2>/dev/null; sleep 1
.venv/bin/vivarium-dashboard serve --workspace . --port 8765 > /tmp/dash.log 2>&1 &
until curl -sS -o /dev/null -w "%{http_code}" http://localhost:8765/ 2>/dev/null | grep -q "^200$"; do sleep 0.5; done
curl -sS -X POST http://localhost:8765/api/composite-test-run -H "Content-Type: application/json" \
  -d '{"id":"multi_cell.composites.chemotaxis","steps":2}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('error:', d.get('error'), '| viz keys:', list((d.get('viz_html') or {}).keys()))"
lsof -ti tcp:8765 | xargs -r kill 2>/dev/null
```

Expected: `error: None | viz keys: ['multibody_viz']`.

- [ ] **Step 6: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/server.py tests/test_composite_explorer_api.py
git commit -m "extract _run_composite_subprocess shared core from _post_composite_test_run"
```

---

## Task 3: `_post_study_run_baseline` — run a Study's baseline composite

**Files:**
- Modify: `vivarium_dashboard/server.py` (new handler + `_append_study_run` helper + route)
- Test: `tests/test_study_runs.py` (new)

- [ ] **Step 1: Write the failing test.** Create `tests/test_study_runs.py`:

```python
"""v3-native Study run handlers — run baseline / variant into the Study's runs.db."""
import sqlite3
import yaml
import pytest


@pytest.fixture
def _study_ws(tmp_path, monkeypatch):
    """Workspace with one v3 study whose baseline is a real viva-munk composite."""
    import vivarium_dashboard.server as srv
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "workspace.yaml").write_text(
        'schema_version: 2\nname: viva-munk\ncreated: "2026-05-14"\n'
        'plugin_version: 0.6.1\npackage_path: multi_cell\n'
    )
    sd = ws / "studies" / "s1"
    (sd / "composites").mkdir(parents=True)
    (sd / "viz").mkdir()
    (sd / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 3, "name": "s1", "created": "2026-05-14",
        "status": "ran", "objective": "",
        "baseline": {"composite": "multi_cell.composites.chemotaxis",
                     "params": {"n_steps": 2}},
        "variants": [
            {"name": "fast", "intervention": {
                "description": "more steps",
                "parameter_overrides": {"n_steps": 3}}},
        ],
        "runs": [], "visualizations": [], "comparisons": [],
        "conclusion": None, "parent_studies": [],
    }))
    monkeypatch.setattr(srv, "WORKSPACE", ws)
    return ws


def test_run_baseline_persists_and_appends(_study_ws):
    from vivarium_dashboard.server import _post_study_run_baseline_for_test
    resp, code = _post_study_run_baseline_for_test(_study_ws, {"study": "s1", "steps": 2})
    assert code == 200, resp
    # runs.db got a row
    db = _study_ws / "studies" / "s1" / "runs.db"
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM runs_meta").fetchone()[0]
    conn.close()
    assert n == 1
    # study.yaml.runs grew by one, with variant=None (baseline)
    spec = yaml.safe_load((_study_ws / "studies" / "s1" / "study.yaml").read_text())
    assert len(spec["runs"]) == 1
    assert spec["runs"][0]["variant"] is None
    assert spec["runs"][0]["run_id"] == resp["simulation_id"]


def test_run_baseline_missing_study(_study_ws):
    from vivarium_dashboard.server import _post_study_run_baseline_for_test
    resp, code = _post_study_run_baseline_for_test(_study_ws, {"study": "nope"})
    assert code == 404
```

- [ ] **Step 2: Run, verify failure.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_runs.py -v -k baseline`
Expected: ImportError.

- [ ] **Step 3: Implement the handler + helpers.** Add to `server.py`:

```python
def _append_study_run(study_dir, run_record: dict) -> None:
    """Append a run record to a Study's study.yaml `runs` list."""
    sf = study_dir / "study.yaml"
    spec = yaml.safe_load(sf.read_text()) or {}
    spec.setdefault("runs", []).append(run_record)
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))


def _resolve_study_baseline_state(pkg, spec_id, params):
    """Resolve a generator composite spec_id + params → a state dict.

    Returns (state, error_dict_or_None). Mirrors the generator branch of
    _post_composite_test_run; Studies always reference generator composites.
    """
    try:
        from pbg_superpowers.composite_generator import (
            _REGISTRY, build_generator, discover_generators,
        )
    except ImportError:
        return None, {"error": "pbg_superpowers not importable"}
    if not _REGISTRY:
        discover_generators()
    entry = _REGISTRY.get(spec_id)
    if entry is None:
        return None, {"error": f"composite {spec_id!r} not in generator registry"}
    try:
        doc = build_generator(entry, overrides=params)
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"generator build failed: {e}"}
    if isinstance(doc, dict) and "state" in doc and isinstance(doc["state"], dict):
        return doc["state"], None
    return doc, None


def _post_study_run_baseline(self, body: dict):
    """POST /api/study-run-baseline {study, steps?}"""
    response, code = _post_study_run_baseline_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_run_baseline_for_test(ws_root, body):
    from vivarium_dashboard.lib import composite_runs as cr

    name = _study_name_from_body(body)
    if not name:
        return {"error": "missing study"}, 400
    study_dir = _study_dir(name)
    sf = study_dir / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404

    spec = yaml.safe_load(sf.read_text()) or {}
    baseline = spec.get("baseline") or {}
    spec_id = baseline.get("composite")
    if not spec_id:
        return {"error": "study has no baseline.composite"}, 400
    params = dict(baseline.get("params") or {})
    steps = int(body.get("steps") or params.get("n_steps") or 5)

    ws_data = yaml.safe_load((ws_root / "workspace.yaml").read_text())
    pkg = ws_data.get("package_path") or ("pbg_" + ws_data.get("name", "").replace("-", "_"))

    state, err = _resolve_study_baseline_state(pkg, spec_id, params)
    if err is not None:
        return err, 400

    db_file = str(study_dir / "runs.db")
    run_id = cr.generate_run_id(spec_id, params)
    response, code = _run_composite_subprocess(
        pkg=pkg, state=state, steps=steps, db_file=db_file,
        run_id=run_id, spec_id=spec_id, label="baseline", sim_name="baseline",
    )
    if code == 200:
        _append_study_run(study_dir, {
            "run_id": run_id, "variant": None, "label": "baseline",
            "status": "completed", "n_steps": steps,
        })
    return response, code
```

Register the route — in Task 2.2's `_POST_ROUTE_MAP`, change the `/api/study-run-baseline` entry from the aliased investigation handler to the new one:

```python
"/api/study-run-baseline": "_post_study_run_baseline",
```

And REMOVE `/api/study-run-baseline` from the `_POST_STUDY_ALIASES` dict (so the alias-injection loop no longer overwrites it). If `_POST_STUDY_ALIASES` is applied AFTER `_POST_ROUTE_MAP` is built, removing the alias entry is sufficient; verify the ordering and adjust.

- [ ] **Step 4: Run, verify pass.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_runs.py -v -k baseline`
Expected: both pass. (Note: this test actually runs a chemotaxis composite in a subprocess — it may take 10-30s.)

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/server.py tests/test_study_runs.py
git commit -m "add v3-native _post_study_run_baseline; runs into the Study's runs.db"
```

---

## Task 4: `_post_study_run_variant` — run a variant (baseline + param overrides)

**Files:**
- Modify: `vivarium_dashboard/server.py` (new handler + route)
- Test: `tests/test_study_runs.py` (extend)

- [ ] **Step 1: Write the failing test.** Append to `tests/test_study_runs.py`:

```python
def test_run_variant_layers_overrides(_study_ws):
    from vivarium_dashboard.server import _post_study_run_variant_for_test
    resp, code = _post_study_run_variant_for_test(
        _study_ws, {"study": "s1", "variant": "fast"})
    assert code == 200, resp
    spec = yaml.safe_load((_study_ws / "studies" / "s1" / "study.yaml").read_text())
    # The new run records the variant name.
    variant_runs = [r for r in spec["runs"] if r.get("variant") == "fast"]
    assert len(variant_runs) == 1
    assert variant_runs[0]["run_id"] == resp["simulation_id"]


def test_run_variant_unknown_variant(_study_ws):
    from vivarium_dashboard.server import _post_study_run_variant_for_test
    resp, code = _post_study_run_variant_for_test(
        _study_ws, {"study": "s1", "variant": "ghost"})
    assert code == 404
```

- [ ] **Step 2: Run, verify failure.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_runs.py -v -k variant`
Expected: ImportError.

- [ ] **Step 3: Implement.** Add to `server.py`:

```python
def _post_study_run_variant(self, body: dict):
    """POST /api/study-run-variant {study, variant, steps?}"""
    response, code = _post_study_run_variant_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_run_variant_for_test(ws_root, body):
    from vivarium_dashboard.lib import composite_runs as cr

    name = _study_name_from_body(body)
    variant_name = (body.get("variant") or "").strip()
    if not name or not variant_name:
        return {"error": "missing study or variant"}, 400
    study_dir = _study_dir(name)
    sf = study_dir / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404

    spec = yaml.safe_load(sf.read_text()) or {}
    baseline = spec.get("baseline") or {}
    spec_id = baseline.get("composite")
    if not spec_id:
        return {"error": "study has no baseline.composite"}, 400

    variant = next((v for v in (spec.get("variants") or [])
                    if v.get("name") == variant_name), None)
    if variant is None:
        return {"error": f"variant {variant_name!r} not found"}, 404

    # Layer the variant's parameter_overrides on top of baseline.params.
    params = dict(baseline.get("params") or {})
    intervention = variant.get("intervention") or {}
    params.update(intervention.get("parameter_overrides") or {})
    steps = int(body.get("steps") or params.get("n_steps") or 5)

    ws_data = yaml.safe_load((ws_root / "workspace.yaml").read_text())
    pkg = ws_data.get("package_path") or ("pbg_" + ws_data.get("name", "").replace("-", "_"))

    state, err = _resolve_study_baseline_state(pkg, spec_id, params)
    if err is not None:
        return err, 400

    db_file = str(study_dir / "runs.db")
    run_id = cr.generate_run_id(spec_id, params)
    response, code = _run_composite_subprocess(
        pkg=pkg, state=state, steps=steps, db_file=db_file,
        run_id=run_id, spec_id=spec_id, label=variant_name, sim_name=variant_name,
    )
    if code == 200:
        _append_study_run(study_dir, {
            "run_id": run_id, "variant": variant_name, "label": variant_name,
            "status": "completed", "n_steps": steps,
        })
    return response, code
```

Register the route in `_POST_ROUTE_MAP`:
```python
"/api/study-run-variant": "_post_study_run_variant",
```
And remove `/api/study-run-variant` from `_POST_STUDY_ALIASES`.

- [ ] **Step 4: Run, verify pass.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_runs.py -v`
Expected: all 4 tests pass.

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/server.py tests/test_study_runs.py
git commit -m "add v3-native _post_study_run_variant; layers variant param overrides"
```

---

## Task 5: `_post_study_variant_add` + `_post_study_variant_delete`

**Why:** v3 variants are pure `study.yaml` entries — `{name, intervention: {description, parameter_overrides, process_overrides}}`. No `composites/<name>.yaml` sidecar needed (param overrides are applied at run time by Task 4). The aliased `_post_investigation_composite_perturb` 404'd because it tried to derive from a `composites/baseline.yaml` sidecar that doesn't exist.

**Files:**
- Modify: `vivarium_dashboard/server.py` (two new handlers + routes)
- Test: `tests/test_study_variants.py` (new)

- [ ] **Step 1: Write the failing tests.** Create `tests/test_study_variants.py`:

```python
"""v3-native Study variant handlers — add/delete variant entries in study.yaml."""
import yaml
import pytest


@pytest.fixture
def _study_ws(tmp_path, monkeypatch):
    import vivarium_dashboard.server as srv
    ws = tmp_path / "ws"
    sd = ws / "studies" / "s1"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 3, "name": "s1", "created": "2026-05-14",
        "status": "ran", "objective": "",
        "baseline": {"composite": "pkg.foo", "params": {}},
        "variants": [], "runs": [], "visualizations": [],
        "comparisons": [], "conclusion": None, "parent_studies": [],
    }))
    monkeypatch.setattr(srv, "WORKSPACE", ws)
    return ws


def test_variant_add_appends_to_study_yaml(_study_ws):
    from vivarium_dashboard.server import _post_study_variant_add_for_test
    resp, code = _post_study_variant_add_for_test(_study_ws, {
        "study": "s1", "name": "hi-sens",
        "description": "triple sensitivity",
        "parameter_overrides": {"sensitivity": 6.0},
    })
    assert code == 200, resp
    spec = yaml.safe_load((_study_ws / "studies" / "s1" / "study.yaml").read_text())
    assert len(spec["variants"]) == 1
    v = spec["variants"][0]
    assert v["name"] == "hi-sens"
    assert v["intervention"]["parameter_overrides"] == {"sensitivity": 6.0}
    assert v["intervention"]["description"] == "triple sensitivity"


def test_variant_add_rejects_duplicate(_study_ws):
    from vivarium_dashboard.server import _post_study_variant_add_for_test
    body = {"study": "s1", "name": "dup", "parameter_overrides": {"a": 1}}
    _post_study_variant_add_for_test(_study_ws, body)
    resp, code = _post_study_variant_add_for_test(_study_ws, body)
    assert code == 409


def test_variant_add_requires_name(_study_ws):
    from vivarium_dashboard.server import _post_study_variant_add_for_test
    resp, code = _post_study_variant_add_for_test(_study_ws, {"study": "s1"})
    assert code == 400


def test_variant_delete_removes_entry(_study_ws):
    from vivarium_dashboard.server import (
        _post_study_variant_add_for_test, _post_study_variant_delete_for_test,
    )
    _post_study_variant_add_for_test(_study_ws, {
        "study": "s1", "name": "gone", "parameter_overrides": {"a": 1}})
    resp, code = _post_study_variant_delete_for_test(
        _study_ws, {"study": "s1", "variant": "gone"})
    assert code == 200
    spec = yaml.safe_load((_study_ws / "studies" / "s1" / "study.yaml").read_text())
    assert spec["variants"] == []


def test_variant_delete_unknown(_study_ws):
    from vivarium_dashboard.server import _post_study_variant_delete_for_test
    resp, code = _post_study_variant_delete_for_test(
        _study_ws, {"study": "s1", "variant": "ghost"})
    assert code == 404
```

- [ ] **Step 2: Run, verify failure.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_variants.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement both handlers.** Add to `server.py`:

```python
def _post_study_variant_add(self, body: dict):
    """POST /api/study-variant-add {study, name, description?,
    parameter_overrides?, process_overrides?}"""
    response, code = _post_study_variant_add_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_variant_add_for_test(ws_root, body):
    study = _study_name_from_body(body)
    variant_name = (body.get("name") or "").strip()
    if not study or not variant_name:
        return {"error": "missing study or variant name"}, 400
    sf = _study_dir(study) / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404

    spec = yaml.safe_load(sf.read_text()) or {}
    variants = spec.setdefault("variants", [])
    if any(v.get("name") == variant_name for v in variants):
        return {"error": f"variant {variant_name!r} already exists"}, 409

    intervention = {"description": body.get("description") or ""}
    if body.get("parameter_overrides"):
        intervention["parameter_overrides"] = body["parameter_overrides"]
    if body.get("process_overrides"):
        intervention["process_overrides"] = body["process_overrides"]
    variants.append({"name": variant_name, "intervention": intervention})
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True, "name": variant_name}, 200


def _post_study_variant_delete(self, body: dict):
    """POST /api/study-variant-delete {study, variant}"""
    response, code = _post_study_variant_delete_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_variant_delete_for_test(ws_root, body):
    study = _study_name_from_body(body)
    variant_name = (body.get("variant") or "").strip()
    if not study or not variant_name:
        return {"error": "missing study or variant"}, 400
    sf = _study_dir(study) / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404

    spec = yaml.safe_load(sf.read_text()) or {}
    variants = spec.get("variants") or []
    remaining = [v for v in variants if v.get("name") != variant_name]
    if len(remaining) == len(variants):
        return {"error": f"variant {variant_name!r} not found"}, 404
    spec["variants"] = remaining
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True}, 200
```

Register routes in `_POST_ROUTE_MAP`:
```python
"/api/study-variant-add": "_post_study_variant_add",
"/api/study-variant-delete": "_post_study_variant_delete",
```
Remove `/api/study-variant-add` from `_POST_STUDY_ALIASES` (it was aliased to `composite-perturb`). `/api/study-variant-delete` was never aliased (Task 4.2 noted it was unwired) — just add it fresh.

- [ ] **Step 4: Run, verify pass.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_variants.py -v`
Expected: all 5 pass.

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/server.py tests/test_study_variants.py
git commit -m "add v3-native study-variant-add + study-variant-delete (study.yaml entries, no sidecar)"
```

---

## Task 6: `_post_study_run_delete` + `_post_study_runs_clear`

**Files:**
- Modify: `vivarium_dashboard/server.py` (two handlers + routes)
- Test: `tests/test_study_runs.py` (extend)

- [ ] **Step 1: Write the failing tests.** Append to `tests/test_study_runs.py`:

```python
def _seed_run(study_ws, run_id, variant=None):
    """Helper: put one run row in the Study's runs.db + study.yaml."""
    import sqlite3
    from vivarium_dashboard.lib.composite_runs import connect
    sd = study_ws / "studies" / "s1"
    db = sd / "runs.db"
    conn = connect(db)
    conn.execute(
        "INSERT INTO runs_meta (run_id, spec_id, label, params_json, "
        "started_at, status) VALUES (?,?,?,?,?,?)",
        (run_id, "pkg.foo", "lbl", "{}", 1.0, "completed"),
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (simulation_id TEXT, step INTEGER, "
        "global_time REAL, state TEXT, PRIMARY KEY (simulation_id, step))"
    )
    conn.execute("INSERT INTO history VALUES (?,?,?,?)", (run_id, 0, 0.0, "{}"))
    conn.commit()
    conn.close()
    sf = sd / "study.yaml"
    spec = yaml.safe_load(sf.read_text())
    spec.setdefault("runs", []).append(
        {"run_id": run_id, "variant": variant, "label": "lbl", "status": "completed"})
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))


def test_run_delete_removes_from_db_and_yaml(_study_ws):
    from vivarium_dashboard.server import _post_study_run_delete_for_test
    _seed_run(_study_ws, "r1")
    _seed_run(_study_ws, "r2")
    resp, code = _post_study_run_delete_for_test(
        _study_ws, {"study": "s1", "run_id": "r1"})
    assert code == 200
    import sqlite3
    conn = sqlite3.connect(str(_study_ws / "studies" / "s1" / "runs.db"))
    meta_ids = [r[0] for r in conn.execute("SELECT run_id FROM runs_meta")]
    hist_ids = [r[0] for r in conn.execute("SELECT DISTINCT simulation_id FROM history")]
    conn.close()
    assert meta_ids == ["r2"]
    assert hist_ids == ["r2"]
    spec = yaml.safe_load((_study_ws / "studies" / "s1" / "study.yaml").read_text())
    assert [r["run_id"] for r in spec["runs"]] == ["r2"]


def test_runs_clear_empties_everything(_study_ws):
    from vivarium_dashboard.server import _post_study_runs_clear_for_test
    _seed_run(_study_ws, "r1")
    _seed_run(_study_ws, "r2")
    resp, code = _post_study_runs_clear_for_test(_study_ws, {"study": "s1"})
    assert code == 200
    import sqlite3
    conn = sqlite3.connect(str(_study_ws / "studies" / "s1" / "runs.db"))
    n_meta = conn.execute("SELECT COUNT(*) FROM runs_meta").fetchone()[0]
    n_hist = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    conn.close()
    assert n_meta == 0 and n_hist == 0
    spec = yaml.safe_load((_study_ws / "studies" / "s1" / "study.yaml").read_text())
    assert spec["runs"] == []
```

- [ ] **Step 2: Run, verify failure.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_runs.py -v -k "delete or clear"`
Expected: ImportError.

- [ ] **Step 3: Implement both handlers.** Add to `server.py`:

```python
def _post_study_run_delete(self, body: dict):
    """POST /api/study-run-delete {study, run_id}"""
    response, code = _post_study_run_delete_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_run_delete_for_test(ws_root, body):
    study = _study_name_from_body(body)
    run_id = (body.get("run_id") or "").strip()
    if not study or not run_id:
        return {"error": "missing study or run_id"}, 400
    study_dir = _study_dir(study)
    sf = study_dir / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404

    db = study_dir / "runs.db"
    if db.is_file():
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("DELETE FROM runs_meta WHERE run_id = ?", (run_id,))
            has_history = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
            ).fetchone()
            if has_history:
                conn.execute("DELETE FROM history WHERE simulation_id = ?", (run_id,))
            conn.commit()
        finally:
            conn.close()

    spec = yaml.safe_load(sf.read_text()) or {}
    spec["runs"] = [r for r in (spec.get("runs") or []) if r.get("run_id") != run_id]
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True}, 200


def _post_study_runs_clear(self, body: dict):
    """POST /api/study-runs-clear {study}"""
    response, code = _post_study_runs_clear_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_runs_clear_for_test(ws_root, body):
    study = _study_name_from_body(body)
    if not study:
        return {"error": "missing study"}, 400
    study_dir = _study_dir(study)
    sf = study_dir / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404

    db = study_dir / "runs.db"
    if db.is_file():
        conn = sqlite3.connect(str(db))
        try:
            conn.execute("DELETE FROM runs_meta")
            has_history = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
            ).fetchone()
            if has_history:
                conn.execute("DELETE FROM history")
            conn.commit()
        finally:
            conn.close()

    spec = yaml.safe_load(sf.read_text()) or {}
    spec["runs"] = []
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True}, 200
```

Register routes in `_POST_ROUTE_MAP`:
```python
"/api/study-run-delete": "_post_study_run_delete",
"/api/study-runs-clear": "_post_study_runs_clear",
```
Remove both from `_POST_STUDY_ALIASES`.

- [ ] **Step 4: Run, verify pass.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_runs.py -v`
Expected: all tests pass (baseline, variant, delete, clear).

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/server.py tests/test_study_runs.py
git commit -m "add v3-native study-run-delete + study-runs-clear"
```

---

## Task 7: `_post_study_comparison_add`

**Why:** the Study Detail view's "Compare selected" button posts a set of run_ids. v3-native: append a `{name, run_ids}` entry to `study.yaml['comparisons']`. (Rendering a comparison view is out of scope for Phase 1.5 — this records the comparison set.)

**Files:**
- Modify: `vivarium_dashboard/server.py` (one handler + route)
- Test: `tests/test_study_variants.py` (extend — it already has the `_study_ws` fixture)

- [ ] **Step 1: Write the failing test.** Append to `tests/test_study_variants.py`:

```python
def test_comparison_add_appends(_study_ws):
    from vivarium_dashboard.server import _post_study_comparison_add_for_test
    resp, code = _post_study_comparison_add_for_test(_study_ws, {
        "study": "s1", "run_ids": ["r1", "r2"]})
    assert code == 200, resp
    spec = yaml.safe_load((_study_ws / "studies" / "s1" / "study.yaml").read_text())
    assert len(spec["comparisons"]) == 1
    assert spec["comparisons"][0]["run_ids"] == ["r1", "r2"]
    assert "name" in spec["comparisons"][0]


def test_comparison_add_requires_two_runs(_study_ws):
    from vivarium_dashboard.server import _post_study_comparison_add_for_test
    resp, code = _post_study_comparison_add_for_test(
        _study_ws, {"study": "s1", "run_ids": ["only-one"]})
    assert code == 400
```

- [ ] **Step 2: Run, verify failure.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_variants.py -v -k comparison`
Expected: ImportError.

- [ ] **Step 3: Implement.** Add to `server.py`:

```python
def _post_study_comparison_add(self, body: dict):
    """POST /api/study-comparison-add {study, run_ids, name?}"""
    response, code = _post_study_comparison_add_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_comparison_add_for_test(ws_root, body):
    study = _study_name_from_body(body)
    run_ids = body.get("run_ids") or []
    if not study:
        return {"error": "missing study"}, 400
    if not isinstance(run_ids, list) or len(run_ids) < 2:
        return {"error": "run_ids must be a list of at least 2 run ids"}, 400
    sf = _study_dir(study) / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404

    spec = yaml.safe_load(sf.read_text()) or {}
    comparisons = spec.setdefault("comparisons", [])
    name = (body.get("name") or "").strip() or f"comparison-{len(comparisons) + 1}"
    comparisons.append({"name": name, "run_ids": list(run_ids)})
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True, "name": name}, 200
```

Register the route in `_POST_ROUTE_MAP`:
```python
"/api/study-comparison-add": "_post_study_comparison_add",
```
Remove `/api/study-comparison-add` from `_POST_STUDY_ALIASES`.

- [ ] **Step 4: Run, verify pass.**

Run: `cd ~/code/vivarium-dashboard && ~/code/viva-munk/.venv/bin/python -m pytest tests/test_study_variants.py -v`
Expected: all 7 pass (5 variant + 2 comparison).

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/server.py tests/test_study_variants.py
git commit -m "add v3-native study-comparison-add (records run_id set in study.yaml)"
```

---

## Task 8: Dead-alias cleanup + frontend wiring + end-to-end verification

**Files:**
- Modify: `vivarium_dashboard/server.py` (audit `_POST_STUDY_ALIASES`)
- Modify: `vivarium_dashboard/static/study-detail.js` (wire `btn-delete-variant`)
- Test: full suite + live e2e

- [ ] **Step 1: Audit the alias map.** In `server.py`, confirm these 7 routes are NO LONGER in `_POST_STUDY_ALIASES` (Tasks 3-7 removed them as they replaced each):
  - `/api/study-run-baseline`, `/api/study-run-variant`, `/api/study-variant-add`, `/api/study-run-delete`, `/api/study-runs-clear`, `/api/study-comparison-add`

```bash
cd ~/code/vivarium-dashboard
grep -nE "study-run-baseline|study-run-variant|study-variant-add|study-run-delete|study-runs-clear|study-comparison-add" vivarium_dashboard/server.py
```
Each should appear ONLY in `_POST_ROUTE_MAP` (pointing at a `_post_study_*` method), NOT in `_POST_STUDY_ALIASES`. If any still appears in `_POST_STUDY_ALIASES`, remove that entry. Routes that SHOULD remain aliased (still backed by investigation handlers, and they work): `/api/study-detail` is GET; `/api/study-set-conclusion`, `/api/study-delete` stay aliased — leave them.

- [ ] **Step 2: Wire `btn-delete-variant` in the frontend.** In `vivarium_dashboard/static/study-detail.js`, find the `_studyDeleteVariant` / `.btn-delete-variant` handler. Task 4.2 left it as an `alert(...)` placeholder. Replace with a real call:

```javascript
  bindAll('.btn-delete-variant', function(btn) {
    var variant = btn.dataset.variant;
    if (!confirm('Delete variant ' + variant + '?')) return;
    api('POST', '/api/study-variant-delete',
        {study: studyName(), variant: variant})
      .then(function(res) {
        if (res.status === 200) location.reload();
        else alert(res.body.error || 'Delete variant failed');
      });
  });
```

(If `bindAll` / `studyName` / `api` differ in the actual file, adapt — the point is: POST `/api/study-variant-delete` with `{study, variant}`.)

- [ ] **Step 3: Run the full Studies test suite.**

```bash
cd ~/code/vivarium-dashboard
~/code/viva-munk/.venv/bin/python -m pytest \
  tests/test_study_runs.py tests/test_study_variants.py \
  tests/test_study_handlers.py tests/test_study_create_from_run.py \
  tests/test_study_aliases.py tests/test_study_dir_resolution.py \
  tests/test_v3_study_validation.py tests/test_study_body_keys.py \
  tests/test_composite_runs.py tests/test_composite_explorer_api.py \
  tests/test_spec_migration.py tests/test_investigations.py -v 2>&1 | tail -30
```
Expected: all pass EXCEPT the one known pre-existing failure (`test_run_investigation_iterates_runs_and_passes_state_doc` — `ModuleNotFoundError: scripts._lib`).

- [ ] **Step 4: End-to-end smoke test — the full Study Detail workflow.**

```bash
cd ~/code/viva-munk
lsof -ti tcp:8765 | xargs -r kill 2>/dev/null; sleep 1
.venv/bin/vivarium-dashboard serve --workspace . --port 8765 > /tmp/dash.log 2>&1 &
until curl -sS -o /dev/null -w "%{http_code}" http://localhost:8765/ 2>/dev/null | grep -q "^200$"; do sleep 0.5; done

# 1. Test run → promote to Study
RUN=$(curl -sS -X POST http://localhost:8765/api/composite-test-run -H "Content-Type: application/json" -d '{"id":"multi_cell.composites.chemotaxis","steps":2}' | python3 -c "import sys,json; print(json.load(sys.stdin).get('simulation_id',''))")
curl -sS -X POST http://localhost:8765/api/study-create-from-run -H "Content-Type: application/json" -d "{\"name\":\"e2e-study\",\"objective\":\"x\",\"description\":\"\",\"source_run_id\":\"$RUN\"}" > /dev/null

# 2. Detail API (was 500 on sim_name)
echo "detail: $(curl -sS -o /dev/null -w '%{http_code}' http://localhost:8765/api/study/e2e-study)"

# 3. Add a variant (was 404 on missing sidecar)
echo "variant-add: $(curl -sS -X POST http://localhost:8765/api/study-variant-add -H 'Content-Type: application/json' -d '{"study":"e2e-study","name":"fast","description":"more steps","parameter_overrides":{"n_steps":3}}' -o /dev/null -w '%{http_code}')"

# 4. Run baseline (was 400 on name-required)
echo "run-baseline: $(curl -sS -X POST http://localhost:8765/api/study-run-baseline -H 'Content-Type: application/json' -d '{"study":"e2e-study","steps":2}' -o /dev/null -w '%{http_code}')"

# 5. Run the variant
echo "run-variant: $(curl -sS -X POST http://localhost:8765/api/study-run-variant -H 'Content-Type: application/json' -d '{"study":"e2e-study","variant":"fast"}' -o /dev/null -w '%{http_code}')"

# 6. Inspect resulting study.yaml
echo "--- study.yaml ---"
cat studies/e2e-study/study.yaml

# 7. Cleanup
rm -rf studies/e2e-study
lsof -ti tcp:8765 | xargs -r kill 2>/dev/null
```

Expected: detail=200, variant-add=200, run-baseline=200, run-variant=200; `study.yaml` shows 1 variant (`fast`) + 3 runs (the promoted one + baseline + variant), each run with a `run_id`.

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/server.py vivarium_dashboard/static/study-detail.js
git commit -m "drop dead study-* aliases; wire btn-delete-variant; phase-1.5 e2e verified"
```

---

## Self-Review

**1. Spec coverage** — the 7 broken routes from the goal:
- `study-run-baseline` → Task 3 ✓
- `study-run-variant` → Task 4 ✓
- `study-variant-add` → Task 5 ✓
- `study-variant-delete` → Task 5 ✓ (also fixes Task 4.2's unwired `btn-delete-variant` — wired in Task 8)
- `study-run-delete` → Task 6 ✓
- `study-runs-clear` → Task 6 ✓
- `study-comparison-add` → Task 7 ✓
- `/api/study/<name>` detail API → Task 1 (sim_name) ✓ — verified in Task 8 Step 4.

**2. Placeholder scan** — Task 2 Step 2 deliberately `pytest.skip`s a characterization test and says so explicitly (the real safety net is the existing `test_composite_explorer_api.py` suite + Task 8 e2e). That's an intentional, explained skip, not a placeholder. No "TBD"/"handle edge cases" elsewhere.

**3. Type consistency** — shared names used consistently across tasks:
- `_run_composite_subprocess(*, pkg, state, steps, db_file, run_id, spec_id, label, sim_name=None, timeout=120)` — defined Task 2, called Tasks 3 & 4.
- `_resolve_study_baseline_state(pkg, spec_id, params) -> (state, err)` — defined Task 3, reused Task 4.
- `_append_study_run(study_dir, run_record)` — defined Task 3, reused Task 4.
- `_study_dir(name)`, `_study_name_from_body(body)` — pre-existing from earlier Studies work; used throughout.
- Every `_post_study_*` handler pairs a thin Handler method with a pure `_post_study_*_for_test(ws_root, body)` function — consistent with the Task 2.3 pattern already in the codebase.

**4. Known carve-outs** — comparison *rendering* is explicitly out of scope (Task 7 records the set only); `process_overrides` variants are stored but Task 4 only applies `parameter_overrides` at run time (process-level perturbation execution is a future concern — the `composites/` dir is kept for it).
