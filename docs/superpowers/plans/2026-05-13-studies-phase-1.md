# Studies — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Phase 1 of the Studies redesign per `docs/superpowers/specs/2026-05-13-studies-design.md`: per-composite `default_n_steps` metadata, a Save-as-Study promote flow, a Study Detail view with six cards, and a one-shot migration of legacy `investigations/` → `studies/`.

**Architecture:** Three repos collaborate. `pbg-superpowers` grows one optional decorator kwarg + dataclass field. `vivarium-dashboard` adds a small set of new handlers, aliases every `/api/investigation-*` route under `/api/study-*` (Phase-1 back-compat), serves a new `/studies/<name>` page, and ships a CLI subcommand for the migration. `viva-munk` (and any other consumer pbg-package) updates each `@composite_generator(...)` to declare `default_n_steps`. Existing investigation handlers are reused as-is — they already cover variants, runs, viz, comparisons, and conclusions.

**Tech Stack:** Python 3.11+, pytest, pyyaml, sqlite3, plum-dispatch (already in use). Vanilla JS for the dashboard frontend (`vivarium_dashboard/static/walkthrough.js`); Jinja2 templates. The viva-munk venv already has everything installed editable.

**Repos affected (relative paths from your `~/code/`):**
- `pbg-superpowers/` — decorator + dataclass field.
- `vivarium-dashboard/` — server handlers, frontend, CLI subcommand. Most of the work.
- `viva-munk/` — per-generator `default_n_steps` values; smoke test that all 9 composites still load.

**Order of work / checkpoint groups:**
1. **Group 1** — `default_n_steps` plumbing across three layers. Smallest, independent. (3 tasks)
2. **Group 2** — Study data model: bump schema_version → 3, drop multi-composite shape, add URL aliases. (3 tasks)
3. **Group 3** — Save-as-Study endpoint + Composite Explorer button + modal. (3 tasks)
4. **Group 4** — Study Detail view: 4 small new handlers + frontend re-skin of Investigation Detail with six cards. (4 tasks)
5. **Group 5** — Migration: `vivarium-dashboard migrate-investigations` CLI subcommand. (2 tasks)

Each group ends in a green test suite + a checkpoint. Groups can be reviewed independently.

---

## Group 1 — `default_n_steps` plumbing

### Task 1.1: Add `default_n_steps` to `@composite_generator` + `GeneratorEntry`

**Files:**
- Modify: `pbg-superpowers/pbg_superpowers/composite_generator.py`
- Test: `pbg-superpowers/tests/test_composite_generator.py` (extend)

- [ ] **Step 1: Write the failing test (extend existing file).** Append to `tests/test_composite_generator.py`:

```python
def test_decorator_accepts_default_n_steps():
    @composite_generator(
        name="t",
        description="",
        parameters={},
        default_n_steps=200,
    )
    def builder(core=None):
        return {}

    entry_id = f"{builder.__module__}.t"
    entry = _REGISTRY[entry_id]
    assert entry.default_n_steps == 200


def test_decorator_default_n_steps_optional():
    @composite_generator(name="t2", description="", parameters={})
    def builder(core=None):
        return {}

    entry_id = f"{builder.__module__}.t2"
    entry = _REGISTRY[entry_id]
    assert entry.default_n_steps is None
```

- [ ] **Step 2: Run tests to confirm they fail.**

Run: `cd ~/code/pbg-superpowers && pytest tests/test_composite_generator.py -v -k default_n_steps`
Expected: 2 failures — `default_n_steps` is not a valid kwarg / `GeneratorEntry` has no field `default_n_steps`.

- [ ] **Step 3: Add the field + kwarg.** Modify `pbg_superpowers/composite_generator.py`. Replace the `GeneratorEntry` dataclass and the `composite_generator` decorator:

```python
@dataclass
class GeneratorEntry:
    """One registered composite-generator function."""

    id: str                           # "<dotted_module>.<name>"
    name: str
    description: str
    parameters: dict[str, dict]       # {name: {type, default, description?}}
    func: Callable[..., dict]
    module: str
    default_n_steps: int | None = None  # framework-owned runtime knob; UI pre-fill


def composite_generator(
    *,
    name: str,
    description: str = "",
    parameters: dict[str, dict] | None = None,
    default_n_steps: int | None = None,
) -> Callable[[Callable[..., dict]], Callable[..., dict]]:
    """Decorator: register a doc-building function.

    `default_n_steps` (optional) is a UI hint for the Composite Explorer's
    `steps` pre-fill. It is NOT a composite-builder kwarg — runtime knobs
    are framework-owned and live next to the generator entry.
    """
    def decorate(fn: Callable[..., dict]) -> Callable[..., dict]:
        entry = GeneratorEntry(
            id=f"{fn.__module__}.{name}",
            name=name,
            description=description,
            parameters=parameters or {},
            func=fn,
            module=fn.__module__,
            default_n_steps=default_n_steps,
        )
        _REGISTRY[entry.id] = entry
        fn._composite_generator_entry = entry
        return fn
    return decorate
```

- [ ] **Step 4: Run all generator tests to confirm green.**

Run: `cd ~/code/pbg-superpowers && pytest tests/test_composite_generator.py -v`
Expected: all pass, including the two new tests.

- [ ] **Step 5: Commit.**

```bash
cd ~/code/pbg-superpowers
git add pbg_superpowers/composite_generator.py tests/test_composite_generator.py
git commit -m "add default_n_steps to @composite_generator for UI pre-fill"
```

### Task 1.2: Propagate `default_n_steps` through `discover_all_composites`

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/lib/composite_lookup.py`
- Test: `vivarium-dashboard/tests/test_composites_kind_module.py` (extend)

- [ ] **Step 1: Find the merge point in `composite_lookup.py`.** Open the file and locate the loop near the bottom of `discover_all_composites` that processes generator entries (the `for gid, entry in merged.items(): if entry.get("kind") != "generator": continue` block — around line 180 today).

- [ ] **Step 2: Write the failing test.** Append to `tests/test_composites_kind_module.py`:

```python
def test_discover_all_composites_propagates_default_n_steps(tmp_path, monkeypatch):
    """Generator entries with default_n_steps surface that field in the catalog."""
    from pbg_superpowers.composite_generator import (
        composite_generator, _REGISTRY,
    )
    from vivarium_dashboard.lib.composite_lookup import discover_all_composites

    _REGISTRY.clear()

    @composite_generator(name="hint", description="", parameters={},
                          default_n_steps=123)
    def builder(core=None):
        return {}

    # Stub: pretend pbg-superpowers discovery returned just our entry
    import pbg_superpowers.composite_discovery as cd

    def fake_discover_all():
        entry_id = f"{builder.__module__}.hint"
        return {
            entry_id: {
                "kind": "generator",
                "id": entry_id,
                "name": "hint",
                "description": "",
                "module": builder.__module__,
                "parameters": {},
                "default_n_steps": 123,
            }
        }

    monkeypatch.setattr(cd, "discover_all", fake_discover_all)

    out = discover_all_composites(tmp_path, "pkg")
    entry_id = f"{builder.__module__}.hint"
    assert entry_id in out
    assert out[entry_id]["default_n_steps"] == 123

    _REGISTRY.clear()
```

- [ ] **Step 3: Run test to verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_composites_kind_module.py::test_discover_all_composites_propagates_default_n_steps -v`
Expected: KeyError or `default_n_steps` not in entry dict.

- [ ] **Step 4: Plumb the field through.** In `composite_lookup.py`, find the generator-merge loop (~line 181) and add `default_n_steps` to the entry dict being added to `out`. Locate this region:

```python
for gid, entry in merged.items():
    if entry.get("kind") != "generator":
        continue
    if gid in out:
        continue  # spec already present; keep spec
    out[gid] = entry
```

The `entry` already comes from `pbg_superpowers.composite_discovery.discover_all`, which itself needs to copy `default_n_steps` from `GeneratorEntry` into its emitted dict. Open `pbg-superpowers/pbg_superpowers/composite_discovery.py`, find the function that converts `GeneratorEntry` instances to dicts (search for `"kind": "generator"`), and add the field:

```python
# Inside the loop that converts each GeneratorEntry → dict
{
    "kind": "generator",
    "id": entry.id,
    "name": entry.name,
    "description": entry.description,
    "module": entry.module,
    "parameters": entry.parameters,
    "default_n_steps": entry.default_n_steps,   # NEW
}
```

(If the exact code structure differs, the change is: wherever GeneratorEntry → dict conversion happens, add `"default_n_steps": entry.default_n_steps`.)

- [ ] **Step 5: Run test to verify pass.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_composites_kind_module.py -v`
Expected: all pass.

- [ ] **Step 6: Commit (vivarium-dashboard + pbg-superpowers).**

```bash
cd ~/code/pbg-superpowers
git add pbg_superpowers/composite_discovery.py
git commit -m "propagate GeneratorEntry.default_n_steps in discover_all() output"

cd ~/code/vivarium-dashboard
git add tests/test_composites_kind_module.py
git commit -m "test default_n_steps survives discover_all_composites pipeline"
```

### Task 1.3: Surface `default_n_steps` in `/api/composites` + frontend pre-fill

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/server.py` (the `_get_composites` handler)
- Modify: `vivarium-dashboard/vivarium_dashboard/static/walkthrough.js` (`onComposite SelectChange` or equivalent)
- Test: `vivarium-dashboard/tests/test_composite_explorer_api.py` (extend)
- Modify: `viva-munk/viva_munk/composites/__init__.py` (add `default_n_steps` to each `@composite_generator` call)

- [ ] **Step 1: Write the failing test.** Append to `tests/test_composite_explorer_api.py`:

```python
def test_api_composites_includes_default_n_steps(_fixtures_workspace):
    """Composites with default_n_steps surface it via /api/composites."""
    handler = _make_handler(_fixtures_workspace, path="/api/composites")
    response = _call_get(handler)
    composites = response["composites"]
    # The fixtures workspace has at least one generator with default_n_steps=42
    # (added by the fixture in step 4 below).
    matching = [c for c in composites if c.get("default_n_steps") == 42]
    assert matching, f"no composite has default_n_steps=42 in {composites}"
```

(If the existing test file uses a different fixture or helper naming, mirror its style. Look at the existing tests in that file for the pattern.)

- [ ] **Step 2: Add a generator with `default_n_steps` to the test fixture.** Edit the fixture pbg package under `vivarium-dashboard/tests/_fixtures/<workspace>/<pbg_pkg>/composites/__init__.py` (find the one used by the existing test_composite_explorer_api tests). Add one decorated generator:

```python
from pbg_superpowers.composite_generator import composite_generator


@composite_generator(name="hint_test", description="", parameters={},
                      default_n_steps=42)
def hint_test(core=None):
    return {}
```

- [ ] **Step 3: Run test to verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_composite_explorer_api.py::test_api_composites_includes_default_n_steps -v`
Expected: assertion failure — `default_n_steps` missing from API response.

- [ ] **Step 4: Update `_get_composites` to include the field.** Find `_get_composites` in `server.py` (around the `/api/composites` route). It currently returns composites with `id`, `kind`, `description`, `module`. Add `default_n_steps`:

```python
out = []
for spec_id, rec in catalog.items():
    out.append({
        "id": spec_id,
        "kind": rec.get("kind", "spec"),
        "name": rec.get("name") or spec_id.rsplit(".", 1)[-1],
        "description": rec.get("description", ""),
        "module": rec.get("module", ""),
        "default_n_steps": rec.get("default_n_steps"),  # NEW
    })
return self._json({"composites": out}, 200)
```

(Adapt the field names to match the existing structure in your `_get_composites`. The crucial addition is the `default_n_steps` line.)

- [ ] **Step 5: Run test to verify pass.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_composite_explorer_api.py -v`
Expected: all pass.

- [ ] **Step 6: Update the Composite Explorer frontend to pre-fill steps.** Find `walkthrough.js` and locate the function that runs when a composite is selected (search for `#ce-steps`). Add a pre-fill on selection. In the function that loads composite details into the panel:

```javascript
function _ceOnCompositeSelect(composite) {
  // ... existing setup ...
  var stepsInput = document.getElementById('ce-steps');
  if (stepsInput) {
    if (composite.default_n_steps != null) {
      stepsInput.value = composite.default_n_steps;
    } else {
      stepsInput.value = 5;  // fallback unchanged
    }
  }
}
```

(The exact function/handler name will depend on existing code; find where `#ce-steps` is read and the composite metadata is loaded.)

- [ ] **Step 7: Wire `default_n_steps` into each viva-munk generator.** Edit `~/code/viva-munk/viva_munk/composites/__init__.py`. Look up the curated `n_steps` per experiment in `viva_munk/experiments/registry.py` and pass each through. Example for chemotaxis (apply analogously to all nine):

```python
@composite_generator(
    name="chemotaxis",
    description="Run/tumble chemotaxis up a static 2D ligand gradient.",
    default_n_steps=200,  # NEW — from registry.py chemotaxis block
)
def chemotaxis(core=None) -> dict:
    return chemotaxis_document()
```

Values to set (cross-check against `viva_munk/experiments/registry.py` first, but these are sensible if registry doesn't have one):

| Composite | `default_n_steps` |
|---|---|
| attachment | 200 |
| bending_pressure | 200 |
| biofilm | 500 |
| chemotaxis | 200 |
| daughter_machine | 300 |
| glucose_growth | 300 |
| inclusion_bodies | 500 |
| mother_machine | 300 |
| quorum_sensing | 300 |

- [ ] **Step 8: Smoke-test viva-munk.** Bounce the dashboard and verify defaults pre-fill:

```bash
cd ~/code/viva-munk
lsof -ti tcp:8765 | xargs -r kill; sleep 1
.venv/bin/vivarium-dashboard serve --workspace . --port 8765 &
sleep 3
curl -sS http://localhost:8765/api/composites | python3 -c "
import sys, json
d = json.load(sys.stdin)
for c in d.get('composites', []):
    if 'viva_munk.composites' in c.get('id', ''):
        print(c['id'], '->', c.get('default_n_steps'))
"
```

Expected output: each composite prints with the curated `default_n_steps` value (e.g. `viva_munk.composites.chemotaxis -> 200`).

- [ ] **Step 9: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add tests/_fixtures tests/test_composite_explorer_api.py \
        vivarium_dashboard/server.py vivarium_dashboard/static/walkthrough.js
git commit -m "surface default_n_steps in /api/composites + CE pre-fill"

cd ~/code/viva-munk
git add viva_munk/composites/__init__.py
git commit -m "declare default_n_steps for each composite generator"
```

**Checkpoint 1:** With Group 1 done, the Composite Explorer now pre-fills `steps` per composite. Skip to Group 2 to land the Study data model.

---

## Group 2 — Study data model + URL aliases

### Task 2.1: Bump `schema_version` 2 → 3 + drop multi-composite

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/lib/investigations.py` (validator)
- Modify: `vivarium-dashboard/vivarium_dashboard/lib/spec_migration.py` (already has v1→v2 migration; add v2→v3)
- Test: `vivarium-dashboard/tests/test_spec_migration.py` (extend)

- [ ] **Step 1: Write the failing test.** Append to `tests/test_spec_migration.py`:

```python
def test_migrate_v2_to_v3_lifts_first_composite_to_baseline(tmp_path):
    """v3 has `baseline: {composite, params}` + drops `composites: [...]`."""
    from vivarium_dashboard.lib.spec_migration import migrate_v2_to_v3

    v2 = {
        "schema_version": 2,
        "name": "x",
        "composites": [
            {"name": "main", "source": "pkg.composites.foo", "parameters": {"a": 1}},
        ],
        "runs": [],
        "variants": [],
        "conclusion": None,
    }
    v3 = migrate_v2_to_v3(v2)
    assert v3["schema_version"] == 3
    assert v3["baseline"] == {"composite": "pkg.composites.foo", "params": {"a": 1}}
    assert "composites" not in v3
    assert v3.get("objective") == ""
    assert v3.get("parent_studies") == []


def test_migrate_v2_to_v3_warns_on_multi_composite():
    """If v2 has >1 composite, migration keeps the first + emits a warning."""
    import warnings
    from vivarium_dashboard.lib.spec_migration import migrate_v2_to_v3

    v2 = {
        "schema_version": 2,
        "name": "y",
        "composites": [
            {"name": "main", "source": "pkg.a"},
            {"name": "alt",  "source": "pkg.b"},
        ],
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v3 = migrate_v2_to_v3(v2)
    msgs = [str(w.message) for w in caught]
    assert any("dropped 1 extra composite" in m for m in msgs)
    assert v3["baseline"]["composite"] == "pkg.a"


def test_migrate_v2_to_v3_idempotent():
    from vivarium_dashboard.lib.spec_migration import migrate_v2_to_v3
    v3_already = {"schema_version": 3, "baseline": {"composite": "x"}}
    assert migrate_v2_to_v3(v3_already) is v3_already
```

- [ ] **Step 2: Run tests to verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_spec_migration.py -v -k v2_to_v3`
Expected: ImportError — `migrate_v2_to_v3` not defined.

- [ ] **Step 3: Implement `migrate_v2_to_v3`.** Open `vivarium_dashboard/lib/spec_migration.py` and append:

```python
import warnings


def migrate_v2_to_v3(spec: dict) -> dict:
    """Migrate a schema_version=2 investigation spec to schema_version=3 study.

    Transforms:
      - Drop `composites: [...]` multi-composite list; promote first as
        `baseline.composite`. Warn if there were multiple.
      - Lift first composite's `parameters: {...}` into `baseline.params`.
      - Add empty `objective: ""` and `parent_studies: []` (reserved).
      - Bump schema_version to 3.

    Idempotent: returns unchanged if already v3.
    """
    if spec.get("schema_version") == 3:
        return spec

    out = dict(spec)
    out["schema_version"] = 3
    out.setdefault("objective", "")
    out.setdefault("parent_studies", [])

    composites = spec.get("composites") or []
    if composites:
        first = composites[0]
        out["baseline"] = {
            "composite": first.get("source") or first.get("name", ""),
            "params": first.get("parameters", {}) or {},
        }
        if len(composites) > 1:
            warnings.warn(
                f"v2→v3 migration: dropped {len(composites)-1} extra composite(s) "
                f"from study {spec.get('name', '?')!r}; Phase 1 Studies are "
                f"single-composite. Recreate as variants if needed.",
                UserWarning,
                stacklevel=2,
            )
        out.pop("composites", None)
    elif "composite" in spec:
        out["baseline"] = {
            "composite": spec["composite"],
            "params": spec.get("parameters", {}) or {},
        }
        out.pop("composite", None)
        out.pop("parameters", None)

    return out
```

- [ ] **Step 4: Run tests to verify pass.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_spec_migration.py -v`
Expected: all pass.

- [ ] **Step 5: Update `load_spec` in `investigations.py` to auto-migrate on load.** Find `load_spec` (or the equivalent loader) in `lib/investigations.py` and add the migration step right after parsing YAML:

```python
def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(path.read_text()) or {}
    # Phase 1 transition: auto-migrate v2 → v3 on read.
    from vivarium_dashboard.lib.spec_migration import migrate_v2_to_v3
    spec = migrate_v2_to_v3(spec)
    # ... existing validation ...
    return spec
```

- [ ] **Step 6: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/lib/spec_migration.py \
        vivarium_dashboard/lib/investigations.py \
        tests/test_spec_migration.py
git commit -m "add v2→v3 schema migration; lift first composite to baseline"
```

### Task 2.2: Alias `/api/investigation-*` routes under `/api/study-*`

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/server.py` (the route dispatcher table)
- Test: `vivarium-dashboard/tests/test_investigations.py` (extend or new test file `test_study_aliases.py`)

- [ ] **Step 1: Write the failing alias test.** Create `tests/test_study_aliases.py`:

```python
"""Verify /api/study-* aliases hit the same handlers as /api/investigation-*."""
import pytest


# These pairs are aliased — both must work identically.
ALIASED_ROUTES = [
    ("/api/investigations",         "/api/studies"),
    ("/api/investigation-detail",   "/api/study-detail"),
    ("/api/investigation-run",      "/api/study-run-baseline"),
    ("/api/investigation-run-one",  "/api/study-run-variant"),
    ("/api/investigation-composite-perturb", "/api/study-variant-add"),
    ("/api/investigation-set-conclusions",   "/api/study-set-conclusion"),
    ("/api/investigation-comparison-add",    "/api/study-comparison-add"),
]


@pytest.mark.parametrize("old,new", ALIASED_ROUTES)
def test_study_alias_routes_resolve(old, new, _fixtures_workspace):
    """Both prefixes route to the same handler dispatch."""
    from vivarium_dashboard.server import _DashboardHandler

    # Inspect the dispatch tables on the handler class. _GET and _POST are
    # the route → method dicts (see existing dispatcher in server.py).
    get_routes = getattr(_DashboardHandler, "_get_routes", {})
    post_routes = getattr(_DashboardHandler, "_post_routes", {})

    # Either both old + new are in GET, or both in POST.
    if old in get_routes:
        assert new in get_routes, f"alias {new} missing from GET routes"
        assert get_routes[new] is get_routes[old]
    elif old in post_routes:
        assert new in post_routes, f"alias {new} missing from POST routes"
        assert post_routes[new] is post_routes[old]
    else:
        pytest.fail(f"original route {old} not in dispatcher tables")
```

(If `server.py` doesn't expose `_get_routes` / `_post_routes` as class attributes, refactor to expose them — they're used by the dispatcher anyway. Or replace this test with an end-to-end HTTP call that hits both URLs and confirms identical responses. The principle is the same: both prefixes work.)

- [ ] **Step 2: Run test to verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_aliases.py -v`
Expected: failure — `/api/study-*` routes don't exist yet.

- [ ] **Step 3: Add aliases.** In `server.py`, find the dispatcher table(s). Each `/api/investigation-X` entry gets a sibling `/api/study-Y` pointing at the same method. Aliasing map:

```python
# At module scope in server.py, define the mapping once for clarity.
_STUDY_ALIASES = {
    # GET routes
    "/api/investigations":         "/api/studies",
    "/api/investigation-detail":   "/api/study-detail",
    "/api/investigation-viz-html": "/api/study-viz-html",
    "/api/investigation-composites": "/api/study-composites",
    "/api/investigation-state-tree": "/api/study-state-tree",
    # POST routes
    "/api/investigation-create":     "/api/study-create",
    "/api/investigation-delete":     "/api/study-delete",
    "/api/investigation-run":        "/api/study-run-baseline",
    "/api/investigation-run-one":    "/api/study-run-variant",
    "/api/investigation-run-delete": "/api/study-run-delete",
    "/api/investigation-runs-clear": "/api/study-runs-clear",
    "/api/investigation-render-viz": "/api/study-viz-render",
    "/api/investigation-add-viz":    "/api/study-viz-add",
    "/api/investigation-composite-perturb": "/api/study-variant-add",
    "/api/investigation-composite-rebuild": "/api/study-variant-rebuild",
    "/api/investigation-set-conclusions":   "/api/study-set-conclusion",
    "/api/investigation-set-observables":   "/api/study-set-observables",
    "/api/investigation-set-overview":      "/api/study-set-description",
    "/api/investigation-comparison-add":    "/api/study-comparison-add",
    "/api/investigation-comparison-update": "/api/study-comparison-update",
    "/api/investigation-group-add":         "/api/study-group-add",
    "/api/investigation-group-update":      "/api/study-group-update",
}
```

Then in the dispatcher init, after the original routes are registered, walk this map and copy the handler:

```python
# After existing route table construction:
for old_route, new_route in _STUDY_ALIASES.items():
    if old_route in self._get_routes:
        self._get_routes[new_route] = self._get_routes[old_route]
    if old_route in self._post_routes:
        self._post_routes[new_route] = self._post_routes[old_route]
```

(Adapt to the actual structure of the dispatcher. The key principle: every `/api/investigation-*` route gets a `/api/study-*` sibling pointing at the same function.)

- [ ] **Step 4: Run alias tests + existing investigation tests to verify pass + no regression.**

```bash
cd ~/code/vivarium-dashboard
pytest tests/test_study_aliases.py tests/test_investigations.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit.**

```bash
git add vivarium_dashboard/server.py tests/test_study_aliases.py
git commit -m "alias /api/study-* routes to existing /api/investigation-* handlers"
```

### Task 2.3: Add 4 new small handlers (set-objective, set-baseline-params, rename, export)

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/server.py` (4 new POST handlers)
- Test: `vivarium-dashboard/tests/test_study_aliases.py` (extend) or new `tests/test_study_handlers.py`

- [ ] **Step 1: Write failing tests for the four new handlers.** Create `tests/test_study_handlers.py`:

```python
"""Tests for the four Study-specific handlers added in Phase 1."""
import json
import yaml
import shutil
from pathlib import Path

import pytest


@pytest.fixture
def _study_workspace(tmp_path):
    """Workspace with one minimal v3 study."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "workspace.yaml").write_text("schema_version: 2\nname: ws\ncreated: \"2026-05-13\"\nplugin_version: 0.6.1\npackage_path: pkg\n")
    sd = ws / "studies" / "s1"
    sd.mkdir(parents=True)
    (sd / "study.yaml").write_text(yaml.safe_dump({
        "schema_version": 3,
        "name": "s1",
        "created": "2026-05-13",
        "status": "ran",
        "objective": "",
        "baseline": {"composite": "pkg.composites.foo", "params": {}},
        "variants": [],
        "runs": [],
        "visualizations": [],
        "conclusion": None,
        "parent_studies": [],
    }))
    return ws


def test_set_objective_updates_yaml(_study_workspace):
    from vivarium_dashboard.server import _post_study_set_objective_for_test
    body = {"study": "s1", "text": "Does X cause Y?"}
    resp, code = _post_study_set_objective_for_test(_study_workspace, body)
    assert code == 200
    spec = yaml.safe_load((_study_workspace / "studies" / "s1" / "study.yaml").read_text())
    assert spec["objective"] == "Does X cause Y?"


def test_set_baseline_params_updates_yaml(_study_workspace):
    from vivarium_dashboard.server import _post_study_set_baseline_params_for_test
    body = {"study": "s1", "params": {"a": 1, "n_steps": 50}}
    resp, code = _post_study_set_baseline_params_for_test(_study_workspace, body)
    assert code == 200
    spec = yaml.safe_load((_study_workspace / "studies" / "s1" / "study.yaml").read_text())
    assert spec["baseline"]["params"] == {"a": 1, "n_steps": 50}


def test_rename_moves_directory_and_updates_name(_study_workspace):
    from vivarium_dashboard.server import _post_study_rename_for_test
    body = {"study": "s1", "new_name": "renamed-study"}
    resp, code = _post_study_rename_for_test(_study_workspace, body)
    assert code == 200
    assert (_study_workspace / "studies" / "renamed-study" / "study.yaml").is_file()
    assert not (_study_workspace / "studies" / "s1").exists()
    spec = yaml.safe_load((_study_workspace / "studies" / "renamed-study" / "study.yaml").read_text())
    assert spec["name"] == "renamed-study"


def test_rename_refuses_collision(_study_workspace):
    # Create a sibling
    (_study_workspace / "studies" / "s2").mkdir()
    (_study_workspace / "studies" / "s2" / "study.yaml").write_text("name: s2")
    from vivarium_dashboard.server import _post_study_rename_for_test
    body = {"study": "s1", "new_name": "s2"}
    resp, code = _post_study_rename_for_test(_study_workspace, body)
    assert code == 409


def test_export_returns_zip_bytes(_study_workspace):
    from vivarium_dashboard.server import _study_export_zip
    data = _study_export_zip(_study_workspace, "s1")
    # First 4 bytes of a zip file are PK\x03\x04
    assert data[:4] == b"PK\x03\x04"
```

Note the `_for_test` suffix on extracted handler bodies — these are pure functions that take the workspace path + body dict and return `(response_dict, http_code)`. Refactoring the existing handlers to expose this shape makes testing trivial and avoids spinning up the full HTTP server in tests. Existing handlers may already follow this pattern; if not, the refactor is part of this task.

- [ ] **Step 2: Run tests to verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_handlers.py -v`
Expected: ImportError — the 4 functions don't exist.

- [ ] **Step 3: Implement the four handlers.** Add to `server.py`:

```python
def _post_study_set_objective(self, body: dict):
    """POST /api/study-set-objective {study, text}"""
    response, code = _post_study_set_objective_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_set_objective_for_test(ws_root, body):
    name = (body.get("study") or "").strip()
    text = body.get("text") or ""
    if not name:
        return {"error": "missing study"}, 400
    sf = ws_root / "studies" / name / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404
    spec = yaml.safe_load(sf.read_text()) or {}
    spec["objective"] = text
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True}, 200


def _post_study_set_baseline_params(self, body: dict):
    """POST /api/study-set-baseline-params {study, params}"""
    response, code = _post_study_set_baseline_params_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_set_baseline_params_for_test(ws_root, body):
    name = (body.get("study") or "").strip()
    params = body.get("params")
    if not name or not isinstance(params, dict):
        return {"error": "missing study or params"}, 400
    sf = ws_root / "studies" / name / "study.yaml"
    if not sf.is_file():
        return {"error": "study not found"}, 404
    spec = yaml.safe_load(sf.read_text()) or {}
    spec.setdefault("baseline", {})["params"] = params
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True}, 200


_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def _post_study_rename(self, body: dict):
    """POST /api/study-rename {study, new_name}"""
    response, code = _post_study_rename_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_rename_for_test(ws_root, body):
    name = (body.get("study") or "").strip()
    new_name = (body.get("new_name") or "").strip()
    if not name or not new_name:
        return {"error": "missing study or new_name"}, 400
    if not _SLUG_RE.match(new_name):
        return {"error": "new_name must be lowercase + dashes"}, 400
    src = ws_root / "studies" / name
    dst = ws_root / "studies" / new_name
    if not src.is_dir():
        return {"error": "study not found"}, 404
    if dst.exists():
        return {"error": f"study {new_name!r} already exists"}, 409
    src.rename(dst)
    sf = dst / "study.yaml"
    spec = yaml.safe_load(sf.read_text()) or {}
    spec["name"] = new_name
    sf.write_text(yaml.safe_dump(spec, sort_keys=False))
    return {"ok": True, "name": new_name}, 200


def _get_study_export(self):
    """GET /api/study-export?study=<name>"""
    qs = urlparse(self.path).query
    params = parse_qs(qs)
    name = (params.get("study", [""])[0] or "").strip()
    if not name:
        return self._json({"error": "missing study"}, 400)
    src = WORKSPACE / "studies" / name
    if not src.is_dir():
        return self._json({"error": "study not found"}, 404)
    data = _study_export_zip(WORKSPACE, name)
    self.send_response(200)
    self.send_header("Content-Type", "application/zip")
    self.send_header("Content-Disposition", f'attachment; filename="{name}.zip"')
    self.send_header("Content-Length", str(len(data)))
    self.end_headers()
    self.wfile.write(data)


def _study_export_zip(ws_root, name):
    """Zip studies/<name>/ to bytes."""
    import io, zipfile
    src = ws_root / "studies" / name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in src.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src.parent))
    return buf.getvalue()
```

Register the routes in the dispatcher table:

```python
self._post_routes["/api/study-set-objective"]      = self._post_study_set_objective
self._post_routes["/api/study-set-baseline-params"] = self._post_study_set_baseline_params
self._post_routes["/api/study-rename"]              = self._post_study_rename
self._get_routes["/api/study-export"]               = self._get_study_export
```

- [ ] **Step 4: Run tests to verify pass.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_handlers.py -v`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add vivarium_dashboard/server.py tests/test_study_handlers.py
git commit -m "add 4 Study handlers: set-objective, set-baseline-params, rename, export"
```

**Checkpoint 2:** Group 2 done. The schema can migrate, all URL aliases are in place, and the four new Study handlers exist. The dashboard reads/writes Study YAMLs correctly; nothing user-visible yet beyond the API surface.

---

## Group 3 — Save-as-Study endpoint + Composite Explorer button

### Task 3.1: Implement `/api/study-create-from-run`

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/server.py` (new POST handler `_post_study_create_from_run`)
- Modify: `vivarium-dashboard/vivarium_dashboard/lib/composite_runs.py` (helper: `copy_run_to_new_db`)
- Test: `vivarium-dashboard/tests/test_study_create_from_run.py` (new)

- [ ] **Step 1: Write the failing test.** Create `tests/test_study_create_from_run.py`:

```python
"""End-to-end test for POST /api/study-create-from-run."""
import sqlite3
import yaml
from pathlib import Path

import pytest


@pytest.fixture
def _ws_with_scratch_run(tmp_path):
    """Workspace with one completed test-run in .pbg/composite-runs.db."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "workspace.yaml").write_text(
        "schema_version: 2\nname: ws\ncreated: \"2026-05-13\"\nplugin_version: 0.6.1\npackage_path: pkg\n"
    )
    pbg = ws / ".pbg"
    pbg.mkdir()
    db = pbg / "composite-runs.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE runs_meta (
            run_id TEXT PRIMARY KEY,
            spec_id TEXT NOT NULL,
            label TEXT,
            params_json TEXT,
            started_at REAL NOT NULL,
            completed_at REAL,
            n_steps INTEGER,
            status TEXT NOT NULL
        );
        CREATE TABLE history (
            simulation_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            global_time REAL,
            state TEXT NOT NULL,
            PRIMARY KEY (simulation_id, step)
        );
    """)
    conn.execute(
        "INSERT INTO runs_meta VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("rid1", "pkg.composites.foo", "test", "{}",
         1715620800.0, 1715620812.0, 10, "completed"),
    )
    conn.executemany(
        "INSERT INTO history VALUES (?, ?, ?, ?)",
        [("rid1", i, float(i), '{"x": ' + str(i) + '}') for i in range(10)],
    )
    conn.commit()
    conn.close()
    return ws


def test_create_from_run_writes_study_yaml(_ws_with_scratch_run):
    from vivarium_dashboard.server import _post_study_create_from_run_for_test
    body = {
        "name": "my-study",
        "objective": "Why?",
        "description": "",
        "source_run_id": "rid1",
    }
    resp, code = _post_study_create_from_run_for_test(_ws_with_scratch_run, body)
    assert code == 200, resp
    sd = _ws_with_scratch_run / "studies" / "my-study"
    assert sd.is_dir()
    spec = yaml.safe_load((sd / "study.yaml").read_text())
    assert spec["schema_version"] == 3
    assert spec["name"] == "my-study"
    assert spec["objective"] == "Why?"
    assert spec["baseline"]["composite"] == "pkg.composites.foo"
    assert len(spec["runs"]) == 1
    assert spec["runs"][0]["run_id"] == "rid1"


def test_create_from_run_copies_history_rows(_ws_with_scratch_run):
    from vivarium_dashboard.server import _post_study_create_from_run_for_test
    body = {"name": "my-study", "objective": "?",
            "description": "", "source_run_id": "rid1"}
    resp, code = _post_study_create_from_run_for_test(_ws_with_scratch_run, body)
    assert code == 200

    db = _ws_with_scratch_run / "studies" / "my-study" / "runs.db"
    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT COUNT(*) FROM history WHERE simulation_id=?", ("rid1",)).fetchone()
    assert rows[0] == 10
    meta = conn.execute("SELECT run_id, spec_id FROM runs_meta").fetchall()
    assert meta == [("rid1", "pkg.composites.foo")]
    conn.close()


def test_create_from_run_leaves_scratch_untouched(_ws_with_scratch_run):
    from vivarium_dashboard.server import _post_study_create_from_run_for_test
    body = {"name": "my-study", "objective": "?",
            "description": "", "source_run_id": "rid1"}
    _post_study_create_from_run_for_test(_ws_with_scratch_run, body)
    scratch = _ws_with_scratch_run / ".pbg" / "composite-runs.db"
    conn = sqlite3.connect(str(scratch))
    assert conn.execute("SELECT COUNT(*) FROM runs_meta").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 10
    conn.close()


def test_create_from_run_refuses_collision(_ws_with_scratch_run):
    from vivarium_dashboard.server import _post_study_create_from_run_for_test
    body = {"name": "my-study", "objective": "?",
            "description": "", "source_run_id": "rid1"}
    _post_study_create_from_run_for_test(_ws_with_scratch_run, body)
    resp, code = _post_study_create_from_run_for_test(_ws_with_scratch_run, body)
    assert code == 409


def test_create_from_run_missing_source(_ws_with_scratch_run):
    from vivarium_dashboard.server import _post_study_create_from_run_for_test
    body = {"name": "n", "objective": "?", "description": "", "source_run_id": "nope"}
    resp, code = _post_study_create_from_run_for_test(_ws_with_scratch_run, body)
    assert code == 404
```

- [ ] **Step 2: Run tests to verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_create_from_run.py -v`
Expected: ImportError — handler doesn't exist.

- [ ] **Step 3: Implement the helper in `composite_runs.py`.** Append to `vivarium_dashboard/lib/composite_runs.py`:

```python
def copy_run_to_new_db(src_db: Path, dst_db: Path, run_id: str) -> int:
    """Copy one run's metadata + history rows from src_db to dst_db.

    Both DBs must use the same schema (runs_meta + history). Bootstraps
    dst_db's schema if missing. Returns the count of history rows copied.
    """
    src = sqlite3.connect(str(src_db))
    src.row_factory = sqlite3.Row

    dst = connect(dst_db)  # bootstraps runs_meta
    # SQLiteEmitter creates the history table lazily on first write; do it eagerly here.
    dst.executescript("""
        CREATE TABLE IF NOT EXISTS history (
            simulation_id TEXT NOT NULL,
            step INTEGER NOT NULL,
            global_time REAL,
            state TEXT NOT NULL,
            PRIMARY KEY (simulation_id, step)
        );
    """)

    meta = src.execute(
        "SELECT * FROM runs_meta WHERE run_id = ?", (run_id,)
    ).fetchone()
    if meta is None:
        src.close(); dst.close()
        raise KeyError(run_id)
    dst.execute(
        "INSERT INTO runs_meta (run_id, spec_id, label, params_json, "
        "started_at, completed_at, n_steps, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (meta["run_id"], meta["spec_id"], meta["label"], meta["params_json"],
         meta["started_at"], meta["completed_at"], meta["n_steps"], meta["status"]),
    )

    rows = src.execute(
        "SELECT step, global_time, state FROM history WHERE simulation_id = ?",
        (run_id,),
    ).fetchall()
    dst.executemany(
        "INSERT INTO history (simulation_id, step, global_time, state) "
        "VALUES (?, ?, ?, ?)",
        [(run_id, r["step"], r["global_time"], r["state"]) for r in rows],
    )
    dst.commit()
    src.close(); dst.close()
    return len(rows)
```

- [ ] **Step 4: Implement `_post_study_create_from_run` in `server.py`.**

```python
def _post_study_create_from_run(self, body: dict):
    """POST /api/study-create-from-run {name, objective, description?, source_run_id}"""
    response, code = _post_study_create_from_run_for_test(WORKSPACE, body)
    return self._json(response, code)


def _post_study_create_from_run_for_test(ws_root, body):
    import datetime
    import shutil
    import tempfile
    from vivarium_dashboard.lib.composite_runs import copy_run_to_new_db

    name = (body.get("name") or "").strip()
    objective = body.get("objective") or ""
    description = body.get("description") or ""
    source_run_id = (body.get("source_run_id") or "").strip()

    if not name or not source_run_id:
        return {"error": "missing name or source_run_id"}, 400
    if not _SLUG_RE.match(name):
        return {"error": "name must be lowercase + dashes"}, 400

    studies_root = ws_root / "studies"
    studies_root.mkdir(parents=True, exist_ok=True)
    dst = studies_root / name
    if dst.exists():
        return {"error": f"study {name!r} already exists"}, 409

    scratch = ws_root / ".pbg" / "composite-runs.db"
    if not scratch.is_file():
        return {"error": "no scratchpad DB"}, 404

    # Read the source run's metadata once to populate baseline.
    src = sqlite3.connect(str(scratch))
    src.row_factory = sqlite3.Row
    meta = src.execute(
        "SELECT spec_id, params_json, n_steps FROM runs_meta WHERE run_id = ?",
        (source_run_id,),
    ).fetchone()
    src.close()
    if meta is None:
        return {"error": "source_run_id not in scratchpad"}, 404
    spec_id = meta["spec_id"]
    try:
        params = json.loads(meta["params_json"] or "{}")
    except (TypeError, ValueError):
        params = {}
    n_steps = int(meta["n_steps"] or 0)
    if n_steps and "n_steps" not in params:
        params["n_steps"] = n_steps

    # Build the study atomically: write to a temp dir, then rename.
    with tempfile.TemporaryDirectory(dir=str(studies_root)) as tmp:
        tmp_path = Path(tmp) / "build"
        tmp_path.mkdir()
        (tmp_path / "composites").mkdir()
        (tmp_path / "viz").mkdir()

        # Copy the run history into the new DB.
        copy_run_to_new_db(scratch, tmp_path / "runs.db", source_run_id)

        spec = {
            "schema_version": 3,
            "name": name,
            "created": datetime.date.today().isoformat(),
            "status": "ran",
            "objective": objective,
            "description": description,
            "baseline": {"composite": spec_id, "params": params},
            "variants": [],
            "runs": [{
                "run_id": source_run_id,
                "variant": None,
                "label": "promoted from scratchpad",
                "status": "completed",
            }],
            "visualizations": [],
            "conclusion": None,
            "parent_studies": [],
        }
        (tmp_path / "study.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))

        # Atomic rename: tmp/build → studies/<name>.
        tmp_path.rename(dst)

    return {"study": name, "url": f"/studies/{name}"}, 200
```

Register in the POST table:

```python
self._post_routes["/api/study-create-from-run"] = self._post_study_create_from_run
```

- [ ] **Step 5: Run tests to verify pass.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_create_from_run.py -v`
Expected: all 5 tests pass.

- [ ] **Step 6: Commit.**

```bash
git add vivarium_dashboard/server.py vivarium_dashboard/lib/composite_runs.py tests/test_study_create_from_run.py
git commit -m "add /api/study-create-from-run; promote scratchpad run to a Study"
```

### Task 3.2: Add the "Save as Study" button + modal to Composite Explorer

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/static/walkthrough.js`
- Modify: `vivarium-dashboard/vivarium_dashboard/templates/composite-explorer.html` (or wherever the post-run viz HTML is rendered)

- [ ] **Step 1: Locate the post-run viz render spot.** Search for where the test-run result is appended to the Composite Explorer panel after a successful `/api/composite-test-run` call. The button must show up there, only after success.

Run: `cd ~/code/vivarium-dashboard && grep -nE "composite-test-run|viz_html|simulation_id" vivarium_dashboard/static/walkthrough.js | head -15`

The handler usually looks like `fetch('/api/composite-test-run', ...).then(r => r.json()).then(d => { ... render viz ...})`. The button goes inside that handler's success branch.

- [ ] **Step 2: Add the button + modal to the template.** Find the Composite Explorer template (likely `templates/composite-explorer.html` or rendered inline by `walkthrough.js`). Add the hidden modal markup near the panel:

```html
<div id="save-as-study-modal" class="modal" style="display:none">
  <div class="modal-body">
    <h3>Save as Study</h3>
    <label>Study name <input id="sas-name" type="text" /></label>
    <label>Objective <textarea id="sas-objective" rows="3"></textarea></label>
    <label>Description <input id="sas-description" type="text" /></label>
    <div class="modal-error" id="sas-error" style="display:none;color:red"></div>
    <div class="modal-actions">
      <button id="sas-cancel">Cancel</button>
      <button id="sas-create" class="primary">Create Study</button>
    </div>
  </div>
</div>
```

Add the button to the test-run result panel:

```html
<button id="save-as-study-btn" style="display:none">Save as Study</button>
```

- [ ] **Step 3: Wire the JS — show the button after a successful run.** In `walkthrough.js`, find the test-run success handler. After viz renders, expose the button:

```javascript
function _ceOnTestRunSuccess(result) {
  // ... existing viz render ...
  window._ceLastRunId = result.simulation_id;
  var btn = document.getElementById('save-as-study-btn');
  if (btn) {
    btn.style.display = 'inline-block';
    btn.onclick = _ceOpenSaveAsStudyModal;
  }
}

function _ceOnTestRunStart() {
  var btn = document.getElementById('save-as-study-btn');
  if (btn) btn.style.display = 'none';   // hide during a run
}

function _ceOpenSaveAsStudyModal() {
  var nameInput = document.getElementById('sas-name');
  // Pre-fill: <composite-name>-<YYMMDD>
  var composite = window._ceSelectedComposite || '';
  var leaf = composite.split('.').pop();
  var date = new Date();
  var yymmdd = String(date.getFullYear()).slice(2)
    + String(date.getMonth() + 1).padStart(2, '0')
    + String(date.getDate()).padStart(2, '0');
  nameInput.value = leaf ? (leaf + '-' + yymmdd) : '';
  document.getElementById('sas-objective').value = '';
  document.getElementById('sas-description').value = '';
  document.getElementById('sas-error').style.display = 'none';
  document.getElementById('save-as-study-modal').style.display = 'block';

  document.getElementById('sas-cancel').onclick = function() {
    document.getElementById('save-as-study-modal').style.display = 'none';
  };
  document.getElementById('sas-create').onclick = _ceSubmitSaveAsStudy;
}

function _ceSubmitSaveAsStudy() {
  var body = {
    name: document.getElementById('sas-name').value.trim(),
    objective: document.getElementById('sas-objective').value,
    description: document.getElementById('sas-description').value,
    source_run_id: window._ceLastRunId,
  };
  fetch('/api/study-create-from-run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).then(function(r) {
    return r.json().then(function(d) { return {status: r.status, body: d}; });
  }).then(function(res) {
    if (res.status === 200) {
      window.location = res.body.url;
    } else {
      var err = document.getElementById('sas-error');
      err.textContent = res.body.error || 'Unknown error';
      err.style.display = 'block';
    }
  });
}
```

- [ ] **Step 4: Smoke-test in the browser.** Bounce the dashboard and walk the flow manually:

```bash
cd ~/code/viva-munk
lsof -ti tcp:8765 | xargs -r kill; sleep 1
.venv/bin/vivarium-dashboard serve --workspace . --port 8765 &
sleep 3
```

Open http://localhost:8765, switch to Composite Explorer, pick `viva_munk.composites.chemotaxis`, click Test Run, wait for completion → click **Save as Study** → fill in `chemotaxis-test`, click **Create Study** → page navigates to `/studies/chemotaxis-test`.

Verify:

```bash
ls ~/code/viva-munk/studies/chemotaxis-test/
cat ~/code/viva-munk/studies/chemotaxis-test/study.yaml
sqlite3 ~/code/viva-munk/studies/chemotaxis-test/runs.db "SELECT run_id, spec_id, n_steps FROM runs_meta;"
```

Expected: directory exists, study.yaml has schema_version 3 + baseline pointing at chemotaxis, runs.db has one row.

- [ ] **Step 5: Commit.**

```bash
cd ~/code/vivarium-dashboard
git add vivarium_dashboard/static/walkthrough.js vivarium_dashboard/templates/
git commit -m "add Save-as-Study button + modal to Composite Explorer"
```

### Task 3.3: Add a backend test exercising the full create-from-run path against a real workspace

**Files:**
- Modify: `vivarium-dashboard/tests/test_study_create_from_run.py` (extend with one e2e test)

- [ ] **Step 1: Add an e2e test that uses a real fixture workspace + the running handler.** Append:

```python
def test_create_from_run_end_to_end(_fixtures_workspace, _http_handler_factory):
    """Run an actual test-run, then promote it."""
    # 1. Use the fixtures workspace; trigger a composite-test-run via /api.
    handler_post = _http_handler_factory(_fixtures_workspace)
    response = handler_post("/api/composite-test-run", {
        "id": "<pick a generator from the fixture>",
        "steps": 2,
        "overrides": {},
    })
    assert response["status"] == 200, response
    run_id = response["body"]["simulation_id"]

    # 2. Promote.
    response = handler_post("/api/study-create-from-run", {
        "name": "e2e",
        "objective": "test",
        "description": "",
        "source_run_id": run_id,
    })
    assert response["status"] == 200, response
    assert (Path(_fixtures_workspace) / "studies" / "e2e" / "study.yaml").is_file()
```

(`_http_handler_factory` is a conftest fixture you may need to add — it should return a callable that exercises a route end-to-end. If too involved, skip this step; the for_test functions already cover the logic. This step is a "nice to have" e2e seam, not mandatory.)

- [ ] **Step 2: Run.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_create_from_run.py -v`
Expected: all green.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_study_create_from_run.py
git commit -m "e2e: real test-run promotes to Study via /api/study-create-from-run"
```

**Checkpoint 3:** Group 3 done. Test Run → Save as Study works end-to-end through the UI. The Study has its own DB and YAML. The Detail view (Group 4) is the next thing the user lands on.

---

## Group 4 — Study Detail view (UI re-skin of Investigation Detail)

The work here is mostly frontend re-skinning + small backend wiring for the rename and Objective fields. The six cards each map to existing investigation-* endpoints (aliased to study-* in Group 2).

### Task 4.1: Add a `/studies/<name>` route serving the Study Detail page

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/server.py` (GET handler for `/studies/<name>`)
- Create: `vivarium-dashboard/vivarium_dashboard/templates/study-detail.html`

- [ ] **Step 1: Write a quick smoke test (template renders + responds 200).** Append to `tests/test_study_handlers.py`:

```python
def test_study_detail_page_renders(_study_workspace, _http_handler_factory):
    handler = _http_handler_factory(_study_workspace)
    response = handler("GET", "/studies/s1")
    assert response["status"] == 200
    body = response["body"]
    # Should contain the study name and "Baseline" card label
    assert "s1" in body
    assert "Baseline" in body or "baseline" in body.lower()


def test_study_detail_404_for_missing(_study_workspace, _http_handler_factory):
    handler = _http_handler_factory(_study_workspace)
    response = handler("GET", "/studies/no-such-study")
    assert response["status"] == 404
```

- [ ] **Step 2: Run, verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_handlers.py -v -k detail_page`
Expected: failure — no route handler.

- [ ] **Step 3: Add the page handler.** In `server.py`:

```python
def _get_study_detail_page(self):
    """GET /studies/<name> — render the Study Detail page."""
    # path is like /studies/<name>
    parts = self.path.strip("/").split("/")
    if len(parts) < 2 or parts[0] != "studies":
        return self._json({"error": "not found"}, 404)
    name = parts[1]
    sd = WORKSPACE / "studies" / name / "study.yaml"
    if not sd.is_file():
        return self._send_html(_render_404(name), code=404)
    spec = yaml.safe_load(sd.read_text()) or {}
    body = _render_study_detail_html(name, spec)
    return self._send_html(body, code=200)


def _send_html(self, body, code=200):
    self.send_response(code)
    self.send_header("Content-Type", "text/html; charset=utf-8")
    self.send_header("Content-Length", str(len(body.encode())))
    self.end_headers()
    self.wfile.write(body.encode())


def _render_study_detail_html(name, spec):
    """Render study-detail.html via Jinja2."""
    import jinja2
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    tpl = env.get_template("study-detail.html")
    return tpl.render(study=spec, name=name)


def _render_404(name):
    return f"<h1>Study not found</h1><p><code>{name}</code> does not exist.</p>"
```

Register the GET route. Since `/studies/<name>` is path-prefixed (not a static `/api/...`), the dispatcher needs a `startswith` branch:

```python
# In the GET dispatcher, before the dict-lookup:
if self.path.startswith("/studies/"):
    return self._get_study_detail_page()
```

- [ ] **Step 4: Create the template skeleton.** Create `vivarium_dashboard/templates/study-detail.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Study: {{ name }}</title>
<link rel="stylesheet" href="/static/style.css">
</head>
<body>
<header class="study-header">
  <h1><span class="study-name" id="study-name">{{ name }}</span>
      <span class="status-badge status-{{ study.status }}">{{ study.status }}</span></h1>
  <div class="study-meta">
    {{ study.runs|length }} run(s) ·
    {{ study.variants|length }} variant(s) ·
    {{ study.status }}
  </div>
  <div class="study-actions">
    <button onclick="_studyRename()">Rename</button>
    <button onclick="_studyExport()">Export</button>
    <button onclick="_studyDelete()" class="danger">Delete</button>
  </div>
</header>

<section class="card" id="card-objective">
  <h2>Objective</h2>
  <div id="objective-text" data-editable="true">{{ study.objective or '(blank — click to write)' }}</div>
</section>

<section class="card" id="card-baseline">
  <h2>Baseline</h2>
  <p>Composite: <code>{{ study.baseline.composite }}</code></p>
  <table id="baseline-params">
    {% for k, v in (study.baseline.params or {}).items() %}
    <tr><th>{{ k }}</th><td>{{ v }}</td></tr>
    {% endfor %}
  </table>
  <button onclick="_studyRunBaseline()">Run baseline</button>
  <button onclick="_studyEditBaselineParams()">Edit params</button>
</section>

<section class="card" id="card-variants">
  <h2>Variants ({{ study.variants|length }})</h2>
  <div id="variants-list">
    {% for v in (study.variants or []) %}
    <div class="variant" data-variant="{{ v.name }}">
      <h3>{{ v.name }}</h3>
      <p>{{ v.intervention.description if v.intervention else '' }}</p>
      <pre>{{ v.intervention.parameter_overrides if v.intervention else '' }}</pre>
      <button onclick="_studyRunVariant('{{ v.name }}')">Run variant</button>
      <button onclick="_studyEditVariant('{{ v.name }}')">Edit</button>
      <button onclick="_studyDeleteVariant('{{ v.name }}')">Delete</button>
    </div>
    {% endfor %}
  </div>
  <button onclick="_studyAddVariant()">+ Add variant</button>
</section>

<section class="card" id="card-runs">
  <h2>Runs ({{ study.runs|length }})</h2>
  <table id="runs-table">
    <thead><tr><th></th><th>Variant</th><th>Label</th><th>Steps</th><th>Status</th><th>Viz</th><th>Actions</th></tr></thead>
    <tbody>
    {% for r in (study.runs or []) %}
    <tr data-run-id="{{ r.run_id }}">
      <td><input type="checkbox" class="run-compare-checkbox" value="{{ r.run_id }}"></td>
      <td>{{ r.variant or 'baseline' }}</td>
      <td>{{ r.label or '' }}</td>
      <td>{{ r.n_steps or '' }}</td>
      <td>{{ r.status }}</td>
      <td>{% if r.viz %}🎞{% endif %}</td>
      <td>
        <button onclick="_studyViewRun('{{ r.run_id }}')">View</button>
        <button onclick="_studyDeleteRun('{{ r.run_id }}')">Delete</button>
      </td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  <button onclick="_studyCompareSelected()">Compare selected</button>
  <button onclick="_studyClearAllRuns()" class="danger">Clear all runs</button>
</section>

<section class="card" id="card-viz">
  <h2>Visualizations</h2>
  <div id="viz-list">
    {% for v in (study.visualizations or []) %}
    <div class="viz-config">{{ v.name }} · {{ v.kind }}</div>
    {% endfor %}
  </div>
  <button onclick="_studyAddViz()">+ Add visualization</button>
</section>

<section class="card" id="card-conclusion">
  <h2>Conclusion</h2>
  <div id="conclusion-text" data-editable="true">
    {{ (study.conclusion and study.conclusion.text) or '(blank)' }}
  </div>
  <button onclick="_studyMarkComplete()">Mark complete</button>
</section>

<script>
window._study = {{ study|tojson }};
window._studyName = "{{ name }}";
</script>
<script src="/static/study-detail.js"></script>
</body>
</html>
```

- [ ] **Step 5: Run page tests.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_study_handlers.py -v -k detail_page`
Expected: both pass.

- [ ] **Step 6: Commit.**

```bash
git add vivarium_dashboard/server.py vivarium_dashboard/templates/study-detail.html tests/test_study_handlers.py
git commit -m "add /studies/<name> page route with six-card detail template"
```

### Task 4.2: Implement card-level frontend behaviours in `study-detail.js`

**Files:**
- Create: `vivarium-dashboard/vivarium_dashboard/static/study-detail.js`

- [ ] **Step 1: Create `study-detail.js` with the click-to-edit, run-baseline, and run-variant flows.** Since this is straightforward DOM glue with HTTP fetches, write the file in one pass:

```javascript
// study-detail.js — wires the six-card Study Detail page to /api/study-* routes.
(function() {
  function api(method, path, body) {
    return fetch(path, {
      method: method,
      headers: body ? {'Content-Type': 'application/json'} : {},
      body: body ? JSON.stringify(body) : null,
    }).then(function(r) {
      return r.json().then(function(d) { return {status: r.status, body: d}; });
    });
  }

  // --- Inline-edit (objective + conclusion) ---
  function makeEditable(el, savePath, field) {
    el.addEventListener('click', function() {
      if (el.querySelector('textarea')) return;
      var t = document.createElement('textarea');
      t.value = el.textContent.trim() === '(blank — click to write)' || el.textContent.trim() === '(blank)' ? '' : el.textContent.trim();
      t.rows = 4;
      t.style.width = '100%';
      el.innerHTML = '';
      el.appendChild(t);
      t.focus();
      t.addEventListener('blur', function() {
        var body = {study: window._studyName};
        body[field] = t.value;
        api('POST', savePath, body).then(function(res) {
          el.textContent = t.value || '(blank)';
        });
      });
    });
  }
  makeEditable(document.getElementById('objective-text'),
               '/api/study-set-objective', 'text');
  makeEditable(document.getElementById('conclusion-text'),
               '/api/study-set-conclusion', 'conclusion');

  // --- Header actions ---
  window._studyRename = function() {
    var n = prompt('New name (lowercase + dashes):', window._studyName);
    if (!n) return;
    api('POST', '/api/study-rename', {study: window._studyName, new_name: n})
      .then(function(res) {
        if (res.status === 200) window.location = '/studies/' + n;
        else alert(res.body.error || 'Rename failed');
      });
  };

  window._studyExport = function() {
    window.location = '/api/study-export?study=' + encodeURIComponent(window._studyName);
  };

  window._studyDelete = function() {
    if (!confirm('Delete this study and all its runs?')) return;
    api('POST', '/api/study-delete', {study: window._studyName})
      .then(function() { window.location = '/studies'; });
  };

  // --- Baseline ---
  window._studyRunBaseline = function() {
    api('POST', '/api/study-run-baseline', {study: window._studyName})
      .then(function(res) {
        if (res.status === 200) location.reload();
        else alert(res.body.error || 'Run failed');
      });
  };

  window._studyEditBaselineParams = function() {
    var params = window._study.baseline.params || {};
    var text = prompt('Edit baseline params (JSON):', JSON.stringify(params, null, 2));
    if (text == null) return;
    try {
      var parsed = JSON.parse(text);
      api('POST', '/api/study-set-baseline-params',
          {study: window._studyName, params: parsed})
        .then(function() { location.reload(); });
    } catch (e) {
      alert('Invalid JSON: ' + e.message);
    }
  };

  // --- Variants ---
  window._studyAddVariant = function() {
    var name = prompt('Variant name:');
    if (!name) return;
    var desc = prompt('Description:', '') || '';
    var po = prompt('Parameter overrides (JSON, e.g. {"rate": 2.0}):', '{}') || '{}';
    try { po = JSON.parse(po); } catch (e) { return alert('Invalid JSON'); }
    api('POST', '/api/study-variant-add', {
      investigation: window._studyName, name: name,
      extends: 'baseline', description: desc, parameter_overrides: po,
    }).then(function(res) {
      if (res.status === 200) location.reload();
      else alert(res.body.error || 'Add variant failed');
    });
  };

  window._studyRunVariant = function(variantName) {
    api('POST', '/api/study-run-variant',
        {study: window._studyName, variant: variantName})
      .then(function() { location.reload(); });
  };

  window._studyDeleteVariant = function(variantName) {
    if (!confirm('Delete variant ' + variantName + '?')) return;
    api('POST', '/api/study-variant-delete',
        {study: window._studyName, variant: variantName})
      .then(function() { location.reload(); });
  };

  window._studyEditVariant = function(variantName) {
    alert('Edit not implemented yet — delete + re-add for now.');
  };

  // --- Runs ---
  window._studyViewRun = function(runId) {
    // Open existing trajectory inspector in a new tab (reuse CE machinery)
    window.open('/composite-explorer?run_id=' + encodeURIComponent(runId), '_blank');
  };

  window._studyDeleteRun = function(runId) {
    if (!confirm('Delete this run?')) return;
    api('POST', '/api/study-run-delete',
        {study: window._studyName, run_id: runId})
      .then(function() { location.reload(); });
  };

  window._studyClearAllRuns = function() {
    if (!confirm('Clear ALL runs in this study?')) return;
    api('POST', '/api/study-runs-clear', {study: window._studyName})
      .then(function() { location.reload(); });
  };

  window._studyCompareSelected = function() {
    var ids = [];
    document.querySelectorAll('.run-compare-checkbox:checked').forEach(function(c) {
      ids.push(c.value);
    });
    if (ids.length < 2) return alert('Select at least two runs.');
    api('POST', '/api/study-comparison-add',
        {study: window._studyName, run_ids: ids})
      .then(function(res) {
        if (res.status === 200) location.reload();
        else alert(res.body.error || 'Compare failed');
      });
  };

  // --- Viz ---
  window._studyAddViz = function() {
    alert('Add visualization: not implemented in Phase 1.');
  };

  // --- Conclusion ---
  window._studyMarkComplete = function() {
    api('POST', '/api/study-set-conclusion',
        {study: window._studyName, mark_complete: true})
      .then(function() { location.reload(); });
  };
})();
```

- [ ] **Step 2: Smoke-test in the browser.** Bounce the dashboard, open the Study you saved in Group 3, click around: edit objective, add a variant, run it, mark complete.

```bash
cd ~/code/viva-munk
lsof -ti tcp:8765 | xargs -r kill; sleep 1
.venv/bin/vivarium-dashboard serve --workspace . --port 8765 &
sleep 3
open http://localhost:8765/studies/chemotaxis-test
```

Walk through the cards. Click **Objective** → type text → blur → reload → text persists. Click **+ Add variant** → fill in → run variant → row appears in Runs card.

- [ ] **Step 3: Commit.**

```bash
git add vivarium_dashboard/static/study-detail.js
git commit -m "wire Study Detail card actions to /api/study-* endpoints"
```

### Task 4.3: Make the Investigations tab list link to `/studies/<name>` (instead of `/investigations/<name>`)

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/templates/index.html` (or wherever the Investigations tab list is rendered)
- Modify: `vivarium-dashboard/vivarium_dashboard/static/walkthrough.js` (any `_renderInvestigationsList` that builds href strings)

- [ ] **Step 1: Find the existing list renderer.**

Run: `cd ~/code/vivarium-dashboard && grep -nE "/investigations/|_renderInvestigations|api/investigations" vivarium_dashboard/static/walkthrough.js vivarium_dashboard/templates/*.html | head`

- [ ] **Step 2: Replace href bases.** Change every `'/investigations/' + inv.name` to `'/studies/' + inv.name`, and every label "Investigations" surfaced to the user to "Studies". (The tab name itself stays "Investigations" for now — that's the Phase 3 graph view; Phase 1 just lists Studies under it.)

- [ ] **Step 3: Smoke-test.** Open http://localhost:8765, switch to the Investigations tab, click on the study you saved → page navigates to `/studies/<name>` (your new Detail view).

- [ ] **Step 4: Commit.**

```bash
git add vivarium_dashboard/static/walkthrough.js vivarium_dashboard/templates/
git commit -m "link Investigations-tab list items to /studies/<name>"
```

### Task 4.4: Add `studies/` to the active-branch `.pbg/local/` gitignore convention

**Files:**
- Modify: `viva-munk/.gitignore` (if you want studies/ kept locally per-user) OR leave studies/ tracked

The spec is silent on whether `studies/` is committed or gitignored. Two reasonable defaults:

- **Commit** Studies (research history is part of the repo) — same treatment as `investigations/` today.
- **Gitignore** Studies under `.pbg/local/studies/` (researcher-local; doesn't pollute shared history).

- [ ] **Step 1: Decide based on workspace convention.** For viva-munk specifically, runs are large (SQLite history blobs) and arguably ephemeral. **Recommendation:** keep `study.yaml` + `notes.md` + `viz/` committed; gitignore `runs.db` since it can be large and binary. Edit `.gitignore`:

```
studies/*/runs.db
```

- [ ] **Step 2: Commit.**

```bash
cd ~/code/viva-munk
git add .gitignore
git commit -m "gitignore studies/*/runs.db (large binary; viz + spec stay committed)"
```

**Checkpoint 4:** Group 4 done. The Study Detail view is fully usable. Save-as-Study → land on Detail → edit objective → add variants → run them → see results → mark complete. End-to-end workflow works.

---

## Group 5 — Migration: `vivarium-dashboard migrate-investigations` CLI subcommand

### Task 5.1: Build the migration script as a CLI subcommand

**Files:**
- Modify: `vivarium-dashboard/vivarium_dashboard/cli.py` (add `migrate-investigations` subcommand)
- Test: `vivarium-dashboard/tests/test_migrate_investigations_to_studies.py` (new)

- [ ] **Step 1: Write failing tests.** Create `tests/test_migrate_investigations_to_studies.py`:

```python
"""Tests for the investigations → studies migration CLI."""
import yaml
from pathlib import Path

import pytest


@pytest.fixture
def _ws_with_v2_investigation(tmp_path):
    """Workspace with one v2 investigation directory."""
    ws = tmp_path / "ws"
    inv = ws / "investigations" / "old"
    inv.mkdir(parents=True)
    (inv / "spec.yaml").write_text(yaml.safe_dump({
        "schema_version": 2,
        "name": "old",
        "created": "2026-04-01",
        "composites": [
            {"name": "main", "source": "pkg.composites.foo", "parameters": {"x": 1}},
        ],
        "runs": [],
        "variants": [],
    }))
    (inv / "notes.md").write_text("hello")
    (inv / "composites").mkdir()
    (inv / "viz").mkdir()
    return ws


def test_migration_creates_studies_dir(_ws_with_v2_investigation):
    from vivarium_dashboard.cli import migrate_investigations_to_studies
    result = migrate_investigations_to_studies(_ws_with_v2_investigation, dry_run=False)
    sd = _ws_with_v2_investigation / "studies" / "old"
    assert sd.is_dir()
    assert (sd / "study.yaml").is_file()
    assert (sd / "notes.md").read_text() == "hello"
    assert not (_ws_with_v2_investigation / "investigations" / "old").exists()
    assert result["migrated"] == 1


def test_migration_dry_run_makes_no_changes(_ws_with_v2_investigation):
    from vivarium_dashboard.cli import migrate_investigations_to_studies
    result = migrate_investigations_to_studies(_ws_with_v2_investigation, dry_run=True)
    assert (_ws_with_v2_investigation / "investigations" / "old").is_dir()
    assert not (_ws_with_v2_investigation / "studies").exists()
    assert result["would_migrate"] == 1


def test_migration_rewrites_spec_to_v3(_ws_with_v2_investigation):
    from vivarium_dashboard.cli import migrate_investigations_to_studies
    migrate_investigations_to_studies(_ws_with_v2_investigation, dry_run=False)
    spec = yaml.safe_load(
        (_ws_with_v2_investigation / "studies" / "old" / "study.yaml").read_text()
    )
    assert spec["schema_version"] == 3
    assert spec["baseline"] == {"composite": "pkg.composites.foo", "params": {"x": 1}}
    assert "composites" not in spec


def test_migration_idempotent(_ws_with_v2_investigation):
    from vivarium_dashboard.cli import migrate_investigations_to_studies
    migrate_investigations_to_studies(_ws_with_v2_investigation, dry_run=False)
    # Running again is a no-op (investigations/ is gone)
    result = migrate_investigations_to_studies(_ws_with_v2_investigation, dry_run=False)
    assert result["migrated"] == 0
```

- [ ] **Step 2: Run, verify failure.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_migrate_investigations_to_studies.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the migration.** Edit `vivarium_dashboard/cli.py`. Add the subcommand registration in the main CLI parser:

```python
def main():
    parser = argparse.ArgumentParser(prog="vivarium-dashboard")
    sub = parser.add_subparsers(dest="cmd")

    # existing: serve
    p_serve = sub.add_parser("serve")
    p_serve.add_argument("--workspace", default=".")
    p_serve.add_argument("--port", type=int, default=None)

    # NEW: migrate-investigations
    p_mig = sub.add_parser("migrate-investigations",
                            help="One-shot migration: investigations/ → studies/")
    p_mig.add_argument("--workspace", default=".")
    p_mig.add_argument("--dry-run", action="store_true",
                       help="Report what would change without writing")

    args = parser.parse_args()
    if args.cmd == "serve":
        return _serve(args)
    if args.cmd == "migrate-investigations":
        ws = Path(args.workspace).resolve()
        result = migrate_investigations_to_studies(ws, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))
        return 0
    parser.print_help()
    return 1
```

Then add the function:

```python
def migrate_investigations_to_studies(ws_root, dry_run=False):
    """One-shot: walk investigations/, rename → studies/, migrate spec v2→v3.

    Returns {migrated|would_migrate: N, errors: [{name, error}], warnings: [...]}.
    """
    from vivarium_dashboard.lib.spec_migration import migrate_v2_to_v3
    import warnings

    inv_root = ws_root / "investigations"
    studies_root = ws_root / "studies"

    if not inv_root.is_dir():
        return {"migrated": 0, "errors": [], "warnings": ["no investigations/ to migrate"]}

    count_key = "would_migrate" if dry_run else "migrated"
    result = {count_key: 0, "errors": [], "warnings": []}

    for inv in sorted(inv_root.iterdir()):
        if not inv.is_dir():
            continue
        spec_path = inv / "spec.yaml"
        if not spec_path.is_file():
            continue
        try:
            spec = yaml.safe_load(spec_path.read_text()) or {}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                v3 = migrate_v2_to_v3(spec)
            for w in caught:
                result["warnings"].append(f"{inv.name}: {w.message}")

            if dry_run:
                result["would_migrate"] += 1
                continue

            studies_root.mkdir(parents=True, exist_ok=True)
            dst = studies_root / inv.name
            if dst.exists():
                result["errors"].append({"name": inv.name,
                                         "error": "destination already exists"})
                continue

            inv.rename(dst)
            # Rename spec.yaml → study.yaml + rewrite v3
            (dst / "spec.yaml").rename(dst / "study.yaml")
            (dst / "study.yaml").write_text(yaml.safe_dump(v3, sort_keys=False))
            result["migrated"] += 1
        except Exception as e:
            result["errors"].append({"name": inv.name, "error": str(e)})

    # If investigations/ is now empty, remove it.
    if not dry_run and inv_root.is_dir() and not any(inv_root.iterdir()):
        inv_root.rmdir()

    return result
```

- [ ] **Step 4: Run tests to verify pass.**

Run: `cd ~/code/vivarium-dashboard && pytest tests/test_migrate_investigations_to_studies.py -v`
Expected: all 4 pass.

- [ ] **Step 5: Commit.**

```bash
git add vivarium_dashboard/cli.py tests/test_migrate_investigations_to_studies.py
git commit -m "add 'vivarium-dashboard migrate-investigations' CLI for v2→v3 migration"
```

### Task 5.2: Document the migration in the project README (or release notes)

**Files:**
- Modify: `vivarium-dashboard/README.md` (or `CHANGELOG.md` if one exists)

- [ ] **Step 1: Add a section to the README.** Append the following to the "Usage" section:

````markdown
### Migration: Investigations → Studies (one-time)

If your workspace has `investigations/<name>/spec.yaml` directories created
before schema_version 3 / Studies, run the migration once:

```bash
vivarium-dashboard migrate-investigations --workspace /path/to/workspace
```

The script renames `investigations/` → `studies/`, bumps each spec from
v2 to v3, and lifts the first composite into `baseline:`. Multi-composite
investigations are migrated with a warning — recreate the extra composites
as variants from the new Study Detail view.

Add `--dry-run` to preview without writing.
````

- [ ] **Step 2: Commit.**

```bash
git add README.md
git commit -m "doc: migrate-investigations CLI usage"
```

**Checkpoint 5:** Group 5 done. All five groups complete; the Phase 1 deliverable is shippable.

---

## Verification (run after all five groups)

- [ ] **Step 1: Full test sweep across all three repos.**

```bash
cd ~/code/pbg-superpowers && pytest -v
cd ~/code/vivarium-dashboard && pytest -v
cd ~/code/viva-munk && .venv/bin/python -c "
import viva_munk  # fires register_processes + composite_generators
from pbg_superpowers.composite_generator import _REGISTRY
assert len(_REGISTRY) == 9, f'expected 9, got {len(_REGISTRY)}'
for entry in _REGISTRY.values():
    assert entry.default_n_steps is not None and entry.default_n_steps > 0, entry
print('OK — 9 generators with default_n_steps')
"
```

Expected: all green; viva-munk smoke check prints "OK".

- [ ] **Step 2: Full end-to-end manual walkthrough.**

```bash
cd ~/code/viva-munk
lsof -ti tcp:8765 | xargs -r kill; sleep 1
.venv/bin/vivarium-dashboard serve --workspace . --port 8765 &
sleep 3
```

Open http://localhost:8765 and verify:

1. Composite Explorer shows `default_n_steps` pre-filled per composite.
2. Run a Test Run for chemotaxis — viz renders.
3. Click **Save as Study** — modal opens with pre-filled name `chemotaxis-<date>`.
4. Submit → land on `/studies/chemotaxis-<date>`.
5. Study Detail shows: name + status badge, blank Objective, Baseline card with composite + params, no variants, one baseline run in Runs card, blank Conclusion.
6. Click Objective → type → blur — text persists across reload.
7. Click + Add variant → name `high-sens`, params `{"sensitivity": 6.0}` → variant appears.
8. Click **Run variant** → new row appears in Runs.
9. Select baseline + variant rows → **Compare selected** → comparison view opens.
10. Edit Conclusion → click **Mark complete** → status badge flips to `complete`.

- [ ] **Step 3: Final commit (tag the release).**

```bash
cd ~/code/vivarium-dashboard
git tag v0.2.0 -m "Studies Phase 1: data model, default_n_steps, save-as-study, detail view"
git push --tags  # optional; only if remote is configured
```

---

## Self-Review notes

1. **Spec coverage check** — every section of the spec is covered:
   - §2 goals 1–5 → all five groups.
   - §5.1 (data model) → Task 2.1 (schema migration) + Task 3.1 (study creation).
   - §5.2 (default_n_steps) → Group 1 (all three tasks).
   - §5.3 (save-as-study) → Group 3.
   - §5.4 (detail view) → Group 4 (all four tasks).
   - §6 (migration script) → Group 5.

2. **Placeholder scan** — searched for "TBD", "TODO", "Add appropriate". The only "not implemented in Phase 1" is the explicit *Add visualization* placeholder, which matches the spec's deferral in §9.

3. **Type consistency** — `default_n_steps: int | None`, `study.yaml.schema_version: 3`, slug regex `^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$`, alias map keyed by `/api/investigation-*` → `/api/study-*` — all consistent across tasks.

4. **Open implementation questions still deferred to the plan** (per spec §9):
   - URL aliasing chosen over full rename: Task 2.2 implements aliases.
   - Param-edit modal: Task 4.2 uses a plain JSON prompt for Phase 1; type-aware widgets are deferred.
   - Process-override recipe UX: Task 4.2 uses a plain JSON prompt; richer wizard deferred.
   - Default viz config seeding: not done; new Studies start with `visualizations: []` and rely on the composite's own `MultibodyVizStep` for runtime viz.
   - Notes card: deferred — `notes.md` exists on disk per the data model but no UI surface in Phase 1.
