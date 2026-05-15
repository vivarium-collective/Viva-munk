# pbg-biomodels absorbs pbg-biomodels-bundle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make `pbg-biomodels` self-contained by absorbing all of `pbg-biomodels-bundle`'s code (LoadBiomodelStep, SimulatorComparisonStep, comparison math, run_biomodels, analysis) AND replacing the duplicated simulator Steps with thin wrappers that delegate to `pbg-tellurium` and `pbg-copasi`. Retire `pbg-biomodels-bundle`.

**Architecture:** Bundle's 5 non-simulator Python files move verbatim into pbg-biomodels with internal import paths rewritten. The 2 simulator Step classes (`LocalCopasiUTCStep`, `LocalTelluriumUTCStep`) are deleted; a new `pbg_biomodels/steps/simulators.py` provides `BiomodelsCopasiStep` and `BiomodelsTelluriumStep` — thin adapter Steps that take runtime inputs and delegate to `pbg_copasi.CopasiUTCStep` / `pbg_tellurium.TelluriumUTCStep` respectively, reshaping tellurium's output to the canonical `{result: numeric_result}` shape.

**Tech Stack:** `process-bigraph`, `bigraph-schema`, `pbg-tellurium`, `pbg-copasi`, `biomodels` (BioModels API client), `matplotlib`/`bigraph-viz`/`rest-process`/`jinja2`.

**Spec:** [`docs/superpowers/specs/2026-05-15-pbg-biomodels-absorb-bundle-design.md`](../specs/2026-05-15-pbg-biomodels-absorb-bundle-design.md)

**Working repos:**
- `~/code/pbg-biomodels` — absorbs functionality; multiple commits
- `~/code/pbg-biomodels-bundle` — retired; one final README commit

**Working venv:** `~/code/pbg-biomodels` has its own `.venv` (or create one if absent). All test runs use `.venv/bin/python`. Both `pbg-tellurium` and `pbg-copasi` must be editable-installed into this venv before tests can pass.

---

## Task B1: Migrate bundle modules into pbg-biomodels (verbatim + import rewrite)

**Files:**
- Create: `~/code/pbg-biomodels/pbg_biomodels/steps/__init__.py`
- Create: `~/code/pbg-biomodels/pbg_biomodels/steps/load_biomodel.py`
- Create: `~/code/pbg-biomodels/pbg_biomodels/steps/simulator_comparison.py`
- Create: `~/code/pbg-biomodels/pbg_biomodels/comparison.py`
- Create: `~/code/pbg-biomodels/pbg_biomodels/run_biomodels.py`
- Create: `~/code/pbg-biomodels/pbg_biomodels/analysis.py`
- Modify: `~/code/pbg-biomodels/pbg_biomodels/__init__.py` (merge bundle's exports)

- [ ] **Step 1: Verify the target subdirectory exists; create if not**

```bash
mkdir -p ~/code/pbg-biomodels/pbg_biomodels/steps
```

- [ ] **Step 2: Copy 5 files verbatim (no edits yet)**

```bash
cp ~/code/pbg-biomodels-bundle/pbg_biomodels_bundle/steps/__init__.py \
   ~/code/pbg-biomodels/pbg_biomodels/steps/__init__.py
cp ~/code/pbg-biomodels-bundle/pbg_biomodels_bundle/steps/load_biomodel.py \
   ~/code/pbg-biomodels/pbg_biomodels/steps/load_biomodel.py
cp ~/code/pbg-biomodels-bundle/pbg_biomodels_bundle/steps/simulator_comparison.py \
   ~/code/pbg-biomodels/pbg_biomodels/steps/simulator_comparison.py
cp ~/code/pbg-biomodels-bundle/pbg_biomodels_bundle/comparison.py \
   ~/code/pbg-biomodels/pbg_biomodels/comparison.py
cp ~/code/pbg-biomodels-bundle/pbg_biomodels_bundle/run_biomodels.py \
   ~/code/pbg-biomodels/pbg_biomodels/run_biomodels.py
cp ~/code/pbg-biomodels-bundle/pbg_biomodels_bundle/analysis.py \
   ~/code/pbg-biomodels/pbg_biomodels/analysis.py
```

Notice we do NOT copy `local_simulators.py` — that file is deleted and replaced by `simulators.py` in Task B2.

- [ ] **Step 3: Rewrite internal `pbg_biomodels_bundle` references → `pbg_biomodels`**

Use sed (BSD/macOS) to apply across all newly-copied files. Word-boundary not strictly necessary because the bundle name is unambiguous as a substring:

```bash
cd ~/code/pbg-biomodels/pbg_biomodels
for f in steps/__init__.py steps/load_biomodel.py steps/simulator_comparison.py comparison.py run_biomodels.py analysis.py; do
    sed -i '' 's/pbg_biomodels_bundle/pbg_biomodels/g' "$f"
done
```

- [ ] **Step 4: Spot-check the rewrites**

```bash
cd ~/code/pbg-biomodels/pbg_biomodels
grep -n "pbg_biomodels_bundle" steps/*.py *.py 2>/dev/null
```
Expected: NO output (all references successfully rewritten).

Also verify the actual imports look sensible:
```bash
grep -nE "^from pbg_biomodels|^import pbg_biomodels" pbg_biomodels/steps/*.py pbg_biomodels/*.py 2>/dev/null
```

- [ ] **Step 5: Merge bundle's `__init__.py` public exports into pbg-biomodels' `__init__.py`**

Read both files:
```bash
cat ~/code/pbg-biomodels-bundle/pbg_biomodels_bundle/__init__.py
cat ~/code/pbg-biomodels/pbg_biomodels/__init__.py
```

Take any `from pbg_biomodels_bundle.X import Y` lines from the bundle's `__init__.py`, rewrite to `from pbg_biomodels.X import Y`, and append to pbg-biomodels' `__init__.py` (skipping any imports already present). Add the names to `__all__` if pbg-biomodels has one. Do NOT delete pbg-biomodels' own existing imports/exports.

- [ ] **Step 6: Verify imports work**

```bash
cd ~/code/pbg-biomodels && .venv/bin/python -c "
import pbg_biomodels
from pbg_biomodels.steps.load_biomodel import LoadBiomodelStep
from pbg_biomodels.steps.simulator_comparison import SimulatorComparisonStep
from pbg_biomodels.comparison import compare_two_engines
print('migration imports ok')
"
```
Expected: `migration imports ok`.

If pbg-biomodels' venv doesn't have all the bundle's transitive deps yet (`tellurium`, `copasi-basico`, `biomodels`, `rest-process`, `matplotlib`, `libsedml`, `libsbml`), the imports will fail at this step. Task B4 fixes pyproject; for now, install whatever's missing manually so this step passes:

```bash
cd ~/code/pbg-biomodels && uv pip install -q -e ~/code/pbg-tellurium -e ~/code/pbg-copasi tellurium copasi-basico biomodels rest-process matplotlib python-libsedml python-libsbml
```

- [ ] **Step 7: Commit**

```bash
cd ~/code/pbg-biomodels
git add pbg_biomodels/steps/ pbg_biomodels/comparison.py pbg_biomodels/run_biomodels.py pbg_biomodels/analysis.py pbg_biomodels/__init__.py
git commit -m "absorb pbg-biomodels-bundle modules into pbg_biomodels (verbatim + import rewrite)"
```

---

## Task B2: Create `simulators.py` wrappers + unit tests

**Files:**
- Create: `~/code/pbg-biomodels/pbg_biomodels/steps/simulators.py`
- Create: `~/code/pbg-biomodels/tests/test_simulators.py`

- [ ] **Step 1: Inspect the canonical Step config schemas**

```bash
sed -n '/^class CopasiUTCStep/,/^class /p' ~/code/pbg-copasi/pbg_copasi/processes.py | head -80
sed -n '/^class TelluriumUTCStep/,/^class /p' ~/code/pbg-tellurium/pbg_tellurium/processes.py | head -80
```

Identify the EXACT config keys each canonical class accepts (e.g., does `CopasiUTCStep` use `model_source`, `duration`, `n_points`? Does `TelluriumUTCStep` use `model`, `model_format`, `start_time`, `end_time`, `n_points`?). The wrappers must use these exact keys.

- [ ] **Step 2: Write `pbg_biomodels/steps/simulators.py`**

```python
"""Thin runtime-input wrappers around pbg-copasi / pbg-tellurium UTC Steps.

Adapter Steps for the compare-biomodel composite: take ``model_source``,
``time``, ``n_points`` as runtime inputs (so LoadBiomodelStep can feed
them dynamically) and emit the canonical ``numeric_result`` shape on the
``result`` output port. Delegates the actual simulation to the canonical
classes in pbg-tellurium and pbg-copasi.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict

from process_bigraph import Step

from pbg_copasi.processes import CopasiUTCStep
from pbg_tellurium.processes import TelluriumUTCStep


_UTC_INPUTS: Dict[str, str] = {
    "model_source": "string",
    "time":         "float",
    "n_points":     "integer",
}


def _validate_n_points(n: Any, where: str) -> int:
    n = int(n)
    if n < 2:
        raise ValueError(f"{where}: n_points must be >= 2, got {n}")
    return n


class BiomodelsCopasiStep(Step):
    """Adapter: runtime ``model_source`` → ``pbg_copasi.CopasiUTCStep``.

    pbg-copasi's CopasiUTCStep takes model via config; this wrapper accepts
    model_source as a runtime input so the compare-biomodel composite can
    feed it from LoadBiomodelStep.
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_UTC_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "numeric_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        n_points = _validate_n_points(state["n_points"], "BiomodelsCopasiStep")
        inner = CopasiUTCStep(config={
            "model_source": state["model_source"],
            # NOTE: confirm the exact keys CopasiUTCStep accepts (duration?
            # time? end_time?) — adapt this dict to match its config_schema.
            "duration": float(state["time"]),
            "n_points": n_points,
        })
        out = inner.update({})  # type: ignore[arg-type]
        # CopasiUTCStep already emits {result: numeric_result} — pass through.
        return {"result": out["result"]}


class BiomodelsTelluriumStep(Step):
    """Adapter: runtime ``model_source`` → ``pbg_tellurium.TelluriumUTCStep``.

    pbg-tellurium's TelluriumUTCStep emits ``time_series`` + ``species_trajectories``;
    we reshape that to the canonical ``numeric_result`` shape used everywhere
    else in pbg-biomodels.
    """

    config_schema: ClassVar[Dict[str, Any]] = {}

    def inputs(self) -> Dict[str, str]:
        return dict(_UTC_INPUTS)

    def outputs(self) -> Dict[str, str]:
        return {"result": "numeric_result"}

    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        n_points = _validate_n_points(state["n_points"], "BiomodelsTelluriumStep")
        inner = TelluriumUTCStep(config={
            "model":         state["model_source"],
            "model_format":  "sbml",
            "start_time":    0.0,
            "end_time":      float(state["time"]),
            "n_points":      n_points,
        })
        out = inner.update({})  # type: ignore[arg-type]
        # Reshape from {time_series, species_trajectories} → {result: numeric_result}
        time_list = list(out["time_series"])
        trajectories = out["species_trajectories"]
        columns = list(trajectories.keys())
        n_rows = len(time_list)
        values = [
            [float(trajectories[c][r]) for c in columns]
            for r in range(n_rows)
        ]
        return {
            "result": {
                "time":    time_list,
                "columns": columns,
                "values":  values,
            }
        }
```

**Important:** Step 1 told you the EXACT config keys each canonical class accepts. If `CopasiUTCStep` uses a different key than `duration` (e.g., `time` or `end_time`), correct the wrapper accordingly. Likewise for tellurium. The template above is a SKETCH — you must adapt it to match the canonical config_schemas precisely.

- [ ] **Step 3: Write `tests/test_simulators.py`**

Two tests, one per wrapper. Use the BIOMD0000000012 SBML fixture already vendored in pbg-copasi (path: `~/code/pbg-copasi/pbg_copasi/composites/repressilator.xml`) or pbg-copasi's `tests/fixtures/BIOMD0000000012_url.xml`.

```python
"""Tests for pbg_biomodels.steps.simulators wrappers."""
from importlib.resources import files
from pathlib import Path

from pbg_biomodels.steps.simulators import (
    BiomodelsCopasiStep,
    BiomodelsTelluriumStep,
)


def _model_path() -> str:
    # Use pbg-copasi's bundled Repressilator model (always present once
    # pbg-copasi is installed in this venv).
    return str(files('pbg_copasi.composites') / 'repressilator.xml')


def test_biomodels_copasi_step_returns_numeric_result():
    step = BiomodelsCopasiStep()
    out = step.update({
        'model_source': _model_path(),
        'time': 10.0,
        'n_points': 11,
    })
    assert 'result' in out
    r = out['result']
    assert set(r.keys()) >= {'time', 'columns', 'values'}
    assert len(r['time']) == 11
    assert len(r['columns']) > 0
    assert len(r['values']) == 11


def test_biomodels_tellurium_step_returns_numeric_result():
    step = BiomodelsTelluriumStep()
    out = step.update({
        'model_source': _model_path(),
        'time': 10.0,
        'n_points': 11,
    })
    assert 'result' in out
    r = out['result']
    assert set(r.keys()) >= {'time', 'columns', 'values'}
    assert len(r['time']) == 11
    assert len(r['columns']) > 0
    assert len(r['values']) == 11
    # Tellurium reshape should produce same-shape rows
    assert all(len(row) == len(r['columns']) for row in r['values'])
```

- [ ] **Step 4: Run the new tests**

```bash
cd ~/code/pbg-biomodels && .venv/bin/python -m pytest tests/test_simulators.py -q
```
Expected: 2 passed.

If a test fails because the wrapper passes the wrong config key to the inner class, fix the wrapper based on Step 1's inspection.

- [ ] **Step 5: Commit**

```bash
cd ~/code/pbg-biomodels
git add pbg_biomodels/steps/simulators.py tests/test_simulators.py
git commit -m "add BiomodelsCopasiStep + BiomodelsTelluriumStep wrappers + unit tests"
```

---

## Task B3: Update compare-biomodel composite step addresses

**Files:**
- Modify: `~/code/pbg-biomodels/pbg_biomodels/composites/compare_biomodel.py`

- [ ] **Step 1: Edit the 4 address constants**

Open `pbg_biomodels/composites/compare_biomodel.py` and replace the 4 constants near the top of the file:

```python
LOAD_STEP_ADDRESS = "local:pbg_biomodels.steps.load_biomodel.LoadBiomodelStep"
COPASI_STEP_ADDRESS = "local:pbg_biomodels.steps.simulators.BiomodelsCopasiStep"
TELLURIUM_STEP_ADDRESS = "local:pbg_biomodels.steps.simulators.BiomodelsTelluriumStep"
COMPARISON_STEP_ADDRESS = "local:pbg_biomodels.steps.simulator_comparison.SimulatorComparisonStep"
```

(Leave `VISUALIZATION_STEP_ADDRESS` untouched — it already references `pbg_biomodels.visualizations.compare_overlay.CompareOverlay`.)

- [ ] **Step 2: Verify the composite still imports**

```bash
cd ~/code/pbg-biomodels && .venv/bin/python -c "
from pbg_biomodels.composites.compare_biomodel import build_compare_biomodel
print('compare-biomodel imports ok')
"
```
Expected: `compare-biomodel imports ok`.

- [ ] **Step 3: Confirm no leftover bundle references in compare_biomodel.py**

```bash
grep -n "pbg_biomodels_bundle" ~/code/pbg-biomodels/pbg_biomodels/composites/compare_biomodel.py
```
Expected: NO output.

- [ ] **Step 4: Commit**

```bash
cd ~/code/pbg-biomodels
git add pbg_biomodels/composites/compare_biomodel.py
git commit -m "compare-biomodel: rewire step addresses to pbg_biomodels (and new wrappers)"
```

---

## Task B4: Update pyproject.toml + workspace.yaml + catalog modules.json

**Files:**
- Modify: `~/code/pbg-biomodels/pyproject.toml`
- Modify: `~/code/pbg-biomodels/workspace.yaml`
- Modify: `~/code/pbg-biomodels/scripts/_catalog/modules.json`

- [ ] **Step 1: Inspect current state**

```bash
cat ~/code/pbg-biomodels/pyproject.toml
cat ~/code/pbg-biomodels/workspace.yaml
cat ~/code/pbg-biomodels-bundle/pyproject.toml  # for the delta
grep -n "bundle\|biomodels-bundle\|biomodels_bundle" ~/code/pbg-biomodels/scripts/_catalog/modules.json
```

- [ ] **Step 2: Update `pyproject.toml` dependencies**

Add to `[project].dependencies`:
- `pbg-tellurium`
- `pbg-copasi`
- Any bundle dep not already listed in pbg-biomodels: `matplotlib`, `rest-process`, `biomodels` (the BioModels API client), `python-libsedml`, `python-libsbml`. Skip `tellurium` and `copasi-basico` — they come transitively via pbg-tellurium and pbg-copasi respectively.

Remove from `[project].dependencies` (if present): `pbg-biomodels-bundle`.

- [ ] **Step 3: Update `workspace.yaml`**

If it lists `pbg-biomodels-bundle` as a workspace member, remove that entry. Otherwise no change.

- [ ] **Step 4: Update `scripts/_catalog/modules.json`**

Remove or update any entries referencing `pbg_biomodels_bundle` or `pbg-biomodels-bundle`. If entries describe specific bundle classes (`LocalCopasiUTCStep`, `LocalTelluriumUTCStep`, `LoadBiomodelStep`, `SimulatorComparisonStep`), update the addresses/paths to point at their new homes in `pbg_biomodels`.

- [ ] **Step 5: Reinstall editable**

```bash
cd ~/code/pbg-biomodels && uv pip install -e .
```
Expected: install completes; `pbg-tellurium` and `pbg-copasi` resolve.

- [ ] **Step 6: Commit**

```bash
cd ~/code/pbg-biomodels
git add pyproject.toml workspace.yaml scripts/_catalog/modules.json
git commit -m "pyproject + workspace + catalog: depend on pbg-tellurium/pbg-copasi; drop bundle"
```

---

## Task B5: Update existing tests + final verification

**Files:**
- Modify: `~/code/pbg-biomodels/tests/test_compare_biomodel_generator.py`
- Modify: `~/code/pbg-biomodels/tests/test_compare_overlay_multi.py`

- [ ] **Step 1: Update test files for new addresses**

```bash
cd ~/code/pbg-biomodels
grep -rn "pbg_biomodels_bundle" tests/
```

Any matches → rewrite `pbg_biomodels_bundle` → `pbg_biomodels` in those files. Also update step-address strings if they reference the old `local_simulators.LocalCopasiUTCStep` etc. — should become `simulators.BiomodelsCopasiStep` / `simulators.BiomodelsTelluriumStep`.

- [ ] **Step 2: Run the full test suite**

```bash
cd ~/code/pbg-biomodels && .venv/bin/python -m pytest tests/ -q
```
Expected: ALL tests pass (2 new wrapper tests + the existing compare_biomodel_generator + compare_overlay_multi tests).

If a test fails, investigate:
- Address-related failures → check that addresses match the new homes
- Output-shape failures in tellurium tests → check the reshape logic in BiomodelsTelluriumStep
- Import failures → check that pyproject deps + sed rewrites are complete

- [ ] **Step 3: Final import sweep — confirm no `pbg_biomodels_bundle` references remain in pbg-biomodels**

```bash
cd ~/code/pbg-biomodels
grep -rn "pbg_biomodels_bundle\|pbg-biomodels-bundle" --include="*.py" --include="*.yaml" --include="*.json" --include="*.toml" --include="*.md" 2>/dev/null | grep -v ".egg-info"
```
Expected: NO output (or only inside documentation that explicitly notes the bundle was retired).

- [ ] **Step 4: Install pbg-biomodels into viva-munk's venv + restart dashboard, verify compare-biomodel discovery**

```bash
~/code/viva-munk/.venv/bin/python -m pip install -e ~/code/pbg-biomodels 2>&1 | tail -3 || \
  (cd ~/code/viva-munk && VIRTUAL_ENV=~/code/viva-munk/.venv uv pip install -q -e ~/code/pbg-biomodels)

# Restart the running dashboard
lsof -ti tcp:8765 | xargs -r kill 2>/dev/null; sleep 1
cd ~/code/viva-munk
.venv/bin/vivarium-dashboard serve --workspace . --port 8765 > /tmp/dash.log 2>&1 &
until curl -sS -o /dev/null -w "%{http_code}" http://localhost:8765/ 2>/dev/null | grep -q "^200$"; do sleep 0.5; done

# Verify compare-biomodel appears
curl -sS http://localhost:8765/api/composites | python3 -c "
import sys, json
d = json.load(sys.stdin)
names = [c['id'] for c in d.get('composites', [])]
biomodels = [n for n in names if 'biomodel' in n.lower()]
print('biomodels composites:', biomodels)
assert any('compare' in n for n in biomodels), 'compare-biomodel not discovered'
print('compare-biomodel is discoverable')
"
```
Expected: `compare-biomodel is discoverable`. If not, check the venv install + that compare-biomodel.py's `@composite_generator` decorator triggers at import.

- [ ] **Step 5: Commit (only if any test changes were made)**

```bash
cd ~/code/pbg-biomodels
git status --porcelain  # see what changed
git add tests/
git commit -m "update existing tests for new pbg_biomodels paths + addresses"
```

If no test changes were needed (tests didn't reference the bundle), skip the commit — no empty commit.

---

## Task B6: Retire pbg-biomodels-bundle (README + push)

**Files:**
- Modify: `~/code/pbg-biomodels-bundle/README.md`

- [ ] **Step 1: Inspect current bundle README**

```bash
cat ~/code/pbg-biomodels-bundle/README.md 2>/dev/null | head -20
```

- [ ] **Step 2: Replace with a "superseded" note**

Write a new short README to `~/code/pbg-biomodels-bundle/README.md`:

```markdown
# pbg-biomodels-bundle (superseded)

This package has been **superseded** by [pbg-biomodels](https://github.com/vivarium-collective/pbg-biomodels), which now contains all functionality that used to live here (LoadBiomodelStep, SimulatorComparisonStep, comparison math, run_biomodels CLI, analysis report builder), and imports simulator Steps directly from [pbg-copasi](https://github.com/vivarium-collective/pbg-copasi) and [pbg-tellurium](https://github.com/vivarium-collective/pbg-tellurium).

**No further development happens in this repository.** New work continues in pbg-biomodels.
```

- [ ] **Step 3: Commit on bundle's main**

```bash
cd ~/code/pbg-biomodels-bundle
git add README.md
git commit -m "doc: mark superseded by pbg-biomodels"
```

- [ ] **Step 4: Push (with user confirmation if it's a shared-state action)**

```bash
cd ~/code/pbg-biomodels-bundle && git push origin main
```

- [ ] **Step 5: User-driven step — archive the GitHub repo manually**

The implementer reports this step but does NOT perform it (archiving is hard to reverse):

> Recommended manual step (not done by this plan):
> ```bash
> gh repo archive vivarium-collective/pbg-biomodels-bundle
> ```
> Run this when you're confident pbg-biomodels covers everything.

---

## Final verification (after all 6 tasks)

```bash
# pbg-biomodels self-contained + tests pass
cd ~/code/pbg-biomodels && .venv/bin/python -m pytest tests/ -q

# No bundle references in pbg-biomodels source
grep -rn "pbg_biomodels_bundle\|pbg-biomodels-bundle" ~/code/pbg-biomodels/ --include="*.py" --include="*.toml" --include="*.yaml" --include="*.json" 2>/dev/null | grep -v ".egg-info"

# Dashboard sees compare-biomodel
curl -sS http://localhost:8765/api/composites | python3 -c "
import sys, json
d = json.load(sys.stdin)
for c in d.get('composites', []):
    if 'biomodel' in c.get('id','').lower():
        print(c['id'])
"

# Optional: run compare-biomodel end-to-end for BIOMD0000000001
# (via the dashboard UI, or via a programmatic Composite() instantiation)
```
