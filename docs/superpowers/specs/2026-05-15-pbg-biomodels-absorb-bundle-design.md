# pbg-biomodels absorbs pbg-biomodels-bundle — Design

**Date:** 2026-05-15
**Status:** Approved, ready for implementation plan
**Affects:** `~/code/pbg-biomodels` (absorbs functionality), `~/code/pbg-biomodels-bundle` (retired), pbg-tellurium + pbg-copasi (consumed as deps; not modified)

## Goal

Make `pbg-biomodels` self-contained so it no longer requires `pbg-biomodels-bundle`. Simulator Step classes are imported from `pbg-tellurium` and `pbg-copasi` (eliminating duplicated simulator wrapping code). Everything else the compare-biomodel composite + batch tooling needs (LoadBiomodelStep, SimulatorComparisonStep, comparison math, biomodel-loading logic, HTML report builder) moves verbatim into pbg-biomodels. `pbg-biomodels-bundle` is retired.

## Approach

`pbg-biomodels` absorbs the bundle's full content. New thin wrapper Steps inside `pbg-biomodels` adapt the canonical pbg-tellurium / pbg-copasi UTC Steps (config-based model spec) to the runtime-input contract the compare composite needs (model_source flows from LoadBiomodelStep). No changes to pbg-tellurium or pbg-copasi.

## File migrations

| Bundle file | → New location in pbg-biomodels | Treatment |
|---|---|---|
| `pbg_biomodels_bundle/steps/load_biomodel.py` | `pbg_biomodels/steps/load_biomodel.py` | verbatim copy + rewrite `pbg_biomodels_bundle.run_biomodels` import → `pbg_biomodels.run_biomodels` |
| `pbg_biomodels_bundle/steps/simulator_comparison.py` | `pbg_biomodels/steps/simulator_comparison.py` | verbatim copy + rewrite `pbg_biomodels_bundle.comparison` import → `pbg_biomodels.comparison` |
| `pbg_biomodels_bundle/steps/local_simulators.py` | 🗑️ **deleted** — superseded | replaced by new `pbg_biomodels/steps/simulators.py` |
| `pbg_biomodels_bundle/comparison.py` | `pbg_biomodels/comparison.py` | verbatim |
| `pbg_biomodels_bundle/run_biomodels.py` | `pbg_biomodels/run_biomodels.py` | verbatim + rewrite any self-references (`pbg_biomodels_bundle.X` → `pbg_biomodels.X`) |
| `pbg_biomodels_bundle/analysis.py` | `pbg_biomodels/analysis.py` | verbatim + rewrite imports |
| `pbg_biomodels_bundle/__init__.py` exports | merged into `pbg_biomodels/__init__.py` | union of public surfaces |
| `pbg_biomodels_bundle/steps/__init__.py` | `pbg_biomodels/steps/__init__.py` | verbatim |

`pbg_biomodels/steps/` directory is created as part of the migration if not already present.

## New file — `pbg_biomodels/steps/simulators.py`

Two thin wrapper Steps replacing `LocalCopasiUTCStep` and `LocalTelluriumUTCStep` from the bundle. They preserve the bundle's API contract (runtime inputs `{model_source, time, n_points}`, output `{result: numeric_result}`) while delegating simulation to the canonical pbg-tellurium / pbg-copasi classes.

### `BiomodelsCopasiStep`
- Imports `pbg_copasi.processes.CopasiUTCStep`.
- `inputs() = {model_source, time, n_points}` (runtime, matching the bundle's contract).
- `outputs() = {result: numeric_result}`.
- `update(state)`: instantiates `CopasiUTCStep` with `config={'model_source': state['model_source'], 'duration': state['time'], 'n_points': state['n_points']}` (or whatever exact config keys the canonical class accepts — verify against `CopasiUTCStep.config_schema` during implementation). Calls inner `update({})`. The canonical CopasiUTCStep already emits `{result: numeric_result}` — pass through.

### `BiomodelsTelluriumStep`
- Imports `pbg_tellurium.TelluriumUTCStep`.
- Same `inputs()` / `outputs()` contract as `BiomodelsCopasiStep`.
- `update(state)`: instantiates `TelluriumUTCStep` with config including `model=state['model_source']`, `model_format='sbml'`, `start_time=0.0`, `end_time=float(state['time'])`, `n_points=int(state['n_points'])`. Calls inner `update({})`. The canonical TelluriumUTCStep emits `{time_series, species_trajectories}` — RESHAPE to `{result: {time, columns, values}}` before returning:
  - `time = out['time_series']`
  - `columns = list(out['species_trajectories'].keys())`
  - `values = [[traj[c][r] for c in columns] for r in range(len(time))]` (row-major matrix)

The reshape is the one place where the API mismatch surfaces; it's localized to this one method.

## Compare-biomodel composite — address updates

In `pbg_biomodels/composites/compare_biomodel.py`, the 4 internal step addresses change:

| Constant | Old value | New value |
|---|---|---|
| `LOAD_STEP_ADDRESS` | `"local:pbg_biomodels_bundle.steps.load_biomodel.LoadBiomodelStep"` | `"local:pbg_biomodels.steps.load_biomodel.LoadBiomodelStep"` |
| `COPASI_STEP_ADDRESS` | `"local:pbg_biomodels_bundle.steps.local_simulators.LocalCopasiUTCStep"` | `"local:pbg_biomodels.steps.simulators.BiomodelsCopasiStep"` |
| `TELLURIUM_STEP_ADDRESS` | `"local:pbg_biomodels_bundle.steps.local_simulators.LocalTelluriumUTCStep"` | `"local:pbg_biomodels.steps.simulators.BiomodelsTelluriumStep"` |
| `COMPARISON_STEP_ADDRESS` | `"local:pbg_biomodels_bundle.steps.simulator_comparison.SimulatorComparisonStep"` | `"local:pbg_biomodels.steps.simulator_comparison.SimulatorComparisonStep"` |

No other changes to `compare_biomodel.py`. `VISUALIZATION_STEP_ADDRESS` already points at `pbg_biomodels.visualizations.compare_overlay.CompareOverlay` — unchanged.

## `pyproject.toml` updates

In `pbg-biomodels`:
- **Add dependencies:** `pbg-tellurium`, `pbg-copasi`, plus any bundle deps not already in pbg-biomodels (the bundle's pyproject lists `tellurium`, `copasi-basico`, `rest-process`, `biomodels`, `matplotlib`, `bigraph-viz` — `tellurium` and `copasi-basico` come transitively via pbg-tellurium / pbg-copasi and can be omitted; others should be added if not already present).
- **Remove dependency:** `pbg-biomodels-bundle` (if listed).

Reinstall editable in the dev venv after editing.

## Workspace + catalog references

Two files in pbg-biomodels reference `pbg-biomodels-bundle`:
- `pbg_biomodels/workspace.yaml` — remove the bundle from any workspace package list.
- `pbg_biomodels/scripts/_catalog/modules.json` — update or remove bundle-specific entries.

## Tests

**Update existing tests** (`tests/test_compare_biomodel_generator.py`, `tests/test_compare_overlay_multi.py`):
- Adjust any hardcoded `pbg_biomodels_bundle.*` strings to `pbg_biomodels.*`.
- Verify they pass after the migration.

**Add new tests** in `tests/test_simulators.py`:
- One test per wrapper: instantiate, call `update()` with a small SBML model (reuse the BIOMD model already in pbg-copasi or load from biocompose's models), assert the output shape is `{result: {time, columns, values}}` with non-empty arrays.

## Retiring `pbg-biomodels-bundle`

After pbg-biomodels passes all tests and the dashboard discovers compare-biomodel successfully:
- Final commit on bundle's `main`: replace `README.md` with a "Superseded — see github.com/vivarium-collective/pbg-biomodels" note.
- Push the commit.
- **Manual:** archive the GitHub repo via `gh repo archive vivarium-collective/pbg-biomodels-bundle`. Not done automatically because archiving is harder to reverse than a code change.

## Out of scope

- **No changes to pbg-tellurium or pbg-copasi.** The API mismatch is absorbed entirely by the two new wrapper Steps in pbg-biomodels.
- **No behavior changes to LoadBiomodelStep, SimulatorComparisonStep, or CompareOverlay** during the migration — they move verbatim.
- **No reformatting / cleanup** of the migrated files beyond the import-path rewrite required to make them work in their new home.
- **No renaming or reorganization** of `run_biomodels.py` (even though it's 837 lines with multiple concerns — preserve as-is, refactor later if desired).
- **No changes to the compare-biomodel composite's data flow** beyond the 4 address constants.
- **No automated archive of pbg-biomodels-bundle GitHub repo** — user does that manually post-merge.

## Verification

After implementation:
1. `cd ~/code/pbg-biomodels && pytest tests/ -q` — all pass.
2. `cd ~/code/pbg-biomodels && python -c "import pbg_biomodels; import pbg_biomodels.steps.simulators; print('ok')"` — imports clean.
3. Install pbg-biomodels into viva-munk's venv (`uv pip install -e ~/code/pbg-biomodels`) and restart the dashboard. Verify `compare-biomodel` composite appears in the Composite Explorer.
4. Run `compare-biomodel` for a small biomodel id (e.g., `BIOMD0000000001`) and verify it produces a comparison without traceback.

## Open questions

None — all clarifications resolved during brainstorming:
- Scope: full absorption of bundle into pbg-biomodels.
- Adapter location: thin wrapper Steps inside pbg-biomodels.
