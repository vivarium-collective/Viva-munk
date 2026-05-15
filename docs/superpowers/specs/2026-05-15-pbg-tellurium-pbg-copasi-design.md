# pbg-tellurium update + pbg-copasi new package — Design

**Date:** 2026-05-15
**Status:** Approved, ready for implementation plan
**Affects:** `~/code/pbg-tellurium` (update), `~/code/pbg-copasi` (new — to be created)
**Source material:** `~/code/biocompose` (branch `biomodels`) — `biocompose/processes/tellurium_process.py` and `biocompose/processes/copasi_process.py`

## Goal

Port biocompose's tellurium and COPASI Step/Process classes into proper `pbg-*` packages, giving each tool a base-class hierarchy that exposes Uniform Time Course (UTC) and Steady-State variants, plus a Process variant where biocompose has one. Both packages end up as full kits (processes + composites + visualizations + demos + tests) that are dashboard-discoverable via the `@composite_generator` convention.

## Approach (Approach A — replace + promote)

Recognize that pbg-tellurium's existing `TelluriumStep` is functionally a UTC step in everything but name (its docstring: "one-shot simulation Step that returns a dense trajectory ... loads a model, simulates a fixed span start-to-end"). Rename it to `TelluriumUTCStep` and adopt biocompose's `BaseTelluriumStep` + `TelluriumSteadyStateStep` factoring. Apply the same hierarchy to pbg-copasi as a net-new package, including the additional `CopasiUTCProcess` variant that exists in biocompose.

No deprecation alias for the rename — the package has a small known surface (4 internal references), all updated together.

## Package layout (both packages, parallel)

```
pbg_<pkg>/
  __init__.py              — public exports
  types.py                 — type registrations (pbg-tellurium has one already; pbg-copasi adds as needed)
  processes.py             — Process + Base + concrete Step subclasses
  visualizations.py        — viz Step subclasses (species trajectory plots, etc.)
  composites/__init__.py   — @composite_generator-decorated builders
tests/
  test_processes.py
  test_composites.py
demo/
  demo_report.py
  composite_report.py
pyproject.toml
README.md
```

pbg-tellurium already has this layout. pbg-copasi gets scaffolded to match.

## Process/Step inventory (after work)

### pbg-tellurium

| Class | Status | Source | Notes |
|---|---|---|---|
| `TelluriumProcess(Process)` | unchanged | existing | incremental time stepping; no biocompose equivalent |
| `BaseTelluriumStep(Step)` | new | biocompose `TelluriumStep` (base) | abstract base: `_tellurium_initialize`, species-id caching, base `inputs()`/`initial_state()` |
| `TelluriumUTCStep(BaseTelluriumStep)` | renamed + refactored | existing `TelluriumStep` + biocompose `TelluriumUTCStep` | one-shot trajectory; preserves pbg-tellurium's richer config_schema (`start_time`/`end_time`/`n_points`) and output shape (`time_series` + `species_trajectories`); rebases its model-loading onto `BaseTelluriumStep._tellurium_initialize` |
| `TelluriumSteadyStateStep(BaseTelluriumStep)` | new | biocompose `TelluriumSteadyStateStep` | steady-state solve |

### pbg-copasi (all new)

| Class | Source | Notes |
|---|---|---|
| `BaseCopasi` | biocompose `BaseCopasi` | mixin: shared model load (via basico/COPASI), concentration getters, helper functions `_set_initial_concentrations` / `_get_transient_concentration` |
| `CopasiUTCStep(Step, BaseCopasi)` | biocompose `CopasiUTCStep` | one-shot UTC trajectory |
| `CopasiSteadyStateStep(Step, BaseCopasi)` | biocompose `CopasiSteadyStateStep` | steady-state solve |
| `CopasiUTCProcess(Process, BaseCopasi)` | biocompose `CopasiUTCProcess` | incremental step variant; no tellurium counterpart in biocompose |

## Implementation merge — `TelluriumUTCStep`

The renamed/refactored `TelluriumUTCStep` MUST preserve, from pbg-tellurium's existing implementation:
- Config schema fields `start_time`, `end_time`, `n_points` (biocompose's `TelluriumUTCStep` has fewer; pbg-tellurium's is richer)
- Output shape: `time_series: overwrite[list]` + `species_trajectories: overwrite[map[list]]`
- The dense-trajectory return contract

AND adopt, from biocompose's hierarchy:
- Loading the model via `BaseTelluriumStep._tellurium_initialize` (cached `self.rr`, `self.species_ids`, `self.reaction_ids`, `self._species_index`)
- `initial_state()` returning the species-concentration dict

The body of `update()` should call `_tellurium_initialize` (from the base) and then run the time-course simulation as the existing implementation does, populating the dense trajectory output.

**Multi-format support preserved.** pbg-tellurium's existing `_load_roadrunner(model_source, model_format, model_file)` supports both `antimony` and `sbml` formats; biocompose's `_tellurium_initialize` is SBML-only. The new `BaseTelluriumStep._tellurium_initialize` MUST delegate to `_load_roadrunner` (passing `model_format` from `self.config`) — NOT call `te.loadSBMLModel` directly — so both `TelluriumUTCStep` and the new `TelluriumSteadyStateStep` inherit the multi-format capability. This avoids silently regressing pbg-tellurium's current capability set.

## `model_path_resolution` util

biocompose's `model_path_resolution` (9 lines, in `biocompose/processes/utils.py`) is used by both `tellurium_process.py` and `copasi_process.py`. Vendor it inline into each package — likely as a private `_model_path_resolution` helper at the top of `processes.py`. No shared `pbg-bio-utils` package; that's premature abstraction.

Simplify the vendored version: drop the `Path(__file__).parent.parent` heuristic (which assumes the biocompose layout). Resolve relative paths against `Path.cwd()` — matches how the dashboard launches simulation subprocesses. URLs (`http://`, `https://`) pass through unchanged, as in biocompose.

```python
def _model_path_resolution(model_source: str) -> str:
    if model_source.startswith(('http://', 'https://')):
        return model_source
    p = Path(model_source)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)
```

## Composites + visualizations

### pbg-tellurium
- Update existing composites in `pbg_tellurium/composites/__init__.py` to reference `TelluriumUTCStep` (was `TelluriumStep`).
- Update `core_with_tellurium()` (or the equivalent core builder) registration: `core.register_link('TelluriumUTCStep', TelluriumUTCStep)` (was `TelluriumStep`).
- Add ONE new `@composite_generator`-decorated builder exercising `TelluriumSteadyStateStep` so it appears in the dashboard's Composite Explorer.

### pbg-copasi
Three `@composite_generator`-decorated builders mirroring the pbg-tellurium pattern:
1. UTC trajectory via `CopasiUTCStep` (the one-shot Step path)
2. UTC trajectory via `CopasiUTCProcess` (the incremental-Process path)
3. SteadyState via `CopasiSteadyStateStep`

Each builder follows pbg-tellurium's existing shape: takes a `core` and parameter overrides, declares parameters in the `@composite_generator(...)` decorator, and includes a `default_n_steps` hint where relevant.

### Visualizations
pbg-copasi gets a `visualizations.py` that copies/adapts pbg-tellurium's viz Steps (species-trajectory plots driven by `time_series` + `species_trajectories` outputs). No cross-package dependency — pbg-copasi does NOT import from pbg-tellurium.

## Tests

### pbg-tellurium
- `tests/test_processes.py`: replace 3 references to `TelluriumStep` with `TelluriumUTCStep` (lines 5, 30, 148 in current file). Adapt the test that instantiates and steps the class so it still passes.
- Add `tests/test_processes.py::test_steady_state_step` covering `TelluriumSteadyStateStep`.

### pbg-copasi
- New `tests/test_processes.py` covering all 4 classes. Port biocompose's `run_copasi_utc()` and `run_copasi_ss()` test runners as proper pytest tests, plus a test for `CopasiUTCProcess`.
- New `tests/test_composites.py` covering the 3 starter composites — instantiate each via its `@composite_generator` builder, run a small number of steps, assert non-empty output.

## Dependencies / metadata

### pbg-tellurium `pyproject.toml`
- Add `tellurium` to `[project].dependencies` (currently missing per inspection).
- Bump `requires-python` from `>=3.9` to `>=3.10` — biocompose code uses `dict[str, dict]` and similar 3.10+ generic syntax.

### pbg-copasi `pyproject.toml` (new)
```toml
[project]
name = "pbg-copasi"
requires-python = ">=3.10"
dependencies = [
    "process-bigraph>=0.0.10",
    "bigraph-schema",
    "copasi-basico",
    "python-copasi",
]
```
Final exact versions of `process-bigraph` / `bigraph-schema` should match pbg-tellurium's at port time.

## Concrete file changes — pbg-tellurium

| File | Change |
|---|---|
| `pbg_tellurium/processes.py` | Rename `class TelluriumStep` → `class TelluriumUTCStep`. Add `class BaseTelluriumStep`. Refactor `TelluriumUTCStep` to inherit from `BaseTelluriumStep`. Add `class TelluriumSteadyStateStep(BaseTelluriumStep)`. Inline `_model_path_resolution` helper. |
| `pbg_tellurium/__init__.py` | Replace `TelluriumStep` export with `TelluriumUTCStep`. Add `BaseTelluriumStep`, `TelluriumSteadyStateStep` exports. |
| `pbg_tellurium/composites/__init__.py` | 3 references: `TelluriumStep` → `TelluriumUTCStep`. Add one new `@composite_generator` for SteadyState. |
| `tests/test_processes.py` | 3 references: `TelluriumStep` → `TelluriumUTCStep`. Add SteadyState test. |
| `pyproject.toml` | Add `tellurium` dep. Bump `requires-python = ">=3.10"`. |

## Concrete file changes — pbg-copasi (all new)

| File | Contents |
|---|---|
| `pyproject.toml` | as above |
| `pbg_copasi/__init__.py` | exports for `BaseCopasi`, `CopasiUTCStep`, `CopasiSteadyStateStep`, `CopasiUTCProcess` |
| `pbg_copasi/types.py` | mirror pbg-tellurium's `types.py` shape (empty / minimal if no custom types are needed) |
| `pbg_copasi/processes.py` | port `BaseCopasi`, `CopasiUTCStep`, `CopasiSteadyStateStep`, `CopasiUTCProcess` from biocompose's `copasi_process.py`; inline `_model_path_resolution` |
| `pbg_copasi/visualizations.py` | adapt pbg-tellurium's viz Steps for the copasi output shape (same shape, no cross-package import) |
| `pbg_copasi/composites/__init__.py` | 3 `@composite_generator` builders (UTC-via-Step, UTC-via-Process, SteadyState) |
| `tests/test_processes.py` | 4 class-level tests + the biocompose test-runner ports |
| `tests/test_composites.py` | 3 composite-instantiation tests |
| `demo/demo_report.py` + `demo/composite_report.py` | port pbg-tellurium's demo shapes |
| `README.md` | brief; mirror pbg-tellurium |

## Out of scope (explicit non-goals)

- **Cross-package comparison composite.** Biocompose's `copasi_tellurium_comparison.json` / `experiments/copasi_tellurium_comparison.py` will NOT be ported in this work. Separate effort if/when desired — would require a host repo that depends on both packages.
- **Shared `pbg-bio-utils` package.** `model_path_resolution` is 9 lines; vendoring inline is fine.
- **Biocompose `experiments/` migration.** Out of scope.
- **Cross-package install coupling.** Neither package auto-installs the other into its venv.
- **Bumping `process-bigraph` / `bigraph-schema` versions** beyond what pbg-tellurium currently uses.

## Verification

After implementation:
1. `cd ~/code/pbg-tellurium && pytest tests/ -q` — all pass, including the renamed-class tests and the new SteadyState test.
2. `cd ~/code/pbg-copasi && pytest tests/ -q` — all pass.
3. Open the vivarium dashboard against a workspace that has both packages installed (e.g., a viva-munk-style workspace with both as dependencies, OR run each package's own dashboard) — confirm the new `@composite_generator` builders appear in the Composite Explorer and a small test-run completes.

## Open questions

None — all clarifications resolved during brainstorming:
- Structural shape: Approach A (replace + promote)
- Rename: yes, no deprecation alias
- pbg-copasi scope: full kit parallel to pbg-tellurium
