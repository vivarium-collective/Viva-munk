# pbg-tellurium update + pbg-copasi new package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port biocompose's tellurium + COPASI Step/Process classes into proper pbg-* packages with a parallel Base + UTC + SteadyState hierarchy. pbg-tellurium gets refactored in place (existing `TelluriumStep` renamed to `TelluriumUTCStep`, new base class + SteadyState). pbg-copasi is scaffolded fresh as a full kit.

**Architecture:** Both packages follow pbg-tellurium's current shape (processes + composites + visualizations + demos + tests). `BaseTelluriumStep` and `BaseCopasi` provide shared model-loading + species-id caching; UTC and SteadyState concrete classes inherit. No cross-package dependencies; `model_path_resolution` is vendored inline in each. Preserves pbg-tellurium's existing multi-format (antimony + SBML) model loading.

**Tech Stack:** Python ≥3.10, `process-bigraph`, `bigraph-schema`, `tellurium` (Phase 1), `copasi-basico` + `python-copasi` (Phase 2). Both packages use the `@composite_generator` convention so they appear in the vivarium-dashboard Composite Explorer.

**Spec:** [`docs/superpowers/specs/2026-05-15-pbg-tellurium-pbg-copasi-design.md`](../specs/2026-05-15-pbg-tellurium-pbg-copasi-design.md)

**Source material:** `~/code/biocompose` (branch `biomodels`) — `biocompose/processes/tellurium_process.py` and `biocompose/processes/copasi_process.py`

**Working repos:**
- `~/code/pbg-tellurium` (exists; Phase 1 modifies)
- `~/code/pbg-copasi` (does NOT exist; Phase 2 creates)

---

## Phase 1 — pbg-tellurium update

Work entirely in `~/code/pbg-tellurium`. Each task ends with a commit on the package's `main` branch (the repo already lives on `main` per the survey).

### Task T1: Add `BaseTelluriumStep` + inline `_model_path_resolution`

**Files:**
- Modify: `~/code/pbg-tellurium/pbg_tellurium/processes.py`

- [ ] **Step 1: Add the `_model_path_resolution` helper near the top of `processes.py`** (after imports, before `_load_roadrunner`)

```python
def _model_path_resolution(model_source: str) -> str:
    """Resolve a model reference to a loadable path or URL.

    URLs pass through unchanged; relative paths resolve against Path.cwd().
    """
    if model_source.startswith(('http://', 'https://')):
        return model_source
    p = Path(model_source)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)
```

Add `from pathlib import Path` to imports if not already present.

- [ ] **Step 2: Add `BaseTelluriumStep` class** above the existing `TelluriumStep` (still un-renamed at this point)

```python
class BaseTelluriumStep(Step):
    """Abstract base for Tellurium-backed Steps.

    Provides shared SBML/antimony model loading via _load_roadrunner,
    plus species-id caching. Subclasses implement update() with the
    specific simulation they perform (UTC, steady state, etc.).
    """

    config_schema = {
        **TelluriumProcess.config_schema,
    }

    def _tellurium_initialize(self):
        model_source = _model_path_resolution(self.config['model'])
        self.rr = _load_roadrunner(
            model_source,
            model_format=self.config['model_format'],
            model_file=self.config.get('model_file', ''),
        )
        self.species_ids = list(self.rr.getFloatingSpeciesIds())
        self.reaction_ids = list(self.rr.getReactionIds())
        self._species_index = {sid: i for i, sid in enumerate(self.species_ids)}

    def initial_state(self):
        if not hasattr(self, 'rr'):
            self._tellurium_initialize()
        conc = self.rr.getFloatingSpeciesConcentrations()
        return {
            'species_concentrations': {
                sid: float(conc[i]) for i, sid in enumerate(self.species_ids)
            }
        }

    def inputs(self):
        return {}

    def outputs(self):
        return {'html': 'string'}  # overridden by concrete subclasses
```

Note the `_load_roadrunner` delegation — this preserves pbg-tellurium's multi-format support (the spec's "Multi-format support preserved" requirement).

- [ ] **Step 3: Verify the file still imports**

```bash
cd ~/code/pbg-tellurium && python -c "from pbg_tellurium.processes import BaseTelluriumStep; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pbg_tellurium/processes.py
git commit -m "add BaseTelluriumStep base class + _model_path_resolution helper"
```

### Task T2: Rename `TelluriumStep` → `TelluriumUTCStep` + refactor onto base

**Files:**
- Modify: `~/code/pbg-tellurium/pbg_tellurium/processes.py`
- Modify: `~/code/pbg-tellurium/pbg_tellurium/__init__.py`

- [ ] **Step 1: In `processes.py`, rename `class TelluriumStep(Step):` to `class TelluriumUTCStep(BaseTelluriumStep):`**

Keep its existing docstring (update wording: "One-shot UTC simulation Step ..."). Keep its `config_schema` additions (`start_time`, `end_time`, `n_points`) but merge with `BaseTelluriumStep.config_schema` instead of `TelluriumProcess.config_schema`:

```python
class TelluriumUTCStep(BaseTelluriumStep):
    """One-shot UTC simulation Step returning a dense trajectory.

    Loads a model, simulates a fixed span start-to-end, and returns
    the full time series as parallel lists. Use when you want a
    static trajectory rather than time-coupled stepping
    (TelluriumProcess covers the incremental case).
    """

    config_schema = {
        **BaseTelluriumStep.config_schema,
        'start_time': {'_type': 'float', '_default': 0.0},
        'end_time': {'_type': 'float', '_default': 10.0},
        'n_points': {'_type': 'integer', '_default': 101},
    }

    def outputs(self):
        return {
            'time_series': 'overwrite[list]',
            'species_trajectories': 'overwrite[map[list]]',
        }

    def update(self, state):
        self._tellurium_initialize()
        # Keep the existing trajectory-simulation body from the prior
        # TelluriumStep.update implementation: use self.rr.simulate(
        # start_time, end_time, n_points), then split into time_series +
        # species_trajectories. self.rr is now provided by the base.
        ...
```

Preserve the existing `update()` body verbatim except that it now reads `self.rr` (set by `_tellurium_initialize`) instead of loading the roadrunner locally.

- [ ] **Step 2: Update `pbg_tellurium/__init__.py` exports**

Change:
```python
from pbg_tellurium.processes import TelluriumProcess, TelluriumStep
```
to:
```python
from pbg_tellurium.processes import (
    TelluriumProcess,
    BaseTelluriumStep,
    TelluriumUTCStep,
)
```

Update the `__all__` list accordingly:
```python
__all__ = [
    'TelluriumProcess',
    'BaseTelluriumStep',
    'TelluriumUTCStep',
]
```

- [ ] **Step 3: Verify the package still imports**

```bash
cd ~/code/pbg-tellurium && python -c "from pbg_tellurium import TelluriumUTCStep, BaseTelluriumStep, TelluriumProcess; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pbg_tellurium/processes.py pbg_tellurium/__init__.py
git commit -m "rename TelluriumStep -> TelluriumUTCStep; refactor onto BaseTelluriumStep"
```

### Task T3: Update composites + tests for the rename

**Files:**
- Modify: `~/code/pbg-tellurium/pbg_tellurium/composites/__init__.py`
- Modify: `~/code/pbg-tellurium/tests/test_processes.py`

- [ ] **Step 1: In `composites/__init__.py`, update all references**

Three references per the earlier grep (lines 27, 114, 119):
- Line 27 import: `from pbg_tellurium.processes import TelluriumProcess, TelluriumUTCStep`
- Line 114 docstring: `"""Return a core with TelluriumProcess + TelluriumUTCStep, the RAM emitter, ..."""`
- Line 119 register call: `core.register_link('TelluriumUTCStep', TelluriumUTCStep)`

- [ ] **Step 2: In `tests/test_processes.py`, update all references**

Three references (lines 5, 30, 148):
- Line 5 import: `from pbg_tellurium.processes import TelluriumProcess, TelluriumUTCStep`
- Line 30 register: `c.register_link('TelluriumUTCStep', TelluriumUTCStep)`
- Line 148 instantiation: `step = TelluriumUTCStep(...)`

- [ ] **Step 3: Run the existing tests to confirm rename is clean**

```bash
cd ~/code/pbg-tellurium && python -m pytest tests/test_processes.py -q
```
Expected: all existing tests pass (no behavior change, only renames).

- [ ] **Step 4: Commit**

```bash
git add pbg_tellurium/composites/__init__.py tests/test_processes.py
git commit -m "update composites + tests for TelluriumStep -> TelluriumUTCStep rename"
```

### Task T4: Add `TelluriumSteadyStateStep` + composite + test

**Files:**
- Modify: `~/code/pbg-tellurium/pbg_tellurium/processes.py`
- Modify: `~/code/pbg-tellurium/pbg_tellurium/__init__.py`
- Modify: `~/code/pbg-tellurium/pbg_tellurium/composites/__init__.py`
- Modify: `~/code/pbg-tellurium/tests/test_processes.py`

- [ ] **Step 1: Add `TelluriumSteadyStateStep` to `processes.py`** (port from biocompose `tellurium_process.py` lines 139-205)

```python
class TelluriumSteadyStateStep(BaseTelluriumStep):
    """Steady-state solve via Tellurium / roadrunner.

    Loads the model and computes steady-state species concentrations
    rather than a trajectory.
    """

    config_schema = {
        **BaseTelluriumStep.config_schema,
    }

    def outputs(self):
        return {
            'steady_state_concentrations': 'overwrite[map[float]]',
        }

    def update(self, state):
        self._tellurium_initialize()
        # Port body from biocompose/processes/tellurium_process.py:139-205:
        # call self.rr.steadyState() (or equivalent), then read
        # self.rr.getFloatingSpeciesConcentrations() and return as a
        # {species_id: concentration} dict under steady_state_concentrations.
        ...
```

Port the steady-state computation logic faithfully from biocompose's `TelluriumSteadyStateStep`.

- [ ] **Step 2: Update `__init__.py` to export `TelluriumSteadyStateStep`**

```python
from pbg_tellurium.processes import (
    TelluriumProcess,
    BaseTelluriumStep,
    TelluriumUTCStep,
    TelluriumSteadyStateStep,
)
__all__ = [
    'TelluriumProcess',
    'BaseTelluriumStep',
    'TelluriumUTCStep',
    'TelluriumSteadyStateStep',
]
```

- [ ] **Step 3: Add a `@composite_generator` builder for SteadyState in `composites/__init__.py`**

```python
@composite_generator(
    name='tellurium-steady-state',
    description='Solve for steady-state species concentrations of an SBML model.',
    parameters={
        'model': {'type': 'string', 'description': 'Path or URL to the model'},
        'model_format': {'type': 'string', 'default': 'sbml'},
    },
)
def tellurium_steady_state_document(core=None, *, model, model_format='sbml'):
    """Composite that runs a Tellurium steady-state solve."""
    return {
        'state': {
            'ss_step': {
                '_type': 'process',
                'address': 'local:TelluriumSteadyStateStep',
                'config': {'model': model, 'model_format': model_format},
                'outputs': {'steady_state_concentrations': ['ss_results']},
            },
        },
    }
```

Also register the new step in the core builder (next to where `TelluriumUTCStep` gets registered):
```python
core.register_link('TelluriumSteadyStateStep', TelluriumSteadyStateStep)
```

- [ ] **Step 4: Add a test for `TelluriumSteadyStateStep` in `tests/test_processes.py`**

```python
def test_tellurium_steady_state_step_runs():
    """SteadyStateStep loads a model and returns species concentrations."""
    from pbg_tellurium.processes import TelluriumSteadyStateStep
    # Use a small SBML/antimony model — mirror what the existing
    # TelluriumUTCStep test uses.
    step = TelluriumSteadyStateStep(
        config={'model': '<test model>', 'model_format': 'antimony'},
    )
    out = step.update({})
    assert 'steady_state_concentrations' in out
    assert isinstance(out['steady_state_concentrations'], dict)
    assert len(out['steady_state_concentrations']) > 0
```

Use the same example model the existing `TelluriumUTCStep` test uses, so the test is self-contained.

- [ ] **Step 5: Run the full test suite**

```bash
cd ~/code/pbg-tellurium && python -m pytest tests/ -q
```
Expected: all tests pass including the new SteadyState test.

- [ ] **Step 6: Commit**

```bash
git add pbg_tellurium/processes.py pbg_tellurium/__init__.py pbg_tellurium/composites/__init__.py tests/test_processes.py
git commit -m "add TelluriumSteadyStateStep + composite + test"
```

### Task T5: pyproject.toml updates + final verify

**Files:**
- Modify: `~/code/pbg-tellurium/pyproject.toml`

- [ ] **Step 1: Bump `requires-python` and add `tellurium` to dependencies**

In `[project]`:
```toml
requires-python = ">=3.10"
dependencies = [
    "process-bigraph>=0.0.10",
    "bigraph-schema",
    "tellurium",
]
```

- [ ] **Step 2: Reinstall editable to pick up dep changes**

```bash
cd ~/code/pbg-tellurium && uv pip install -e .
```
Expected: install completes; `tellurium` resolves.

- [ ] **Step 3: Run the full test suite as a final gate**

```bash
cd ~/code/pbg-tellurium && python -m pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "pyproject: add tellurium dep; bump requires-python to 3.10"
```

---

## Phase 2 — pbg-copasi new package

Create `~/code/pbg-copasi` from scratch. Each task ends with a commit on the package's `main` branch (initialize a fresh git repo in Task C1).

### Task C1: Scaffold the package + pyproject.toml + git init

**Files:**
- Create: `~/code/pbg-copasi/pyproject.toml`
- Create: `~/code/pbg-copasi/pbg_copasi/__init__.py` (empty initially)
- Create: `~/code/pbg-copasi/pbg_copasi/composites/__init__.py` (empty)
- Create: `~/code/pbg-copasi/tests/__init__.py` (empty)
- Create: `~/code/pbg-copasi/.gitignore`

- [ ] **Step 1: Create the directory tree**

```bash
mkdir -p ~/code/pbg-copasi/pbg_copasi/composites ~/code/pbg-copasi/tests ~/code/pbg-copasi/demo
touch ~/code/pbg-copasi/pbg_copasi/__init__.py
touch ~/code/pbg-copasi/pbg_copasi/composites/__init__.py
touch ~/code/pbg-copasi/tests/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

Mirror pbg-tellurium's structure. Minimum viable:

```toml
[project]
name = "pbg-copasi"
version = "0.1.0"
description = "process-bigraph-compatible COPASI Steps and Processes (UTC + SteadyState)"
requires-python = ">=3.10"
dependencies = [
    "process-bigraph>=0.0.10",
    "bigraph-schema",
    "copasi-basico",
    "python-copasi",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pbg_copasi"]
```

Check pbg-tellurium's `pyproject.toml` for the exact `[build-system]` + `[tool.hatch.*]` form and mirror it.

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.egg-info/
.venv/
*.pyc
```

- [ ] **Step 4: Initialize git + first commit**

```bash
cd ~/code/pbg-copasi
git init
git add .
git commit -m "scaffold pbg-copasi package (empty)"
```

- [ ] **Step 5: Verify editable install works in the active venv**

```bash
cd ~/code/pbg-copasi && uv pip install -e .
```
Expected: install completes; `copasi-basico` and `python-copasi` resolve.

### Task C2: Port the 4 process classes into `processes.py`

**Files:**
- Create: `~/code/pbg-copasi/pbg_copasi/processes.py`

- [ ] **Step 1: Write `processes.py`** — port from `~/code/biocompose/biocompose/processes/copasi_process.py` (lines 21-364)

Structure:

```python
"""COPASI-backed Step and Process implementations.

Ports BaseCopasi mixin + CopasiUTCStep + CopasiSteadyStateStep +
CopasiUTCProcess from biocompose's copasi_process.py. Inlines the
model_path_resolution helper.
"""
import os
from pathlib import Path
from typing import Dict, Any

from pandas import DataFrame
from process_bigraph import Process, Step
import COPASI
from basico import (
    # mirror biocompose imports — lines 8-15 of copasi_process.py
)


def _model_path_resolution(model_source: str) -> str:
    """Resolve a model reference to a loadable path or URL."""
    if model_source.startswith(('http://', 'https://')):
        return model_source
    p = Path(model_source)
    if not p.is_absolute():
        p = Path.cwd() / p
    return str(p)


def _set_initial_concentrations(changes, dm):
    # Port verbatim from copasi_process.py lines 21-43
    ...


def _get_transient_concentration(name, dm):
    # Port verbatim from copasi_process.py lines 44-56
    ...


class BaseCopasi:
    """Shared COPASI model-load + concentration helpers.

    Mixin (no Process/Step base). Concrete classes inherit BOTH
    Step/Process and BaseCopasi.
    """
    # Port verbatim from copasi_process.py lines 58-105.
    # Replace any usage of the imported model_path_resolution with the
    # local _model_path_resolution helper.


class CopasiUTCStep(Step, BaseCopasi):
    """One-shot UTC trajectory via COPASI."""
    # Port verbatim from copasi_process.py lines 107-175


class CopasiSteadyStateStep(Step, BaseCopasi):
    """Steady-state solve via COPASI."""
    # Port verbatim from copasi_process.py lines 176-268


class CopasiUTCProcess(Process, BaseCopasi):
    """UTC simulation as an incremental Process (vs the Step variant)."""
    # Port verbatim from copasi_process.py lines 269-364
```

The "port verbatim" lines should be a literal copy from the biocompose source, with one targeted change: any `model_path_resolution(...)` call becomes `_model_path_resolution(...)` (the local helper).

- [ ] **Step 2: Verify import works**

```bash
cd ~/code/pbg-copasi && python -c "from pbg_copasi.processes import BaseCopasi, CopasiUTCStep, CopasiSteadyStateStep, CopasiUTCProcess; print('ok')"
```
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pbg_copasi/processes.py
git commit -m "port BaseCopasi + CopasiUTCStep + CopasiSteadyStateStep + CopasiUTCProcess from biocompose"
```

### Task C3: `__init__.py` + `types.py` + `tests/test_processes.py`

**Files:**
- Modify: `~/code/pbg-copasi/pbg_copasi/__init__.py`
- Create: `~/code/pbg-copasi/pbg_copasi/types.py`
- Create: `~/code/pbg-copasi/tests/test_processes.py`

- [ ] **Step 1: Write `pbg_copasi/__init__.py`** — exports

```python
"""pbg-copasi — COPASI-backed Steps and Processes for process-bigraph."""
from pbg_copasi.processes import (
    BaseCopasi,
    CopasiUTCStep,
    CopasiSteadyStateStep,
    CopasiUTCProcess,
)

__all__ = [
    'BaseCopasi',
    'CopasiUTCStep',
    'CopasiSteadyStateStep',
    'CopasiUTCProcess',
]
```

- [ ] **Step 2: Write `pbg_copasi/types.py`** — minimal, mirror pbg-tellurium's

Look at `~/code/pbg-tellurium/pbg_tellurium/types.py`. If it's minimal (just defines a `register_types(core)` function with no custom types), copy that shape with a no-op `register_types(core)`. If pbg-tellurium has actual type registrations, evaluate whether pbg-copasi needs equivalents.

- [ ] **Step 3: Write `tests/test_processes.py`**

Port from biocompose's `run_copasi_utc()` (line 365) and `run_copasi_ss()` (line 382) test runners, plus add a test for `CopasiUTCProcess`. Use an example SBML model — either a small inline one or a fixture path.

```python
"""Tests for pbg_copasi.processes — UTC Step, SteadyState Step, UTC Process."""
import pytest

from process_bigraph import ProcessTypes
from pbg_copasi.processes import (
    CopasiUTCStep, CopasiSteadyStateStep, CopasiUTCProcess,
)


@pytest.fixture
def core():
    c = ProcessTypes()
    c.register_link('CopasiUTCStep', CopasiUTCStep)
    c.register_link('CopasiSteadyStateStep', CopasiSteadyStateStep)
    c.register_link('CopasiUTCProcess', CopasiUTCProcess)
    return c


def test_copasi_utc_step_runs_to_completion(core):
    """Port of biocompose's run_copasi_utc(); produces a non-empty trajectory."""
    # Use a small SBML model — pull from biocompose's test models if needed.
    step = CopasiUTCStep(config={'model_source': '<sbml path>', ...})
    out = step.update({})
    assert 'time_series' in out
    assert len(out['time_series']) > 0


def test_copasi_steady_state_step_returns_concentrations(core):
    """Port of biocompose's run_copasi_ss(); returns finite species concentrations."""
    step = CopasiSteadyStateStep(config={'model_source': '<sbml path>', ...})
    out = step.update({})
    assert 'steady_state_concentrations' in out
    assert len(out['steady_state_concentrations']) > 0


def test_copasi_utc_process_advances_one_step(core):
    """The Process variant advances incrementally."""
    proc = CopasiUTCProcess(config={'model_source': '<sbml path>', ...})
    initial = proc.initial_state()
    update = proc.update(initial, interval=1.0)
    assert update is not None
```

Fill in the exact `<sbml path>` and config keys to match what biocompose's process classes expect (read their `config_schema` definitions).

- [ ] **Step 4: Run the tests**

```bash
cd ~/code/pbg-copasi && python -m pytest tests/test_processes.py -q
```
Expected: all 3 process tests pass.

- [ ] **Step 5: Commit**

```bash
git add pbg_copasi/__init__.py pbg_copasi/types.py tests/test_processes.py
git commit -m "add __init__ + types + test_processes covering all 4 ported classes"
```

### Task C4: `visualizations.py` + composites + `test_composites.py`

**Files:**
- Create: `~/code/pbg-copasi/pbg_copasi/visualizations.py`
- Modify: `~/code/pbg-copasi/pbg_copasi/composites/__init__.py`
- Create: `~/code/pbg-copasi/tests/test_composites.py`

- [ ] **Step 1: Write `pbg_copasi/visualizations.py`** — adapt from pbg-tellurium

Read `~/code/pbg-tellurium/pbg_tellurium/visualizations.py`. Port its viz Step classes verbatim into pbg-copasi, with the imports changed (no `pbg_tellurium` references — make the file self-contained). The output shapes (`time_series` + `species_trajectories`) are the same across both packages, so the viz code transfers cleanly.

- [ ] **Step 2: Write `pbg_copasi/composites/__init__.py`** — 3 `@composite_generator` builders

```python
"""Starter composites for pbg-copasi.

Three builders covering: UTC trajectory via Step, UTC via Process,
and SteadyState. All decorated with @composite_generator so they
appear in the vivarium-dashboard Composite Explorer.
"""
from process_bigraph import ProcessTypes
from pbg_superpowers.composite_generator import composite_generator

from pbg_copasi.processes import (
    CopasiUTCStep, CopasiSteadyStateStep, CopasiUTCProcess,
)


def core_with_copasi():
    """Return a core with all pbg-copasi classes registered."""
    core = ProcessTypes()
    core.register_link('CopasiUTCStep', CopasiUTCStep)
    core.register_link('CopasiSteadyStateStep', CopasiSteadyStateStep)
    core.register_link('CopasiUTCProcess', CopasiUTCProcess)
    return core


@composite_generator(
    name='copasi-utc-step',
    description='One-shot UTC trajectory via CopasiUTCStep.',
    parameters={
        'model_source': {'type': 'string'},
        'start_time': {'type': 'float', 'default': 0.0},
        'end_time': {'type': 'float', 'default': 10.0},
        'n_points': {'type': 'integer', 'default': 101},
    },
    default_n_steps=1,
)
def copasi_utc_step_document(core=None, *, model_source, start_time=0.0, end_time=10.0, n_points=101):
    """Composite running a one-shot UTC trajectory via the Step variant."""
    return {
        'state': {
            'utc': {
                '_type': 'step',
                'address': 'local:CopasiUTCStep',
                'config': {
                    'model_source': model_source,
                    'start_time': start_time,
                    'end_time': end_time,
                    'n_points': n_points,
                },
                'outputs': {
                    'time_series': ['traj_time'],
                    'species_trajectories': ['traj_species'],
                },
            },
        },
    }


@composite_generator(
    name='copasi-utc-process',
    description='Incremental UTC simulation via CopasiUTCProcess.',
    parameters={
        'model_source': {'type': 'string'},
    },
    default_n_steps=100,
)
def copasi_utc_process_document(core=None, *, model_source):
    """Composite running UTC incrementally as a Process."""
    return {
        'state': {
            'utc': {
                '_type': 'process',
                'address': 'local:CopasiUTCProcess',
                'config': {'model_source': model_source},
                'inputs': {'species_concentrations': ['species']},
                'outputs': {'species_concentrations': ['species']},
            },
        },
    }


@composite_generator(
    name='copasi-steady-state',
    description='Solve for steady-state species concentrations via COPASI.',
    parameters={
        'model_source': {'type': 'string'},
    },
    default_n_steps=1,
)
def copasi_steady_state_document(core=None, *, model_source):
    """Composite that runs a COPASI steady-state solve."""
    return {
        'state': {
            'ss': {
                '_type': 'step',
                'address': 'local:CopasiSteadyStateStep',
                'config': {'model_source': model_source},
                'outputs': {'steady_state_concentrations': ['ss_results']},
            },
        },
    }
```

Adjust the exact composite-document shapes (`'_type': 'step'` vs `'process'`, the inputs/outputs wiring) to match pbg-tellurium's working composite-document conventions. Read `~/code/pbg-tellurium/pbg_tellurium/composites/__init__.py` for the canonical pattern and follow it.

- [ ] **Step 3: Write `tests/test_composites.py`**

Mirror pbg-tellurium's `tests/test_composites.py`. Each of the 3 builders gets a test that:
1. Calls the builder with a small example model
2. Constructs a Composite from the result
3. Runs it for a few steps
4. Asserts non-empty output

- [ ] **Step 4: Run the composite tests**

```bash
cd ~/code/pbg-copasi && python -m pytest tests/test_composites.py -q
```
Expected: 3 composite tests pass.

- [ ] **Step 5: Commit**

```bash
git add pbg_copasi/visualizations.py pbg_copasi/composites/__init__.py tests/test_composites.py
git commit -m "add visualizations + 3 starter composites + test_composites"
```

### Task C5: Demos + README + final verify

**Files:**
- Create: `~/code/pbg-copasi/demo/demo_report.py`
- Create: `~/code/pbg-copasi/demo/composite_report.py`
- Create: `~/code/pbg-copasi/README.md`

- [ ] **Step 1: Port the two demo scripts**

Read `~/code/pbg-tellurium/demo/demo_report.py` and `~/code/pbg-tellurium/demo/composite_report.py`. Port their shape verbatim into `~/code/pbg-copasi/demo/`, adapting:
- Import `from pbg_copasi import ...` instead of pbg_tellurium
- Use a small COPASI-friendly model (SBML)
- Use the new `copasi-utc-step` / `copasi-steady-state` composite builders

The goal is a demo script that produces a runnable report for each variant, matching pbg-tellurium's demo experience.

- [ ] **Step 2: Write `README.md`**

```markdown
# pbg-copasi

process-bigraph-compatible COPASI Steps and Processes (UTC + SteadyState).

## Install

\`\`\`bash
uv pip install -e .
\`\`\`

## Classes

- `CopasiUTCStep` — one-shot UTC trajectory (Step)
- `CopasiUTCProcess` — incremental UTC simulation (Process)
- `CopasiSteadyStateStep` — steady-state solve
- `BaseCopasi` — shared model-load mixin

## Composites

Three `@composite_generator` builders are exposed for dashboard discovery:
`copasi-utc-step`, `copasi-utc-process`, `copasi-steady-state`.

## Tests

\`\`\`bash
python -m pytest tests/ -q
\`\`\`
```

- [ ] **Step 3: Run the demo scripts to confirm they work end-to-end**

```bash
cd ~/code/pbg-copasi && python demo/demo_report.py
cd ~/code/pbg-copasi && python demo/composite_report.py
```
Expected: both run without error and produce report output.

- [ ] **Step 4: Final full test gate**

```bash
cd ~/code/pbg-copasi && python -m pytest tests/ -q
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add demo/demo_report.py demo/composite_report.py README.md
git commit -m "add demos + README; final verification"
```

---

## Verification (after all 10 tasks)

```bash
# Phase 1: pbg-tellurium
cd ~/code/pbg-tellurium && python -m pytest tests/ -q
# Expected: all tests pass (existing tests + new SteadyState test)

# Phase 2: pbg-copasi
cd ~/code/pbg-copasi && python -m pytest tests/ -q
# Expected: all tests pass (test_processes + test_composites)

# Dashboard surface check (manual): open the vivarium dashboard against
# a workspace that has both packages installed; confirm the new
# tellurium-steady-state, copasi-utc-step, copasi-utc-process, and
# copasi-steady-state composites appear in the Composite Explorer.
```
