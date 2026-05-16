---
title: Studies — first-class research unit with default-runtime metadata and Save-as-Study flow
date: 2026-05-13
status: draft
author: Eran Agmon
brainstorm_session: viva-munk (sqlemitter branch)
phase: 1 of 3 (Studies → Study edges → Investigation graph view)
---

# Studies — Phase 1: data model, default runtimes, and Save-as-Study

> **Scope note.** This spec covers Phase 1 only: defining a *Study* as a
> first-class persistent unit (one composite + accumulating runs), surfacing
> per-composite default runtimes in the Composite Explorer, and adding a
> one-click "Save as Study" promote from a Test Run. Phase 2 (edges between
> Studies, planned states, conclusion-spawns-follow-up) and Phase 3 (the
> Investigations tab as a rendered DAG) are deliberately deferred and called
> out in §6.

## 1. Context

The dashboard today has three run-related tiers:

- **Test Run** — Composite Explorer scratchpad. Runs persist to
  `.pbg/composite-runs.db`. *Replaced* on each new test run for the same
  composite (single-row scratch). Good for exploration, lossy for any
  accumulation.
- **Investigation** — persistent named container at
  `investigations/<name>/spec.yaml` + `runs.db` + `viz/` + `notes.md`. Has
  `baseline`, `variants`, `groups`, `comparisons`, `conclusions`,
  `parameter_overrides`, and `process_overrides` (process-level
  perturbations). The current implementation is already ~90% Study-shaped
  but is multi-composite by default (the `composites: [...]` list).
- **Catalog promote** — saves a parameterized variant back to the
  workspace's reusable composite catalog as YAML.

In the user's mental model:

- **Study** = a bundle of simulations sharing one baseline composite,
  parameter-swept and intervened-upon, producing a conclusion.
- **Investigation** = a *chain/graph* of Studies, where one Study's
  conclusion can spawn a follow-up Study built on a variant or
  intervention of the previous composite. The Investigations tab is
  the rendered DAG of Study cards.

The existing system conflates the two: "Investigation" today actually
means what the user calls a "Study", and there is no
Investigation-as-DAG layer above it yet.

## 2. Goals (Phase 1)

1. **Study** becomes the first-class user-facing unit, replacing the
   current "Investigation" label everywhere in the dashboard UI.
2. Each Study has: `objective`, one `baseline` composite + params,
   `variants` (parameter overrides + process-level interventions),
   accumulated `runs`, `visualizations`, `conclusion`, and (reserved
   but unused) `parent_studies`.
3. Per-composite `default_n_steps` lives next to the composite
   generator and pre-fills the Composite Explorer.
4. After a Test Run, **Save as Study** promotes the run into a new
   Study in one click — no re-execution, the run's history rows are
   copied into the Study's own `runs.db`.
5. The Study Detail view is a stacked-cards page with affordances to
   add variants, run them, compare runs, render viz, and write the
   conclusion. Backend reuse where possible; no rewrites.

## 3. Non-goals

- **Edges between Studies.** `parent_studies: []` is reserved but unused
  in Phase 1. No "Spawn follow-up" button. (Phase 2.)
- **Graph rendering** of the Investigations tab. (Phase 3.)
- **Planned-but-not-run Study states.** Phase 1 statuses are `running`,
  `ran`, `complete`, `failed`. `planned` is added in Phase 2 alongside
  edges.
- **Multi-composite-per-Study.** A Study is one baseline composite. Today's
  multi-composite Investigations migrate to single-composite Studies by
  taking the first listed composite and dropping the rest with a warning.
- **Cross-Study comparisons** (running viz/stats across Study boundaries).
  Phase 2/3.
- **Distributed run execution, multi-user permissions, external API
  stability.** Out of scope forever for this design.

## 4. Architecture

The four pieces:

1. **Composite generator metadata** (`pbg-superpowers`) — new optional
   `default_n_steps: int` kwarg on `@composite_generator`, propagated
   into the catalog via `discover_all_composites()`, surfaced by
   `/api/composites` and pre-filled in the Composite Explorer.
2. **Study data model** (`studies/<name>/` directory) — `study.yaml`
   (schema_version: 3) + per-Study `runs.db` + sidecar
   `composites/`, `viz/`, `notes.md`. Migrated from `investigations/`.
3. **Save-as-Study flow** (dashboard server + frontend) — new endpoint
   `POST /api/study-create-from-run` that copies a scratchpad run into
   a new Study; new button + modal in Composite Explorer.
4. **Study Detail view** (dashboard frontend) — stacked-cards page at
   `/studies/<name>` replacing today's Investigation Detail page.
   Backed by existing `/api/investigation-*` handlers, aliased or
   renamed to `/api/study-*` (TBD in plan).

## 5. Component changes

### 5.1. Study data model + storage

**Directory layout:** `studies/<study-name>/`. One Study per directory; the
directory name is the slug.

```
studies/<name>/
├── study.yaml          # spec (renamed from spec.yaml)
├── runs.db             # SQLite: runs_meta + SQLiteEmitter `history`
├── composites/         # sidecar derived composites (one per intervention)
│   └── <variant>.yaml
├── viz/                # rendered viz HTML / extracted GIF
│   └── <run-or-variant>.html
└── notes.md            # free-form lab-notebook
```

**`study.yaml` shape** (schema_version: 3):

```yaml
schema_version: 3
name: chemotaxis-gradient-sweep
created: "2026-05-13"
status: ran                                     # running | ran | complete | failed
objective: |
  Does increasing the sensitivity parameter improve up-gradient bias
  when the gradient is shallow?
baseline:
  composite: viva_munk.composites.chemotaxis   # composite spec_id
  params:                                       # baseline param overrides
    n_cells: 12
    sensitivity: 2.0
    n_steps: 200                                # runtime
variants:
  - name: high-sensitivity
    intervention:
      description: "Triple sensitivity"
      parameter_overrides: {sensitivity: 6.0}
  - name: no-pressure
    intervention:
      description: "Remove the Pressure step"
      process_overrides:
        - {op: remove, path: ['pressure']}
    document: composites/no-pressure.yaml       # materialized sidecar
runs:
  - run_id: viva_munk.composites.chemotaxis__1715620800__a3f2c1
    variant: high-sensitivity                   # null = baseline
    label: "high-sens, seed=42"
    status: completed
    started_at: 1715620800.0
    completed_at: 1715620812.7
    n_steps: 200
    viz: viz/high-sensitivity-r1.html
visualizations:
  - name: trajectory-overlay
    kind: multibody_gif
    config: {frame_duration_ms: 100}
conclusion:                                     # null until drawn
  text: |
    Triple sensitivity restored up-gradient bias under shallow gradient.
  drawn_at: "2026-05-13"
parent_studies: []                              # reserved for Phase 2
```

**Differences from today's investigation spec:**

- `schema_version: 3` (was 2).
- New top-level `objective:` (research question; markdown).
- New top-level `baseline: {composite, params}`.
- Drop the multi-composite `composites: [...]` list — a Study is one
  composite.
- `parent_studies: []` reserved for Phase 2 edges; ignored now.
- Everything else (`variants`, `runs`, `visualizations`, `conclusion`,
  sidecar `composites/`, `viz/`, `notes.md`, `runs.db`) carries over.

**runs.db schema:** unchanged from today's investigation `runs.db`.

### 5.2. Composite default runtime metadata

**Decorator change** in `pbg_superpowers.composite_generator`:

```python
@composite_generator(
    name="chemotaxis",
    description="...",
    parameters={...},
    default_n_steps=200,      # new optional kwarg
)
def chemotaxis(core=None, ...) -> dict: ...
```

`GeneratorEntry` dataclass gets `default_n_steps: int | None = None` as a
new field. `None` means "no opinion, fall back to UI default".

**Discovery propagation.** `discover_all_composites()` in
`vivarium_dashboard/lib/composite_lookup.py` already wraps each
generator entry into the catalog dict; add `default_n_steps` to the
emitted entry alongside `id`, `kind`, `description`, `module`.

**Dashboard surface.** `/api/composites` returns one entry per composite;
include `default_n_steps` in each entry's JSON.

**Composite Explorer pre-fill** (`vivarium_dashboard/static/walkthrough.js`):
on composite selection, read `composite.default_n_steps`; if set, write it
into the `#ce-steps` input. User can still edit before clicking Test Run.
If absent, keep today's hardcoded fallback of `5`.

**viva-munk wiring.** Phase 1 also sets sensible per-generator defaults.
Source-of-truth: `viva_munk/experiments/registry.py` already carries
curated `n_steps` per experiment; the implementation plan mechanically
applies those to each `@composite_generator(...)` in
`viva_munk/composites/__init__.py`.

**Why a separate field, not piggybacking on `parameters: {...}`?**
`parameters` declares *composite-builder kwargs*; `n_steps` is a *runtime
knob* the framework (not the composite) owns. Other future runtime metadata
(`default_interval`, `default_frame_duration_ms`) belongs alongside
`default_n_steps`, not mixed into composite parameters.

### 5.3. Save-as-Study flow

**Affordance.** A **Save as Study** button appears under the latest run's
viz in the Composite Explorer, but only after a successful Test Run
(hidden during a run and after a failed run).

**Flow:**

1. User clicks **Save as Study** → modal with three fields:
   - `Study name` (slug, lowercase + dashes; pre-filled
     `<composite-name>-<YYMMDD>`).
   - `Objective` (multiline markdown; the research question).
   - `Description` (optional, short one-liner).
2. Submit → `POST /api/study-create-from-run` with the just-completed
   `run_id` plus the modal fields.
3. Server creates `studies/<slug>/`, copies the source run into it, returns
   the new Study URL.
4. Dashboard navigates to `/studies/<slug>` (the Study Detail view).

**New endpoint:** `POST /api/study-create-from-run`

Request:

```json
{
  "name":         "chemotaxis-gradient-sweep",
  "objective":   "Does sensitivity restore up-gradient bias?",
  "description": "",
  "source_run_id": "viva_munk.composites.chemotaxis__1715620800__a3f2c1"
}
```

Server-side handler (`vivarium_dashboard/server.py`):

1. **Validate** — slug shape (regex); refuse if `studies/<name>/` exists
   (409); refuse if `source_run_id` not in `.pbg/composite-runs.db` (404).
2. **Load source run** — read `runs_meta` row + all `history` rows for
   `simulation_id == source_run_id` from `.pbg/composite-runs.db`.
3. **Resolve baseline** — `spec_id` from the runs_meta row → `baseline.composite`;
   `params_json` → `baseline.params`.
4. **Create directory** — `studies/<name>/` with `composites/`, `viz/`.
5. **Write `study.yaml`** — populated from modal fields + baseline + a
   single-element `runs` list referencing the moved run.
   `schema_version: 3`, `status: ran`, `parent_studies: []`,
   `variants: []`, `visualizations: []`, `conclusion: null`.
6. **Open `studies/<name>/runs.db`** — bootstrap the same schema as the
   scratch DB; `INSERT INTO runs_meta SELECT ...`; `INSERT INTO history
   SELECT ...` for that one `simulation_id`. The destination DB now owns
   the run.
7. **Materialize viz** — if the run's `viz_html` store had content, write
   it to `studies/<name>/viz/<run_id>.html`. For animated-GIF data URIs,
   also extract and save the raw `.gif`.
8. **Leave the scratchpad alone** — `.pbg/composite-runs.db` keeps the
   row. The next Test Run will replace it as today.

Response:

```json
{
  "study": "chemotaxis-gradient-sweep",
  "url":   "/studies/chemotaxis-gradient-sweep"
}
```

**Atomicity.** All writes (study.yaml, runs.db, viz files) happen inside
a temp directory; success = atomic rename to `studies/<name>/`. Any
failure leaves the workspace unchanged.

**Error handling:**

- *Slug collision* → 409 + inline modal error; user adjusts name.
- *Source run missing* (e.g., another tab ran a new test between
  modal-open and submit) → 404 + modal banner "Source run no longer in
  scratchpad; re-run before saving."
- *Write failure* → 500 + OS error message; temp dir is cleaned up; no
  partial state in `studies/`.
- *Empty objective* — allowed; can be filled in from the Study Detail
  view later.

**Scope guards (Phase 1):**

- No "Add this run to an existing Study" affordance from Composite
  Explorer. Adding runs to an existing Study happens inside the Study
  Detail view via its **Run baseline / Run variant** buttons.
- No bulk-promote of multiple test runs (the scratchpad only holds one
  at a time).
- No auto-suggest of slug from objective; the pre-fill is just
  `<composite-name>-<YYMMDD>`.

### 5.4. Study Detail view

**URL:** `/studies/<name>`. Replaces today's Investigation Detail page.

**Layout — six stacked cards.** Markdown-style ASCII mockup:

```
┌─────────────────────────────────────────────────────────────────┐
│ [name]                  [status badge]  [last-updated]   [⋯]   │
│ <slug>                  · 3 runs · 1 variant · ran              │
└─────────────────────────────────────────────────────────────────┘
┌── Objective ───────────────────────────────────────────────────┐
│ <editable markdown — the research question>                     │
└─────────────────────────────────────────────────────────────────┘
┌── Baseline ────────────────────────────────────────────────────┐
│ Composite: viva_munk.composites.chemotaxis                     │
│ Params table (read-only): n_cells: 12, sensitivity: 2.0, ...    │
│ [Run baseline]   [Edit params]                                  │
└─────────────────────────────────────────────────────────────────┘
┌── Variants ────────────────────────────────────────────────────┐
│ high-sensitivity  · "Triple sensitivity"                        │
│   parameter_overrides: {sensitivity: 6.0}                       │
│   [Run variant] [Edit] [Delete]                                 │
│ [+ Add variant]                                                 │
└─────────────────────────────────────────────────────────────────┘
┌── Runs (3) ────────────────────────────────────────────────────┐
│ ☐  Variant      Label     Steps  Status   Viz   Actions         │
│ ☐  baseline     default   200    ✓ ran    🎞    View ⋯          │
│ ☐  high-sens    seed=42   200    ✓ ran    🎞    View ⋯          │
│ ☐  high-sens    seed=7    200    ⚠ failed —     Tail ⋯          │
│ [Compare selected]   [Clear all runs]                           │
└─────────────────────────────────────────────────────────────────┘
┌── Visualizations ──────────────────────────────────────────────┐
│ trajectory-overlay · multibody_gif · used by 3 runs             │
│ [+ Add visualization]                                           │
└─────────────────────────────────────────────────────────────────┘
┌── Conclusion ──────────────────────────────────────────────────┐
│ <editable markdown — blank until drawn>                         │
│ [Mark complete]  (transitions status → complete)                │
└─────────────────────────────────────────────────────────────────┘
```

**Card-by-card behavior + endpoint mapping (Phase 1 reuses existing
investigation handlers; URL rename or alias TBD in plan):**

| Card           | Affordance         | Endpoint (today)                              | Phase 1 alias                 |
|----------------|--------------------|-----------------------------------------------|-------------------------------|
| Header         | Rename             | *new* (renames the directory + updates `name:`)   | `/api/study-rename`           |
| Header         | Delete Study       | `_post_investigation_delete`                   | `/api/study-delete`           |
| Header         | Export ZIP         | *new* small handler                            | `/api/study-export`           |
| Objective      | Click-to-edit      | *new* tiny handler                             | `/api/study-set-objective`    |
| Baseline       | Edit params        | (modal only — saves into `baseline.params`)    | `/api/study-set-baseline-params` |
| Baseline       | Run baseline       | `_post_investigation_run`                      | `/api/study-run-baseline`     |
| Variants       | + Add variant      | `_post_investigation_composite_perturb`        | `/api/study-variant-add`      |
| Variants       | Run variant        | `_post_investigation_run_one`                  | `/api/study-run-variant`      |
| Variants       | Edit / Delete      | `_post_investigation_composite_perturb` (edit) / `_post_investigation_run_delete` (variant cleanup) | `/api/study-variant-update`, `/api/study-variant-delete` |
| Runs           | View trajectory    | existing `_ceLoadState` flow (frontend only)   | unchanged                     |
| Runs           | Compare selected   | `_post_investigation_comparison_add`           | `/api/study-comparison-add`   |
| Runs           | Delete run         | `_post_investigation_run_delete`               | `/api/study-run-delete`       |
| Runs           | Clear all          | `_post_investigation_runs_clear`               | `/api/study-runs-clear`       |
| Visualizations | + Add / render     | `_post_investigation_add_viz` / `_post_investigation_render_viz` | `/api/study-viz-add`, `/api/study-viz-render` |
| Conclusion     | Edit + Mark complete | `_post_investigation_set_conclusions`        | `/api/study-set-conclusion`   |

**New (small) handlers:**

- `POST /api/study-set-objective {study, text}` — writes
  `study.yaml.objective`.
- `POST /api/study-set-baseline-params {study, params}` — replaces
  `study.yaml.baseline.params`.
- `POST /api/study-rename {study, new_name}` — renames the directory
  and updates `name:` field.
- `GET /api/study-export {study}` — streams a zip of the directory.

**Backend reuse, not rewrite.** The investigation handlers stay; either
the URL prefix is renamed across both server and frontend in one change,
or the new prefix is aliased to the existing handlers and the old prefix
is deprecated for one release. Implementation plan picks one (alias is
lower-risk); the spec is agnostic.

**Empty-state UX.** A freshly-promoted Study (from §5.3) lands with one
baseline run, no variants, blank conclusion. Natural next click:
**+ Add variant**. The page never blocks behind tutorial flow.

## 6. Migration

**One-shot script** `scripts/migrate_investigations_to_studies.py`:

1. Walk `investigations/*/spec.yaml`.
2. For each, transform → `studies/<name>/study.yaml`:
   - Bump `schema_version: 2 → 3`.
   - If multi-composite `composites: [...]`: take first as
     `baseline.composite`; warn if >1, write the rest as commented-out
     stub variants the user re-creates manually.
   - If legacy `composite:` string: lift into `baseline.composite` and
     `parameters: {...}` → `baseline.params`.
   - Promote first-of-`simulations` / `runs[0]` overrides into
     `baseline.params`.
   - Preserve `variants`, `runs`, `visualizations`, `conclusion`,
     `groups`, `comparisons`.
   - Add `parent_studies: []` and `objective: ""` (stub).
3. `git mv` sidecar dirs (`composites/`, `viz/`, `runs.db`, `notes.md`).
4. After all subdirs are moved, `git mv investigations/ studies/`.

**viva-munk has zero existing investigations**, so this is a no-op for
this workspace. The script primarily serves other workspaces that have
accumulated investigations.

**Back-compat for URLs.** Phase 1 keeps `/api/investigation-*` routes
aliased to the same handlers for one release; the frontend switches
immediately to `/api/study-*`. After one release the aliases are
deleted.

## 7. Deferred (later phases)

**Phase 2: Study edges + planned states.**
- Populate `parent_studies: [{study_id, reason}]` with real edges.
- New `status: planned` for Studies authored before any run.
- "Spawn follow-up Study" button on the Conclusion card → creates a new
  Study with this one as parent.
- Cross-Study comparison endpoints (compare runs across Study
  boundaries).

**Phase 3: Investigations-as-graph view.**
- Investigations tab re-skinned as a DAG of Study cards with edges
  drawn from `parent_studies`.
- Pan, zoom, drag-node, click-node-to-open-detail.
- Subgraph/cluster annotations for named research lines.
- Mostly UI work; the underlying data shape is locked in Phase 2.

**Composite runtime metadata growth.** `default_n_steps: int` is Phase 1;
non-breakingly grow into `default_runtime: {interval, n_steps,
total_simulated_time, default_frame_duration_ms}` if/when a future
generator needs it.

## 8. Acceptance criteria

A workspace developer can:

1. **Pre-fill default runtime.** Open the Composite Explorer for any
   composite with `default_n_steps` set; the steps input is pre-filled
   with that value. Override and run normally.
2. **Save a Test Run as a Study.** After a successful Test Run, click
   **Save as Study**, fill in name + objective in the modal, submit, and
   land on the new Study's Detail page. The run's history is now in
   `studies/<name>/runs.db`; the source row in `.pbg/composite-runs.db`
   is untouched.
3. **Add a variant to a Study.** From the Study Detail view, click **+ Add
   variant**, specify name + parameter_overrides (and optionally
   process_overrides), click **Run variant**, see the new run appear in
   the Runs card with its viz.
4. **Compare two runs.** Select ≥2 rows in the Runs card, click
   **Compare selected**, see the comparison view (existing UI, just
   rebound).
5. **Draw a conclusion.** Edit the Conclusion card's markdown, click
   **Mark complete**; the Study's status badge flips to `complete`.

A workspace developer with existing Investigations can:

6. **Migrate.** Run `scripts/migrate_investigations_to_studies.py`; all
   existing `investigations/*/` become `studies/*/`, multi-composite
   Investigations are split with a warning, and the dashboard now lists
   them as Studies.

## 9. Open implementation questions (deferred to plan)

- **URL aliasing vs full rename.** Plan picks one; spec is agnostic.
- **Param-edit modal UX details.** Type-aware widgets vs. plain inputs;
  validation timing.
- **Process-override recipe UX.** The Add-variant wizard for
  `process_overrides` needs a tiny path-picker UI — not specified here.
- **Viz config defaults for newly-created Studies.** Phase 1 seeds with
  the composite's `MultibodyVizStep` defaults; whether the Visualizations
  card pre-populates that or starts empty is a UX call.
- **`studies/<name>/notes.md` editing surface.** Spec doesn't specify
  whether the Notes card is a separate area or folded into another card.
