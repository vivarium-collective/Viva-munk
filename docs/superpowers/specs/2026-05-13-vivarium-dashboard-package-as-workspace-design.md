---
title: Package-as-workspace mode for vivarium-dashboard
date: 2026-05-13
status: draft
author: Eran Agmon
brainstorm_session: viva-munk (sqlemitter branch)
---

# Package-as-workspace mode for vivarium-dashboard

> **Note on location.** This design was drafted from inside `viva-munk` because
> that's where the question surfaced ("can we open this package in the
> dashboard?"). The bulk of the implementation work lives in
> `vivarium-dashboard` (with smaller changes in `pbg-template` and
> `pbg-superpowers`); the spec should travel to whichever repo owns the
> implementation plan once we land on one. See §8.

## 1. Context

Today, opening a pbg-package in the vivarium dashboard requires scaffolding a
*separate workspace directory* via `/pbg-workspace`, installing the package
into the workspace's venv, and wiring the workspace's `workspace.yaml` to
reference it. That is the right model for the **multi-package composition
case** (e.g., wiring `pbg-smoldyn` + `pbg-cobra` + `viva-munk` together), but
it is heavy ceremony for the overwhelmingly common **single-package case**
("I have a pbg-package, I just want to open it in the dashboard").

Three observations make a lighter mode viable:

1. `vivarium-dashboard` is already extracted as a standalone pip package
   (since pbg-template v0.5.0). The dashboard server, templates, assets, and
   lib helpers all live inside it — the workspace tree just needs
   `workspace.yaml` and (optionally) some research-process state.
2. Composite discovery is already cross-package. `discover_all_composites()`
   in `vivarium_dashboard/lib/composite_lookup.py` scans every installed
   `pbg-*` distribution for `<package>/composites/*.composite.{yaml,yml,json}`
   and merges `@composite_generator`-decorated functions via
   `pbg_superpowers.composite_discovery.discover_all`. Once a package is pip-
   installed, its composites are visible to any dashboard — no workspace.yaml
   entry required.
3. The schema for `workspace.yaml` (`.pbg/schemas/workspace.schema.json`)
   already supports `package_path` as an optional field — the
   workspace-points-at-an-external-package case was half-imagined when the
   schema was authored.

The remaining gate is `cli.py`'s hard requirement that `workspace.yaml` exist
at cwd before serving. Lift that gate and let the dashboard synthesise a
minimal manifest, and the package itself becomes a serveable workspace.

## 2. Goals

- Inside any pbg-package, `vivarium-dashboard serve` Just Works — one
  command, no scaffolding step, no separate workspace directory.
- Footprint inside the package: one file (`workspace.yaml`) on first run.
  `.pbg/local/` (gitignored) for runtime state. No other top-level dirs
  written until they're needed.
- Today's `/pbg-workspace`-scaffolded workspaces continue working unchanged.
- The multi-package composition flow stays unchanged: still create a sibling
  workspace via `/pbg-workspace`.
- Migration cost for `pbg-template` is reductive (the scaffold gets smaller,
  not larger).

## 3. Non-goals

- Changing the composite discovery convention. It already works.
- Building a new top-level CLI (e.g., `pbg`). The user-facing primitive
  remains `vivarium-dashboard serve`.
- Auto-migrating existing packages' composite layouts. (e.g., viva-munk's
  `viva_munk/experiments/documents/*.py` → `viva_munk/composites/*.yaml` is
  a separate, package-side migration with its own plan.)
- Read-only / no-write-to-disk mode. Persistence (registered imports,
  decisions log, simulation runs) requires writing somewhere; this design
  writes to the package repo (committed) or `.pbg/local/` (gitignored).

## 4. Design overview

`vivarium-dashboard serve` becomes the single entry point. Run with no args
inside a pbg-package:

1. Detect the package via `pyproject.toml` (`auto_init.detect_package_path`).
2. If no `workspace.yaml` exists in cwd, auto-write a minimal one (~6 lines)
   with `package_path` pointing at the detected package directory.
3. Start the server as today.

Composite discovery is unchanged. Persistent state (decisions log, phases,
simulation runs, etc.) lives at the package repo root following the standard
pbg-template layout: research-process state committed, runtime state in
`.pbg/local/` gitignored.

Schemas, templates, and asset helpers move from `pbg-template/.pbg/schemas/`
(copied per workspace) to `vivarium_dashboard/schemas/` and
`vivarium_dashboard/templates/` (pip-installed once, reused by every
workspace). Workspace-local schema copies remain supported for back-compat.

## 5. Component-by-component changes

### 5.1. vivarium-dashboard

**`cli.py`** — relax the workspace.yaml requirement:
- Today (line 22-23): hard error if `workspace.yaml` is absent.
- Change: if absent, call `auto_init.bootstrap(workspace)`, which writes a
  minimal manifest and returns its path. Then proceed normally.
- Add `--package-path=<dir>` flag that overrides auto-detection (passed
  through to `auto_init`).

**New `vivarium_dashboard/auto_init.py`:**
- `detect_package_path(cwd: Path) -> str` — reads `pyproject.toml` and
  resolves the package directory in this priority order:
  1. `[tool.setuptools].packages` → first entry.
  2. `[tool.hatch.build.targets.wheel].packages` → first entry.
  3. Single top-level directory containing `__init__.py` whose name matches
     `re.sub("-", "_", project.name)`.
  4. Fallback `"."` with a warning.
- `synthesise_workspace_yaml(cwd: Path, package_path: str) -> dict` — builds
  the minimal manifest:
  ```yaml
  schema_version: 2
  name: <from [project].name in pyproject.toml, or dir name as fallback>
  created: <today, YYYY-MM-DD>
  plugin_version: <vivarium_dashboard.__version__>
  package_path: <detected or overridden>
  ```
- `bootstrap(cwd: Path, override_package_path: str | None = None) -> Path` —
  detects, synthesises, writes `workspace.yaml`, optionally appends
  `.pbg/local/` to `.gitignore` if a `.gitignore` already exists, returns the
  path.

**`vivarium_dashboard/schemas/`** (new package-data directory):
- Move `template/.pbg/schemas/workspace.schema.json` here.
- Update `lib/workspace_yaml.py` validator to load schema from the pip
  package by default; only check `<workspace>/.pbg/schemas/` if it exists
  (back-compat override).

**Lazy directory creation:**
- Audit every site that today assumes `datasets/`, `references/`,
  `experiments/`, `reports/` exists. Wrap each with
  `Path(dir).mkdir(parents=True, exist_ok=True)` at first write.
- Sites to update (initial sweep, may expand during implementation):
  - `server.py` — anywhere `WORKSPACE / "datasets"` etc. is written to.
  - `lib/imports.py` — install path destination.
  - `lib/composite_runs.py` — where run artefacts land.
  - `lib/pdf_metadata.py` — references PDF storage.

**Templates and assets:**
- `reports/index.html.j2` and `reports/assets/` already live in
  `vivarium_dashboard/templates/`. Confirm no remaining workspace-side
  copies are required.

### 5.2. pbg-template

The scaffold becomes smaller — what used to be copied per workspace now lives
in the dashboard pip package.

**Remove from the scaffold tree:**
- `template/datasets/`
- `template/references/`
- `template/experiments/`
- `template/reports/`
- `template/scripts/` (entire directory; `serve.sh` is replaced by direct
  `vivarium-dashboard serve` invocation, documented in NEXT_STEPS.md.j2)
- `template/.pbg/schemas/`

**Keep:**
- `template/workspace.yaml.j2` (still useful when scaffolding an explicit
  multi-package workspace — `package_path` set to a relative path).
- `template/pyproject.toml.j2`.
- `template/README.md.j2`, `template/NEXT_STEPS.md.j2`.

**Updates:**
- `template-init.sh` simplifies (fewer files to render).
- README rewrites the lead: "The common case is one pbg-package = one
  workspace; run `vivarium-dashboard serve` inside the package. This
  template is for the explicit multi-package composition case."

### 5.3. pbg-superpowers

- `/pbg-workspace` skill: unchanged behavior, but README and skill
  description updated to clarify it's for the multi-package case.
- `pbg_superpowers/package_audit.py`: add one optional check —
  *"package has at least one auto-discoverable composite"* (`warn`, not
  `fail`). Surfaces the "you can run `vivarium-dashboard serve` here"
  affordance during audits.
- No new skill needed. The user-facing command remains
  `vivarium-dashboard serve`.

### 5.4. Consumer packages (e.g., viva-munk)

Out of scope for this design — flagged here only so consumers know what to
expect:

- A package becomes auto-discoverable by the dashboard when its composites
  live under `<package>/composites/` as `*.composite.{yaml,yml,json}` files,
  OR as `@composite_generator`-decorated functions inside the package.
- viva-munk specifically has 8 ad-hoc Python builders in
  `viva_munk/experiments/documents/*.py`. Migrating them is a separate
  plan.

## 6. User flows

### 6.1. Happy path

```
cd ~/code/viva-munk
.venv/bin/pip install -e .            # one-time
.venv/bin/vivarium-dashboard serve
  → no workspace.yaml found; writing one (package_path: viva_munk)
  → serving at http://localhost:8000
```

The dashboard shows the process registry (auto-discovered via
`bigraph-schema`), any composites the package conforms with the convention
for, plus composites from any other pip-installed `pbg-*` packages in the
venv.

### 6.2. Empty package (no composites yet)

`vivarium-dashboard serve` still works; dashboard shows the process registry
and an empty composites list. The empty state is a known surface — no crash.

### 6.3. Today's pbg-template workspace

Unchanged. The "auto-init if missing" branch never fires because
`workspace.yaml` already exists. When the workspace upgrades its
`vivarium-dashboard` dep, schemas/templates served from the pip package
override the (now optional) workspace-local copies; the local
`.pbg/schemas/` directory continues to win if it's still present.

### 6.4. Multi-package composite workspace

Still scaffold via `/pbg-workspace`, install the multiple pbg-packages into
the workspace's venv, serve from the workspace directory. Identical to
today.

## 7. Edge cases

| # | Case | Resolution |
|---|------|------------|
| 1 | No `pyproject.toml` at all | `detect_package_path` falls back to `"."`. Warn that discovery may yield nothing. |
| 2 | Flat-layout pyproject (no `[tool.setuptools].packages`) | Walk top-level dirs; pick the single dir with `__init__.py` whose name matches dist name. Else fall back to `"."` with warning. |
| 3 | Monorepo with multiple packages declared | Pick the one matching dist name; if no match, pick first and warn with `--package-path` override hint. |
| 4 | `workspace.yaml` already exists | Don't clobber. Use as-is; if `schema_version != 2`, run existing `spec_migration` flow. |
| 5 | Auto-detected `package_path` wrong | `vivarium-dashboard serve --package-path=viva_munk` persists into the auto-init'd `workspace.yaml`. Subsequent serves don't re-detect. |
| 6 | Schema drift between dashboard-package and workspace-local | Workspace-local `.pbg/schemas/` wins if present (legacy); else dashboard-package schema. Document in `vivarium_dashboard/schemas/README.md`. |
| 7 | Package not pip-installed in current venv | One-line check at serve startup: warn that in-place composites won't appear until `pip install -e .` runs. Don't block. |
| 8 | No `.gitignore` in package repo | Don't create one. Only append `.pbg/local/` to an existing `.gitignore`. |
| 9 | Dashboard run from a non-package directory (e.g., empty dir) | `detect_package_path` falls back to `"."`, auto-init writes a stub `workspace.yaml`, dashboard serves an empty state. (Useful for exploration.) |
| 10 | Both `[tool.setuptools]` and `[tool.hatch]` declared (mixed build) | Setuptools wins per detection priority; hatchling is a fallback for hatch-only packages. |

## 8. Implementation ownership and rollout

**Repos affected:**
1. `vivarium-dashboard` — 80% of the work (cli.py, auto_init.py, schemas
   relocation, lazy dir creation, validator update).
2. `pbg-template` — 15% of the work (scaffold slimming, README rewrite,
   template-init.sh simplification).
3. `pbg-superpowers` — 5% of the work (audit check, skill copy updates).

**Rollout order:**
1. Land `vivarium-dashboard` changes behind a single release. Version bump
   (minor, e.g. 0.6.0).
2. Update `pbg-template` to depend on the new dashboard version; remove the
   scaffold dirs that now live in the dashboard package; update README.
3. Update `pbg-superpowers` plugin + audit script.
4. Update `viva-munk` to follow the composite convention (separate plan).

**Compatibility window:**
- During the rollout, today's workspaces must keep working with both the old
  and new dashboard versions. Workspace-local `.pbg/schemas/` continues to
  resolve correctly until a workspace explicitly removes it.

## 9. Open questions (out of scope for this spec; deferred to plan)

- Exactly which `server.py` / `lib/*.py` call sites need the lazy-mkdir
  treatment. Initial sweep listed above; full audit during implementation.
- Whether `auto_init` should also seed an `experiments/_runs.yaml` (the
  dashboard reads it for the runs catalog) — punt to first-use lazy creation
  for now.
- Whether the `bypass-selection = true` hatchling trick in
  `pyproject.toml.j2` still applies when the workspace tree is much smaller
  (only `workspace.yaml` + the package itself). Likely still needed; verify.
- How `pbg-superpowers`' `/pbg-workspace` skill text changes — the multi-
  package use case becomes a niche, and the skill description should reflect
  that.

## 10. Acceptance criteria

A pbg-package developer can:

1. Inside their package repo (e.g., `viva-munk`), run
   `vivarium-dashboard serve` with no prior setup beyond `pip install -e .`
   and see their package's processes + composites in the dashboard.
2. Open an existing pbg-template workspace (created before this change) and
   still serve it identically to today.
3. Run `/pbg-workspace my-multi-pkg-research` for the multi-package case and
   wire `pbg-smoldyn` + `pbg-cobra` together as today.

All three flows pass on the same `vivarium-dashboard` version.
