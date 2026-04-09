# multi-cell

Multi-cell simulations with 2D physics, built on [process-bigraph](https://github.com/vivarium-collective/process-bigraph) and [pymunk](https://www.pymunk.org/).

### **[Demos](https://vivarium-collective.github.io/Viva-munk/)**

## Goal

Provide a composable framework for simulating populations of growing, dividing cells with realistic 2D physics and shared chemical environments. Cells are modeled as capsule-shaped rigid bodies (pymunk segments) — or as multi-segment compound bodies that can flex and bend — that grow, divide, secrete particles, attach to surfaces, and interact physically through collisions and confinement. Concentration fields are modeled with a finite-difference diffusion process and coupled to cells through a per-tick exchange step. The framework uses the bigraph process architecture, making it easy to compose new cell behaviors, environmental structures, and analysis pipelines.

Default parameters are calibrated to *E. coli* proportions (~1 μm wide, ~2 μm at birth, ~4 μm at division, ~30–40 min doubling time).

## Quick Start

```bash
git clone https://github.com/vivarium-collective/Viva-munk.git
cd Viva-munk
pip install -e .

# Run all experiments and open the HTML report
python -m multi_cell.experiments.test_suite

# Run without auto-opening the browser
python -m multi_cell.experiments.test_suite --no-open
```

## Architecture

```
multi_cell/
  pymunk_agent_type.py       # Custom PymunkAgent type with optimized dispatch
  types/
    positive.py              # PositiveFloat / PositiveArray / Concentration / SetFloat
  processes/
    multibody.py             # PymunkProcess — 2D physics (rigid + bending cells)
    grow_divide.py           # GrowDivide — exponential growth, division, mutation
    pressure.py              # Pressure — vectorized per-cell mechanical pressure
    diffusion_advection.py   # DiffusionAdvection — 2D field diffusion + advection
    cell_field_exchange.py   # CellFieldExchange — cell ↔ field uptake/secretion
    secrete_eps.py           # SecreteEPS — EPS particle secretion
    remove_crossing.py       # RemoveCrossing — remove cells past a boundary
  plots/
    multibody_plots.py       # GIF rendering with phylogeny / pressure coloring
                             # and concentration-field heatmap overlays
  experiments/
    test_suite.py            # Experiment registry, runner, HTML report generator
```

### Processes

- **PymunkProcess** (Process): Manages the pymunk 2D physics space. Handles three kinds of agents — rigid segment cells, multi-segment bending cells (compound bodies linked by pivot joints + damped rotary springs), and circular particles. Syncs state to bodies, steps the physics with sub-stepping, applies damping/jitter, optionally enforces adhesion to a wall, and emits position/velocity/polyline deltas.
- **GrowDivide** (Process): Per-cell exponential mass growth. Optionally gated by a Monod factor on a local nutrient (`local[nutrient_key]`) and inhibited by mechanical pressure (`exp(-pressure / pressure_k)`). When mass exceeds a threshold, the cell divides into two daughters along its axis; daughters can inherit mutated parameters and (for bending cells) a fresh straight polyline so they don't inherit the mother's pose.
- **Pressure** (Step): Computes a per-cell mechanical pressure proxy from cell-cell and cell-wall overlap depths (vectorized via numpy). Written back to each cell's `pressure` field for downstream consumers.
- **DiffusionAdvection** (Process): Explicit FTCS diffusion + upwind advection on a 2D `map[mol_id → array]` field, with ghost-layer boundary conditions (periodic / neumann / dirichlet / dirichlet_ghost).
- **CellFieldExchange** (Process): Couples `pymunk_agent` cells to a 2D field map. Each tick it samples each cell's local field concentration into `cell.local`, and applies each cell's `cell.exchange` amounts back into the corresponding field bin (Δconcentration = Δamount / bin_volume).
- **SecreteEPS** (Process): Per-cell EPS particle secretion at a rate proportional to cell mass. Particles are placed on the cell surface and added to the environment. Optionally gated to attached cells only.
- **RemoveCrossing** (Step): Removes any cell whose position exceeds a configurable x/y boundary. Used to model the flow channel in mother-machine experiments.

### Custom types

`pymunk_agent` is a `Node`-subclass type with hand-optimized `apply`/`reconcile`/`realize`/`check` dispatch (no per-field plum dispatch on the hot path). It carries the geometric and physical state of one cell (mass, radius, length, location, velocity, …) plus optional fields used by downstream processes (`polyline`, `attached`, `pressure`, `local`, `exchange`).

The `multi_cell.types.positive` module provides minimal `PositiveFloat`, `PositiveArray`, `Concentration`, and `SetFloat` types for the field state, accumulator-with-clamp semantics adapted from [spatio-flux](https://github.com/vivarium-collective/spatio-flux).

## Experiments

The current registry (`multi_cell/experiments/test_suite.py`):

- **daughter_machine** — single cell grows in an open chamber with an absorbing right wall.
- **mother_machine** — narrow dead-end channels (~1.5 μm wide); cells grow vertically and are removed when they reach the flow channel.
- **with_particles** — cells grow in a chamber seeded with passive particles; cells push and rearrange them.
- **attachment** — adhesin-bearing cells attach to the bottom surface via PivotJoints; adhesins split between daughters at division.
- **glucose_growth** — cells on a 2D glucose field. `DiffusionAdvection` spreads glucose, `CellFieldExchange` applies cell consumption every tick, and `GrowDivide` gates rate by Monod kinetics. Cells stop dividing once their local patch is depleted.
- **bending_pressure** — multi-segment bending capsules grow into a colony. `Pressure` computes mechanical pressure from neighbor / wall contacts and `GrowDivide` applies `exp(-pressure / pressure_k)` inhibition. Cells visibly bend AND slow.

`test_suite.py` runs each one, captures a GIF, a bigraph composition viz, and a serialized state JSON, and generates an HTML report (`out/report.html`) with timing, cell counts, descriptions, and embedded media.

## Dependencies

- [process-bigraph](https://github.com/vivarium-collective/process-bigraph) — bigraph process framework
- [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) — typed schemas
- [bigraph-viz](https://github.com/vivarium-collective/bigraph-viz) — bigraph visualization
- [pymunk](https://www.pymunk.org/) — 2D physics engine
- numpy, scipy, matplotlib, pillow
