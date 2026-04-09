# multi-cell

Multi-cell simulations with 2D physics, built on [process-bigraph](https://github.com/vivarium-collective/process-bigraph) and [pymunk](https://www.pymunk.org/).

## Goal

Provide a composable framework for simulating populations of growing, dividing cells with realistic 2D physics. Cells are modeled as capsule-shaped rigid bodies (pymunk segments) that grow, divide, secrete particles, and interact physically through collisions and confinement. The framework uses the bigraph process architecture, making it easy to compose new cell behaviors, environmental structures, and analysis pipelines.

Default parameters are calibrated to *E. coli* proportions (~1 um wide, ~2 um at birth, ~4 um at division, ~40 min doubling time).

## Experiments Report

**[View the experiments report](https://vivarium-collective.github.io/pymunk-process/)** ([source](doc/index.html))

| Experiment | Description |
|------------|-------------|
| **Single Cell Growth** | One cell grows and divides into a colony over 4 hours |
| **Mother Machine** | Cells in narrow dead-end channels with a flow channel that removes expelled cells |
| **Biofilm** | Cells grow, divide, and secrete EPS particles that accumulate into a matrix |

## Quick Start

```bash
git clone https://github.com/vivarium-collective/pymunk-process.git
cd pymunk-process
pip install -e .

# Run all experiments and open the HTML report
python -m multi_cell.experiments.test_suite --output out

# Run a single experiment
python -m multi_cell.experiments.test_suite --tests single_cell_growth --output out
```

## Architecture

```
multi_cell/
  processes/
    multibody.py          # PymunkProcess — 2D physics simulation
    grow_divide.py        # GrowDivide — exponential growth and division
    secrete_eps.py        # SecreteEPS — EPS particle secretion
    remove_crossing.py    # RemoveCrossing — remove cells past a boundary
  plots/
    multibody_plots.py    # GIF rendering with phylogeny coloring
  experiments/
    test_suite.py         # Experiment definitions and HTML report generation
```

### Processes

- **PymunkProcess** (Process): Manages the pymunk 2D physics space. Syncs cell/particle state to rigid bodies, steps the physics with sub-stepping, applies damping and jitter forces, and returns position/velocity deltas.
- **GrowDivide** (Process): Per-cell exponential mass growth. When mass exceeds a threshold, the cell divides into two daughters placed along its axis. Daughters can inherit mutated growth parameters.
- **SecreteEPS** (Process): Per-cell EPS particle secretion at a rate proportional to cell mass. Particles are placed on the cell surface and added to the environment.
- **RemoveCrossing** (Step): Removes any cell whose y-position exceeds a configurable threshold. Used to model the flow channel in mother machine experiments.

## Dependencies

- [process-bigraph](https://github.com/vivarium-collective/process-bigraph) — Bigraph process framework
- [bigraph-viz](https://github.com/vivarium-collective/bigraph-viz) — Bigraph visualization
- [pymunk](https://www.pymunk.org/) — 2D physics engine
