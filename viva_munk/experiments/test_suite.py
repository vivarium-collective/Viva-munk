"""Backwards-compat shim for the multi-cell experiment test suite.

The implementation now lives in:

    viva_munk.experiments.documents.*   — per-experiment document builders
    viva_munk.experiments.registry      — EXPERIMENT_REGISTRY
    viva_munk.experiments.runner        — run_experiment + helpers
    viva_munk.experiments.report        — generate_html_report
    viva_munk.experiments.cli           — main(), CLI argument parsing

This module re-exports the public surface so existing imports keep working
and the CLI entry point ``python -m viva_munk.experiments.test_suite``
still launches the runner.
"""
from viva_munk.experiments.documents import (
    daughter_machine_document,
    attachment_document,
    glucose_growth_document,
    bending_pressure_document,
    mother_machine_document,
    biofilm_document,
    chemotaxis_document,
    inclusion_bodies_document,
    quorum_sensing_document,
)
from viva_munk.experiments.registry import EXPERIMENT_REGISTRY
from viva_munk.experiments.runner import (
    run_experiment,
    PYMUNK_CORE,
    _splice_process_configs,
)
from viva_munk.experiments.report import (
    generate_html_report,
    _gather_metadata,
    _json_viewer_js,
)
from viva_munk.experiments.cli import main, _run_one_in_subprocess

__all__ = [
    'EXPERIMENT_REGISTRY',
    'PYMUNK_CORE',
    'run_experiment',
    'generate_html_report',
    'main',
    'daughter_machine_document',
    'attachment_document',
    'glucose_growth_document',
    'bending_pressure_document',
    'mother_machine_document',
    'biofilm_document',
    'chemotaxis_document',
    'inclusion_bodies_document',
    'quorum_sensing_document',
]


if __name__ == '__main__':
    main()
