"""Command-line entry point for the multi-cell experiment test suite.

Usage:
    python -m multi_cell.experiments.test_suite                # all experiments, parallel
    python -m multi_cell.experiments.test_suite --serial       # one at a time
    python -m multi_cell.experiments.test_suite --tests glucose_growth attachment
"""
import argparse
import concurrent.futures as _futures
import multiprocessing as _mp
import os
import time

from multi_cell.experiments.registry import EXPERIMENT_REGISTRY
from multi_cell.experiments.runner import run_experiment
from multi_cell.experiments.report import generate_html_report


def _run_one_in_subprocess(name_and_output):
    """Worker entry point: run a single experiment in a fresh process and
    return its result dict. Each worker re-imports the module so it gets its
    own PymunkProcess core (no shared mutable state across workers)."""
    name, output_dir = name_and_output
    # Force matplotlib non-interactive backend in subprocesses.
    import matplotlib
    matplotlib.use('Agg', force=True)
    return run_experiment(name, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description='multi-cell test suite')
    parser.add_argument(
        '--tests', nargs='+',
        default=list(EXPERIMENT_REGISTRY.keys()),
        choices=list(EXPERIMENT_REGISTRY.keys()),
        help='Which experiments to run',
    )
    parser.add_argument('--output', default='out', help='Output directory')
    parser.add_argument('--open', action='store_true', default=True, help='Open report in browser')
    parser.add_argument('--no-open', dest='open', action='store_false', help='Do not open report in browser (for CI)')
    parser.add_argument(
        '--serial', dest='parallel', action='store_false', default=True,
        help='Run experiments sequentially in this process (default: parallel)',
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='Number of parallel workers (0 = min(n_tests, cpu_count))',
    )
    args = parser.parse_args()

    t_total = time.time()

    if args.parallel and len(args.tests) > 1:
        n_workers = args.workers or min(len(args.tests), _mp.cpu_count() or 1)
        print(f'Running {len(args.tests)} experiments across {n_workers} workers...', flush=True)
        all_results = [None] * len(args.tests)
        # Preserve registry order in the report regardless of completion order.
        order = {name: i for i, name in enumerate(args.tests)}
        ctx = _mp.get_context('spawn')
        with _futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futures = {
                ex.submit(_run_one_in_subprocess, (name, args.output)): name
                for name in args.tests
            }
            for fut in _futures.as_completed(futures):
                name = futures[fut]
                try:
                    res = fut.result()
                    all_results[order[name]] = res
                    print(f'  ✓ {name} ({res.get("elapsed", 0):.1f}s, {res.get("n_cells", 0)} cells)', flush=True)
                except Exception as e:
                    print(f'  ✗ {name} FAILED: {e}', flush=True)
        all_results = [r for r in all_results if r is not None]
    else:
        all_results = []
        for name in args.tests:
            all_results.append(run_experiment(name, output_dir=args.output))

    html_path = generate_html_report(all_results, output_dir=args.output)
    print(f'\nTotal wall time: {time.time() - t_total:.1f}s')

    if args.open:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(html_path)}')


if __name__ == '__main__':
    main()
