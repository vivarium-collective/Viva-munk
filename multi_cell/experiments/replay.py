"""Replay a simulation from the history database to regenerate its GIF.

Every run of ``python -m multi_cell.experiments.test_suite`` writes one row
per step into ``<output_dir>/history.db``, plus a ``simulations`` row that
records the experiment name, config, and serialized composite state. This
module uses that information to reconstruct the GIF without re-running the
simulation.

Usage:
    python -m multi_cell.experiments.replay --list
    python -m multi_cell.experiments.replay <simulation_id>
    python -m multi_cell.experiments.replay <simulation_id> --out replays/
"""
import argparse
import os
import sys

from process_bigraph.emitter import (
    list_simulations, load_history, load_simulation_metadata,
)

from multi_cell.experiments.runner import render_gif, DB_FILE


def _resolve_db_path(output_dir):
    return os.path.join(output_dir, DB_FILE)


def list_runs(output_dir='out'):
    db_path = _resolve_db_path(output_dir)
    if not os.path.exists(db_path):
        print(f'no history db at {db_path}')
        return
    rows = list_simulations(db_path)
    if not rows:
        print(f'no simulations recorded in {db_path}')
        return
    print(f'{len(rows)} simulation(s) in {db_path}:\n')
    for r in rows:
        name = r.get('name') or '(no name)'
        started = r.get('started_at') or '?'
        cfg = 'config✓' if r.get('has_config') else 'no-config'
        print(f"  {r['simulation_id']}  {name:24s}  {started}  steps={r['step_count']}  {cfg}")


def replay(simulation_id, output_dir='out', out=None, filename=None):
    db_path = _resolve_db_path(output_dir)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f'no history db at {db_path}')

    meta = load_simulation_metadata(db_path, simulation_id)
    if meta is None:
        raise KeyError(f'no simulations row for {simulation_id} in {db_path}')

    composite_config = meta.get('composite_config') or {}
    run_meta = meta.get('metadata') or {}
    name = meta.get('name') or run_meta.get('experiment_name') or simulation_id
    config = run_meta.get('config') or {}
    env_size = config.get('env_size', 600)

    results = load_history(db_path, simulation_id)
    if not results:
        raise RuntimeError(f'no history rows for {simulation_id}')

    target_dir = out or output_dir
    os.makedirs(target_dir, exist_ok=True)
    filename = filename or f'{name}_replay_{simulation_id[:8]}'

    print(f'replaying {name} ({simulation_id}) — {len(results)} steps')
    gif_path = render_gif(
        filename, results, composite_config, config, target_dir, env_size,
    )
    print(f'GIF: {gif_path}')
    return gif_path


def main():
    parser = argparse.ArgumentParser(
        description='Replay a recorded experiment into a fresh GIF.',
    )
    parser.add_argument('simulation_id', nargs='?',
                        help='ID to replay. Omit with --list to list runs.')
    parser.add_argument('--output', default='out',
                        help='Directory containing history.db (default: out)')
    parser.add_argument('--out', default=None,
                        help='Where to write the replay GIF (default: same as --output)')
    parser.add_argument('--filename', default=None,
                        help='Filename for the GIF (without extension)')
    parser.add_argument('--list', action='store_true',
                        help='List all recorded simulations and exit')
    args = parser.parse_args()

    if args.list or not args.simulation_id:
        list_runs(args.output)
        return 0

    try:
        replay(args.simulation_id, output_dir=args.output,
               out=args.out, filename=args.filename)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
