"""Wait for the active declared batch, then validate/reuse it and run new fits.

A batch is considered alive while its status file, per-epoch resume checkpoint,
or log file keeps changing. If nothing changes for ``--idle-timeout`` seconds
the batch is declared dead instead of polling forever, and the operator is
told how to recover.
"""
from pathlib import Path
import argparse, json, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[2]
status_path = ROOT / 'results/training_queue_status.json'


def newest_activity(status: dict) -> float:
    tag = status.get('running')
    paths = [status_path]
    if tag:
        paths.append(ROOT / f'results/checkpoints/sensitivity/{tag}_resume.pt')
        paths.append(ROOT / f'results/logs/{tag}.log')
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


def wait_for_batch(idle_timeout: float) -> None:
    last_change = time.time()
    previous = None
    while True:
        raw = status_path.read_text()
        if raw != previous:
            previous, last_change = raw, time.time()
        status = json.loads(raw)
        if status.get('failed'):
            raise RuntimeError(status)
        if status.get('completed'):
            return
        activity = newest_activity(status)
        if activity > last_change:
            last_change = activity
        if time.time() - last_change > idle_timeout:
            raise RuntimeError(
                f'No batch activity for {idle_timeout:.0f}s (status={status}). '
                'The batch looks dead: inspect results/logs/, then delete or '
                'repair results/training_queue_status.json and rerun.')
        time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--idle-timeout', type=float, default=1800.0,
                        help='seconds without batch activity before declaring the batch dead')
    args = parser.parse_args()
    if status_path.exists():
        try:
            wait_for_batch(args.idle_timeout)
        except FileNotFoundError:
            print('Status file disappeared; starting the queue directly', flush=True)
            raise SystemExit(subprocess.call(
                [sys.executable, '-u', str(ROOT / 'experiments/train_sensitivity/run_queue.py')], cwd=ROOT))
        print('Previous batch finished; continuing extended protocol', flush=True)
    else:
        print('No active batch status found; starting the queue directly', flush=True)
    raise SystemExit(subprocess.call(
        [sys.executable, '-u', str(ROOT / 'experiments/train_sensitivity/run_queue.py')], cwd=ROOT))


if __name__ == '__main__':
    main()
