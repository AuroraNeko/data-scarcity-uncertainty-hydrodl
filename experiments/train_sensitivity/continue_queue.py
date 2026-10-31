"""Wait for the active declared batch, then validate/reuse it and run new fits."""
from pathlib import Path
import json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
status_path=ROOT/'results/training_queue_status.json'
if status_path.exists():
    while True:
        status=json.loads(status_path.read_text())
        if status.get('failed'):raise RuntimeError(status)
        if status.get('completed'):break
        time.sleep(3)
    print('Previous batch finished; continuing extended protocol',flush=True)
else:
    print('No active batch status found; starting the queue directly',flush=True)
raise SystemExit(subprocess.call([sys.executable,'-u',str(ROOT/'experiments/train_sensitivity/run_queue.py')],cwd=ROOT))
