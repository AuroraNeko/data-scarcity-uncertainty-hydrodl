"""Independently recompute representative frozen scores using only delivered arrays."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.protocol import conformal_radius, scores, interval_metrics, observed_groups, integrated_quantile_score
from src.utils import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    tables = ROOT / 'results/tables'
    checks = []
    for years in [1, 15]:
        record = json.loads((tables / f'corrected_scarcity/scarcity_{years}yr_basinseed42_modelseed42.json').read_text())
        for key in ['checkpoint', 'predictions']:
            assert sha256_file(ROOT / record[key]) == record[key + '_sha256']
            checks.append(f'{years}yr_{key}_hash')
        with np.load(ROOT / record['predictions']) as z:
            cm, tm = z['val_mask'] > 0, z['test_mask'] > 0
            cq, q = z['val_prediction_log'][cm].astype(float), z['test_prediction_log'][tm].astype(float)
            cy, y = z['val_target_log'][cm].astype(float), z['test_target_log'][tm].astype(float)
            b = z['test_basin_ids'][tm]
        radius = conformal_radius(scores(cq[:, 0], cq[:, 2], cy))
        lo, hi = q[:, 0] - radius, q[:, 2] + radius
        groups = observed_groups(y, b)
        cells = pd.read_csv(tables / 'coverage_cells.csv')
        for dimension, group, selected in [('pooled', 'all', np.ones(len(y), bool)),
                                            ('observed_flow', 'high', groups == 2)]:
            expected = cells[(cells.year == years) & (cells.seed == 42) &
                (cells.method == 'global') & (cells.dimension == dimension) & (cells.group == group)].iloc[0]
            actual = interval_metrics(lo[selected], hi[selected], y[selected])
            for field in ['picp', 'mpiw', 'winkler', 'n']:
                assert np.isclose(actual[field], expected[field], atol=1e-10, rtol=1e-10), (years, dimension, field)
                checks.append(f'{years}yr_{dimension}_{field}')
    norm = json.loads((ROOT / 'data/metadata/normalization_stats.json').read_text())
    with np.load(ROOT / 'results/predictions/corrected_full_671/test_predictions.npz') as z:
        valid = z['mask'] > 0
        y = z['target'][valid].astype(float) * norm['target_std'] + norm['target_mean']
        q = z['single_qr'][valid].astype(float) * norm['target_std'] + norm['target_mean']
    value = integrated_quantile_score(y, q, [.05, .5, .95]).mean()
    expected = json.loads((tables / 'scoring_summary.json').read_text())['rows'][0]['pooled']
    assert abs(value - expected) < 1e-12
    checks.append('full_basin_common_grid_iqs')
    report = {'passed': True, 'checks': checks, 'n_checks': len(checks), 'single_qr_pooled_iqs': value,
              'scope': 'Frozen primary endpoints, masked within-basin high group, and full-basin IQS; no raw-data dependency'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
