"""Bind primary and endpoint training reuse to the processed inputs actually read."""
import csv

from src.data.dataset import DYNAMIC_VARS, STATIC_COLS, primary_basin_ids
from src.sensitivity_identity import canonical_sha
from src.utils import sha256_file


def processed_source_hashes(root, basins):
    if len(basins) != len(set(basins)):
        raise ValueError('Duplicate training basins')
    sources = {}
    for basin in basins:
        path = root / 'data/processed/camels_us' / f'{basin}.csv'
        sources[basin] = sha256_file(path)
    return sources


def input_identity(root, record):
    basins = record['selected_basins']
    sources = processed_source_hashes(root, basins)
    protocol = {k: record.get(k) for k in [
        'protocol_version', 'years', 'model_seed', 'basin_seed', 'subset_id',
        'architecture', 'seq_len', 'batch_size', 'epochs', 'training_budget',
        'steps_per_round', 'train_target_period', 'calibration_target_period',
        'test_target_period', 'checkpoint_selection', 'normalization_fit_period',
    ]}
    identity = {'schema': 1, 'selected_basins': basins, 'source_sha256': sources,
                'dynamic_columns': DYNAMIC_VARS, 'static_columns': STATIC_COLS,
                'protocol': protocol}
    return {**identity, 'sha256': canonical_sha(identity)}


def require_current_inputs(root, record, subset_id=None):
    if subset_id is None:
        # Resolve from this root, including fresh extracted checkouts.
        metadata = root / 'data/metadata/basin_metadata.csv'
        with metadata.open(encoding='utf-8') as stream:
            available = [row['basin_id'] for row in csv.DictReader(stream)]
        expected = primary_basin_ids(available)
    else:
        path = root / 'results/tables/scarcity_replications/replication_subsets.csv'
        with path.open(encoding='utf-8') as stream:
            rows = [row for row in csv.DictReader(stream) if row['subset_id'] == subset_id]
        if rows and all(row.get('selection_order') for row in rows):
            rows.sort(key=lambda row: int(row['selection_order']))
        expected = [row['basin_id'] for row in rows]
    if record['selected_basins'] != expected:
        raise ValueError('Current basin selection differs from fitted selection')
    if record.get('input_identity') != input_identity(root, record):
        raise ValueError('Training inputs or column definitions changed or were not audited')
