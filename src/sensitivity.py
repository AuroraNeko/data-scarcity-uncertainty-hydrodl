"""Declared endpoint-sensitivity training protocol.

The 48 predeclared fits are the single source of truth shared by the
training queue (stage 5), the trainer itself, and the analysis (stage 7);
ids, windows, seeds and budgets must stay identical wherever they are
consumed.
"""

QUEUE_PRIORITY = {
    'validation_': 0,
    'temporal_': 1,
    'multilevel_': 2,
    'tcn_': 3,
    'memory': 4,
}


def queue_position(spec: dict) -> int:
    """Execution rank of a spec; every declared id must match a prefix."""
    for prefix, rank in QUEUE_PRIORITY.items():
        if spec['id'].startswith(prefix):
            return rank
    raise ValueError(f'no queue priority declared for spec id {spec["id"]}')


def specs():
    rows=[]
    for year in [1985,1990,1991,1992,1993,1994]:
        for seed in [42,123,456]:
            rows.append(dict(id=f'temporal_wy{year}_s{seed}',train_start=f'{year-1}-10-01',train_end=f'{year}-09-30',
                             seed=seed,head='3tau',architecture='lpu',validation=False,budget='epochs'))
    for years in [1,15]:
        for seed in [42,123,456]:
            common=dict(train_start='1994-10-01' if years==1 else '1980-10-01',train_end='1995-09-30',seed=seed)
            rows.append(dict(id=f'validation_{years}yr_s{seed}',**common,head='3tau',architecture='lpu',validation=True,budget='epochs'))
            if years==15:
                rows.append(dict(id=f'validation_equal_{years}yr_s{seed}',**common,head='3tau',architecture='lpu',validation=True,budget='equal_updates'))
            rows.append(dict(id=f'multilevel_{years}yr_s{seed}',**common,head='13tau',architecture='lpu',validation=False,budget='epochs'))
            rows.append(dict(id=f'tcn_{years}yr_s{seed}',**common,head='3tau',architecture='tcn',validation=False,budget='epochs'))
    for seq_len in [180,365]:
        for seed in [42,123,456]:
            rows.append(dict(id=f'memory{seq_len}_15yr_s{seed}',train_start='1980-12-31',train_end='1995-09-30',
                             seed=seed,head='3tau',architecture='lpu',validation=False,budget='epochs',seq_len=seq_len))
            if seq_len==365:
                rows.append(dict(id=f'memory365_1yr_s{seed}',train_start='1994-10-01',train_end='1995-09-30',
                                 seed=seed,head='3tau',architecture='lpu',validation=False,budget='epochs',seq_len=365))
    return rows
