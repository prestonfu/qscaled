"""
This script is used to compute proposed hyperparameters from our grid search
results on the Deepmind Control Suite. If you'd like to use this script on your
own data, check:
* README for instructions and
* `ExampleOneSeedPerRunCollector` from `qscaled/wandb_utils/one_seed_per_run.py`
* `SweepConfig` from `qscaled/utils/configs.py`
* `compute_params` from `qscaled/scripts/compute_params.py`
for further details.
"""

import numpy as np

np.random.seed(42)

from qscaled.scripts.compute_params import compute_params
from qscaled.utils.configs import SweepConfig

config = SweepConfig(
    name='dmc_sweep',  # Zip filename
    max_returns={},  # No need to normalize returns; DMC is already 0-1000.
    returns_key='online_returns',
    utds_to_predict=[0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    wandb_collector=None,
    baseline_utd_at=2,
)

compute_params(config, '../outputs')
