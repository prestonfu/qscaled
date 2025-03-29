"""
This script is used to compute proposed hyperparameters from our grid search
results on OpenAI Gym. If you'd like to use this script on your own data,
check:
* README for instructions and
* `ExampleOneSeedPerRunCollector` from `qscaled/wandb_utils/one_seed_per_run.py`
* `SweepConfig` from `qscaled/utils/configs.py`
* `compute_params` from `qscaled/scripts/compute_params.py`
for further details.
"""

import numpy as np

np.random.seed(42)

from qscaled.scripts.compute_params import compute_params
from qscaled.wandb_utils.one_seed_per_run import ExampleOneSeedPerRunCollector
from qscaled.utils.configs import SweepConfig

wandb_collect = False

if wandb_collect:
    wandb_collector = ExampleOneSeedPerRunCollector(
        'username', 'crl', wandb_tags=['sac_grid_manual_250206']
    )
    # Remove these lines if you'd like; some of our runs crashed
    wandb_collector.remove_short(0.95)
    wandb_collector.trim(num_seeds=8, compare_metric='episode/return', verbose=True)
else:
    wandb_collector = None

# Maximum possible returns (estimated with infinite data and compute) on each
# environment. These are mostly eyeballed such that runs reach 80% (hence
# the 1.25 multiplier) but not 90%. There is some variation for different environments.

config = SweepConfig(
    name='gym_sweep',  # Zip filename
    max_returns={
        'HalfCheetah-v4': 7300 * 1.25,
        'Walker2d-v4': 4000 * 1.25,
        'Ant-v4': 5300 * 1.25,
        'Humanoid-v4': 5200 * 1.25,
    },
    returns_key='episode/return',
    utds_to_predict=[0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    wandb_collector=wandb_collector,
    baseline_utd_at=2,
)

compute_params(config, '../outputs')
