"""
This script is used to compute proposed hyperparameters from our grid search
results on OpenAI gym. If you'd like to use this script on your own data,
check:
* README for instructions and 
* `ExampleOneSeedPerRunCollector` from `qscaled/wandb_utils/one_seed_per_run.py`
* `SweepConfig` from `qscaled/utils/configs.py`
* `compute_params` from `qscaled/core/grid_search/bootstrap_envsteps_to_thresh.py`
for further details.
"""

from compute_params import compute_params
from qscaled.wandb_utils.one_seed_per_run import ExampleOneSeedPerRunCollector
from qscaled.utils.configs import SweepConfig

wandb_collect = False

if wandb_collect:
    wandb_collector = ExampleOneSeedPerRunCollector('prestonfu', 'crl', wandb_tags=['sac_grid_manual_250206'])
    # Remove these lines if you'd like; some of our runs crashed
    wandb_collector.remove_short(0.95)
    wandb_collector.trim(num_seeds=8, compare_metric='episode/return', verbose=True)
else:
    wandb_collector = None

config = SweepConfig(
    name='gym_sweep',
    max_returns={
        'HalfCheetah-v4': 7300 * 1.25,
        'Walker2d-v4': 4000 * 1.25,
        'Ant-v4': 5300 * 1.25,
        'Humanoid-v4': 5200 * 1.25,
    },
    returns_key='episode/return',
    utds_to_predict=[0.25, 0.5, 1, 2, 4, 8, 16],
    wandb_collector=wandb_collector,
)

compute_params(config)
