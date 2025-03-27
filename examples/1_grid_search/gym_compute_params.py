import os
import numpy as np
import pandas as pd

from qscaled.core.preprocessing import bootstrap_crossings
from qscaled.core.grid_search.bootstrap_envsteps_to_thresh import (
    grid_best_uncertainty_lr,
    grid_best_uncertainty_bs,
    get_bootstrap_optimal,
    compute_bootstrap_averages,
)
from qscaled.core.grid_search.linear_fit import (
    make_linear_fit_separate_slope, 
    make_linear_fit_shared_slope,
    tabulate_proposed_hparams_separate_slope,
    tabulate_proposed_hparams_shared_slope,
    tabulate_baseline_hparams
)
from qscaled.wandb_utils.one_seed_per_run import ExampleOneSeedPerRunCollector
from qscaled.utils.zip_handler import fetch_zip_data
from qscaled.utils.configs import SweepConfig

np.random.seed(42)

def main():
    """
    To use this code on your own data:
    1. Label your Wandb runs with tags.
    2. Fill in `MyCollector` in qscaled/wandb_utils.
    3. Update the information in `SweepConfig` below.

    The latter two steps take ~5 minutes!

    If you set `wandb_collect == True`, your `zip` file will be rebuilt using your
    Wandb collector. Otherwise, the `zip` file must be present.
    """

    wandb_collect = False
    name = "gym_sweep"

    if wandb_collect:
        wandb_collector = ExampleOneSeedPerRunCollector(
            "prestonfu", "crl", wandb_tags=["sac_grid_manual_250206"]
        )
        # Remove these lines if you'd like
        wandb_collector.remove_short(0.95)
        wandb_collector.trim(num_seeds=8, compare_metric="episode/return", verbose=True)
    else:
        wandb_collector = None

    # Maximum possible returns (estimated with infinite data and compute) on each
    # environment. These are mostly eyeballed such that runs reach 80% (hence
    # the 1.25 multiplier) but not 90%. There is some variation for different environments.

    config = SweepConfig(
        name=name,
        max_returns={
            "HalfCheetah-v4": 7300 * 1.25,
            "Walker2d-v4": 4000 * 1.25,
            "Ant-v4": 5300 * 1.25,
            "Humanoid-v4": 5200 * 1.25,
        },
        returns_key="episode/return",
        utds_to_predict=[0.25, 0.5, 1, 2, 4, 8, 16],
        wandb_collector=wandb_collector,
    )
    
    print_hparams = True

    grid_search_df = fetch_zip_data(config, use_cached=True)
    grid_search_df = bootstrap_crossings(grid_search_df, config.thresholds, filename=name)

    # Bootstrapping and Fitting
    best_lr = grid_best_uncertainty_lr(grid_search_df)
    best_bs = grid_best_uncertainty_bs(grid_search_df)
    best_lr_bs = (
        best_lr.groupby(["env_name", "utd"])
        .apply(get_bootstrap_optimal, include_groups=False)
        .reset_index()
    )
    best_lr_bs = compute_bootstrap_averages(best_lr, best_bs, best_lr_bs)

    # Empirically, we find that using a shared slope does better.
    lr_regs_separate, bs_regs_separate = make_linear_fit_separate_slope(best_lr_bs, '../outputs', name)
    lr_reg_shared, bs_reg_shared = make_linear_fit_shared_slope(best_lr_bs, '../outputs', name)
    
    tabulate_kw = dict(df_grid=grid_search_df, utds_to_predict=config.utds_to_predict, outputs_dir='../outputs', save_path=name, verbose=print_hparams)
    tabulate_proposed_hparams_separate_slope(lr_regs=lr_regs_separate, bs_regs=bs_regs_separate, **tabulate_kw)
    tabulate_proposed_hparams_shared_slope(lr_reg_shared=lr_reg_shared, bs_reg_shared=bs_reg_shared, **tabulate_kw)
    tabulate_baseline_hparams(**tabulate_kw)


if __name__ == "__main__":
    main()