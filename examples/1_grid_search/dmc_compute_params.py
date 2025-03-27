import numpy as np
import pandas as pd

from qscaled.core.preprocessing import bootstrap_crossings
from qscaled.core.grid_search.bootstrap_envsteps_to_thresh import (
    grid_best_uncertainty_lr,
    grid_best_uncertainty_bs,
    get_bootstrap_optimal,
    compute_bootstrap_averages,
)
from qscaled.core.grid_search.linear_fit import linear_fit_separate, linear_fit_shared
from qscaled.utils.configs import SweepConfig

np.random.seed(42)


def main():
    """
    To use this code on your own data:
    1. Label your Wandb runs with tags.
    2. Fill in `MyCollector` in qscaled/wandb_utils.
    3. Update the information in `SweepConfig` below.

    The latter two steps take ~10 minutes!

    If you set `wandb_collect == True`, your `zip` file will be rebuilt using your
    Wandb collector. Otherwise, the `zip` file must be present.
    """
    wandb_collect = False
    wandb_collector = None  # Zip data was collected separately.
    name = 'dmc_sweep'

    # Maximum possible returns (estimated with infinite data and compute) on each
    # environment. These are mostly eyeballed such that runs reach 80% (hence
    # the 1.25 multiplier) but not 90%. There is some variation for different environments.

    config = SweepConfig(
        name=name,
        max_returns={},  # No need to normalize returns, DMC is already 0-1000.
        returns_key=None,  # Not present in zip data.
        utds_to_predict=[0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128],
        wandb_collector=wandb_collector,
        zip_load_cls=FullGroupedLoaderUnlabeled,
        env_step_freq=1000,
        env_step_start=5000,
    )

    grid_search_df = save_and_load(config)
    grid_search_df = bootstrap_crossings(grid_search_df, config.thresholds, filename=name)

    # Bootstrapping and Fitting
    best_lr = grid_best_uncertainty_lr(grid_search_df)
    best_bs = grid_best_uncertainty_bs(grid_search_df)
    best_lr_bs = best_lr.groupby(['env_name', 'utd']).apply(get_bootstrap_optimal, include_groups=False).reset_index()
    best_lr_bs = compute_bootstrap_averages(best_lr, best_bs, best_lr_bs)

    # Empirically, we find that using a shared slope does better.
    (
        proposed_lr_values_separate,
        proposed_bs_values_separate,
        lr_slopes_separate,
        lr_intercepts_separate,
        bs_slopes_separate,
        bs_intercepts_separate,
    ) = linear_fit_separate(
        config.utds_to_predict,
        grid_search_df,
        best_lr_bs,
        outputs_dir='../outputs',
        save_path=None,
        plot=False,
    )

    (
        proposed_lr_values_shared,
        proposed_bs_values_shared,
        lr_shared_slope_shared,
        lr_env_intercepts_shared,
        bs_shared_slope_shared,
        bs_env_intercepts_shared,
    ) = linear_fit_shared(
        config.utds_to_predict,
        grid_search_df,
        best_lr_bs,
        outputs_dir='../outputs',
        save_path=name,
        plot=False,
    )

    pd.options.display.float_format = '{:.2e}'.format
    proposed_values_df = tabulate_proposed_params(
        config.utds_to_predict,
        proposed_lr_values_shared,
        proposed_bs_values_shared,
        outputs_dir='../outputs',
        save_path=name,
        verbose=True,
    )

    baseline_values_df = tabulate_baseline_params(
        config.utds_to_predict,
        utd=2,
        df=grid_search_df,
        outputs_dir='../outputs',
        save_path=name,
    )


if __name__ == '__main__':
    main()
