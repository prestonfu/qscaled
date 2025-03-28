import numpy as np

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
    tabulate_baseline_hparams,
)
from qscaled.utils.zip_handler import fetch_zip_data

np.random.seed(42)


def compute_params(config):
    print_hparams = True

    grid_search_df = fetch_zip_data(config, use_cached=True)
    grid_search_df = bootstrap_crossings(grid_search_df, config.thresholds, filename=config.name, use_cached=True)

    # Bootstrapping and Fitting
    best_lr = grid_best_uncertainty_lr(grid_search_df)
    best_bs = grid_best_uncertainty_bs(grid_search_df)
    best_lr_bs = best_lr.groupby(['env_name', 'utd']).apply(get_bootstrap_optimal, include_groups=False).reset_index()
    best_lr_bs = compute_bootstrap_averages(best_lr, best_bs, best_lr_bs)

    # Empirically, we find that using a shared slope does better.
    lr_regs_separate, bs_regs_separate = make_linear_fit_separate_slope(best_lr_bs, '../outputs', config.name)
    lr_reg_shared, bs_reg_shared = make_linear_fit_shared_slope(best_lr_bs, '../outputs', config.name)

    tabulate_kw = dict(
        df_grid=grid_search_df,
        utds_to_predict=config.utds_to_predict,
        outputs_dir='../outputs',
        save_path=config.name,
        verbose=print_hparams,
    )
    tabulate_proposed_hparams_separate_slope(lr_regs=lr_regs_separate, bs_regs=bs_regs_separate, **tabulate_kw)
    tabulate_proposed_hparams_shared_slope(lr_reg_shared=lr_reg_shared, bs_reg_shared=bs_reg_shared, **tabulate_kw)
    tabulate_baseline_hparams(**tabulate_kw)
