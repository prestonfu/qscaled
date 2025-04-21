from qscaled.utils.zip_handler import fetch_zip_data
from qscaled.core.preprocessing import bootstrap_crossings
from qscaled.core.grid_search import bootstrapping, linear_fit


def compute_params(config, output_dir):
    print_hparams = True

    grid_search_df = fetch_zip_data(config, use_cached=True)
    grid_search_df = bootstrap_crossings(
        grid_search_df, config.thresholds, filename=config.name, use_cached=True
    )

    # Bootstrapping and Fitting
    best_lr = bootstrapping.grid_best_uncertainty_lr(grid_search_df)
    best_bs = bootstrapping.grid_best_uncertainty_bs(grid_search_df)
    best_lr_bs = (
        best_lr.groupby(['env_name', 'utd'])
        .apply(bootstrapping.get_bootstrap_optimal)
        .reset_index()
    )
    best_lr_bs = bootstrapping.compute_bootstrap_averages(best_lr, best_bs, best_lr_bs)

    # Empirically, we find that using a shared slope does better.
    lr_regs_separate, bs_regs_separate = linear_fit.make_linear_fit_separate_slope(
        best_lr_bs, '../outputs', config.name
    )
    lr_reg_shared, bs_reg_shared = linear_fit.make_linear_fit_shared_slope(
        best_lr_bs, '../outputs', config.name
    )

    tabulate_kw = dict(
        df_grid=grid_search_df,
        utds_to_predict=config.utds_to_predict,
        outputs_dir=output_dir,
        save_path=config.name,
        verbose=print_hparams,
    )
    linear_fit.tabulate_proposed_hparams_separate_slope(
        lr_regs=lr_regs_separate, bs_regs=bs_regs_separate, **tabulate_kw
    )
    linear_fit.tabulate_proposed_hparams_shared_slope(
        lr_reg_shared=lr_reg_shared, bs_reg_shared=bs_reg_shared, **tabulate_kw
    )
    linear_fit.tabulate_baseline_hparams(utd_at=config.baseline_utd_at, **tabulate_kw)
