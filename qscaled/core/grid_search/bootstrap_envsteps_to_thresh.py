import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qscaled.core.preprocessing import get_envs, get_utds, get_batch_sizes, get_learning_rates


def _grid_best_uncertainty(df, param_name, print_pivot=False):
    """
    Make and print a table with uncertainty-corrected best
    param_name = learning rate (batch size) for each
    environment, batch size (learning rate), and UTD with environment as rows
    and utd x batch size (learning rate) as columns.

    This description is somewhat confusing; the docstrings for
    `grid_best_uncertainty_lr` and `grid_best_uncertainty_bs` are more clear.
    """
    assert param_name in ['lr', 'bs']

    if param_name == 'lr':
        param_key = 'learning_rate'
        group_key = 'batch_size'
    else:
        param_key = 'batch_size'
        group_key = 'learning_rate'

    grouped = df.groupby(['env_name', group_key, 'utd'])

    # Find best learning rate (batch size) for each group
    results = []
    for (env, group_value, utd), group in grouped:
        threshold_i = -1

        # Time to hit thresholds[threshold_i]
        param_groups = group.groupby(param_key)
        time_to_threshold = param_groups.apply(
            lambda x: x['crossings'].iloc[0][threshold_i], include_groups=False
        ).dropna()

        if len(time_to_threshold) > 0:
            best_value = time_to_threshold.idxmin(skipna=True)
            min_time = time_to_threshold[best_value]
        else:
            best_value = float('nan')
            min_time = float('inf')

        # Get bootstrap samples
        time_to_threshold_bootstrap = param_groups.apply(
            lambda x: x['crossings_bootstrap'].iloc[0][:, threshold_i],
            include_groups=False,
        )
        param_values = np.array(time_to_threshold_bootstrap.index)
        times_bootstrap = np.array(time_to_threshold_bootstrap.tolist())

        # Find best learning rate (batch size)
        times_bootstrap_inf = np.where(np.isnan(times_bootstrap), np.inf, times_bootstrap)
        best_value_bootstrap = param_values[np.argmin(times_bootstrap_inf, axis=0)]
        min_time_bootstrap = np.min(times_bootstrap_inf, axis=0)

        results.append(
            {
                'env_name': env,
                group_key: group_value,  # batch size (learning rate)
                'utd': utd,
                f'best_{param_name}': best_value,
                'time_to_threshold': min_time,
                f'best_{param_name}_bootstrap': best_value_bootstrap,
                'time_to_threshold_bootstrap': min_time_bootstrap,
            }
        )

    df_best = pd.DataFrame(results)

    if print_pivot:
        pd.set_option('display.float_format', '{:.1e}'.format)

        pivot_df = df_best.pivot_table(
            index='utd', columns=['env_name', group_key], values=f'best_{param_name}', aggfunc='first'
        )
        print(f'\nBest {param_key}:')
        print(pivot_df.to_string())

        pivot_df_bootstrap = df_best.pivot_table(
            index='utd',
            columns=['env_name', group_key],
            values=f'best_{param_name}_bootstrap',
            aggfunc=lambda x: np.mean(np.stack(x)),
        )
        print(f'\nUncertainty-Corrected Best {param_key}:')
        print(pivot_df_bootstrap.to_string())

    return df_best


def grid_best_uncertainty_lr(df, print_pivot=False):
    """
    Make and print a table with uncertainty-corrected best learning rate for
    each environment, batch size, and UTD with environment as rows and
    utd x batch size as columns.
    """
    return _grid_best_uncertainty(df, 'lr', print_pivot)


def grid_best_uncertainty_bs(df, print_pivot=False):
    """
    Make and print a table with uncertainty-corrected best batch size for
    each environment, learning rate, and UTD with environment as rows and
    utd x learning rate as columns.
    """
    return _grid_best_uncertainty(df, 'bs', print_pivot)


def get_bootstrap_optimal(group):
    """Get bootstrapped optimal batch sizes."""
    # Get time to threshold bootstrap array for all batch sizes
    batch_sizes = group['batch_size'].values
    lr_bootstrap = np.stack(
        group['best_lr_bootstrap'].values
    )  # 100 bootstrap samples all have different optimal learning rates
    times_bootstrap = np.stack(
        group['time_to_threshold_bootstrap'].values
    )  # 100 times to threshold corresponding to different bootstrap samples

    # Find optimal batch size index for each bootstrap sample
    # Replace nans with large values so they are never selected as minimum
    times_bootstrap = np.nan_to_num(times_bootstrap, nan=np.inf)
    optimal_indices_bootstrap = np.argmin(times_bootstrap, axis=0)  # 100 indices of optimal batch sizes
    best_lr_bootstrap = lr_bootstrap[
        optimal_indices_bootstrap, np.arange(times_bootstrap.shape[1])
    ]  # for each bootstrap sample, get the learning rate corresponding to the optimal batch size
    best_times_bootstrap = times_bootstrap[
        optimal_indices_bootstrap, np.arange(times_bootstrap.shape[1])
    ]  # for each bootstrap sample, get the time to threshold corresponding to the optimal batch size
    best_bs_bootstrap = batch_sizes[optimal_indices_bootstrap]  # for each bootstrap sample, get the optimal batch size

    # Get point estimate
    best_idx = group['time_to_threshold'].argmin()
    best_bs = group.iloc[best_idx]['batch_size']
    best_lr = group.iloc[best_idx]['best_lr']
    best_time = group.iloc[best_idx]['time_to_threshold']

    return pd.Series(
        {
            'best_lr': best_lr,
            'best_bs': best_bs,
            'time_to_threshold': best_time,
            'best_lr_bootstrap': best_lr_bootstrap,
            'best_bs_bootstrap': best_bs_bootstrap,
            'time_to_threshold_bootstrap': best_times_bootstrap,
        }
    )


def compute_bootstrap_averages(df_best_lr, df_best_bs, df_best_lr_bs):
    """Get optimal bs and lr averaged across lr and bs, respectively."""

    for env in get_envs(df_best_lr_bs):
        env_mask = df_best_lr_bs['env_name'] == env
        env_data = df_best_lr_bs[env_mask]

        # Calculate bootstrap mean and std for each UTD
        best_bs_bootstrap = np.stack(env_data['best_bs_bootstrap'].values)
        mean_bs_bootstrap = np.mean(best_bs_bootstrap, axis=1)
        std_bs_bootstrap = np.std(best_bs_bootstrap, axis=1)
        df_best_lr_bs.loc[env_mask, 'best_bs_bootstrap_mean'] = mean_bs_bootstrap
        df_best_lr_bs.loc[env_mask, 'best_bs_bootstrap_std'] = std_bs_bootstrap

        # Calculate mean and std across learning rates
        env_data_mean = df_best_bs[df_best_bs['env_name'] == env]
        utd_groups = env_data_mean.groupby('utd')
        mean_bs = utd_groups['best_bs'].mean()
        std_bs = utd_groups['best_bs'].std()
        df_best_lr_bs.loc[env_mask, 'best_bs_lrmean'] = [mean_bs[utd] for utd in env_data['utd']]
        df_best_lr_bs.loc[env_mask, 'best_bs_lrmean_std'] = [std_bs[utd] for utd in env_data['utd']]

        # Calculate bootstrap mean and std across learning rates
        best_bs_bootstrap = np.stack([np.stack(g['best_bs_bootstrap'].values) for _, g in utd_groups])
        mean_bs_all = np.mean(best_bs_bootstrap, axis=(1, 2))
        std_bs_all = np.std(best_bs_bootstrap, axis=(1, 2))
        df_best_lr_bs.loc[env_mask, 'best_bs_bootstrap_lrmean'] = [
            mean_bs_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']
        ]
        df_best_lr_bs.loc[env_mask, 'best_bs_bootstrap_lrmean_std'] = [
            std_bs_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']
        ]

        # Calculate bootstrap mean and std for each UTD
        best_lr_bootstrap = np.stack(env_data['best_lr_bootstrap'].values)
        mean_lr_bootstrap = np.mean(best_lr_bootstrap, axis=1)
        std_lr_bootstrap = np.std(best_lr_bootstrap, axis=1)
        df_best_lr_bs.loc[env_mask, 'best_lr_bootstrap_mean'] = mean_lr_bootstrap
        df_best_lr_bs.loc[env_mask, 'best_lr_bootstrap_std'] = std_lr_bootstrap

        # Calculate mean and std across batch sizes
        env_data_mean = df_best_lr[df_best_lr['env_name'] == env]
        utd_groups = env_data_mean.groupby('utd')
        mean_lr = utd_groups['best_lr'].mean()
        std_lr = utd_groups['best_lr'].std()
        df_best_lr_bs.loc[env_mask, 'best_lr_bsmean'] = [mean_lr[utd] for utd in env_data['utd']]
        df_best_lr_bs.loc[env_mask, 'best_lr_bsmean_std'] = [std_lr[utd] for utd in env_data['utd']]

        # Calculate bootstrap mean and std across batch sizes
        best_lr_bootstrap = np.stack([np.stack(g['best_lr_bootstrap'].values) for _, g in utd_groups])
        mean_lr_all = np.mean(best_lr_bootstrap, axis=(1, 2))
        std_lr_all = np.std(best_lr_bootstrap, axis=(1, 2))
        df_best_lr_bs.loc[env_mask, 'best_lr_bootstrap_bsmean'] = [
            mean_lr_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']
        ]
        df_best_lr_bs.loc[env_mask, 'best_lr_bootstrap_bsmean_std'] = [
            std_lr_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']
        ]

    return df_best_lr_bs


def plot_bootstrap_average_params(df_best_lr_bs):
    """
    Plot hyperparameters and batch sizes for the various methods computed
    by `compute_bootstrap_averages`.
    """
    envs = get_envs(df_best_lr_bs)
    n_envs = len(envs)

    # Create first figure for batch size plots
    n_cols = 4
    n_rows = (n_envs + n_cols - 1) // n_cols

    # Plot 1: Optimal learning rate vs UTD with bootstrap CIs and mean optimal learning rate
    fig_lr, axes_lr = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
    axes_lr = axes_lr.flatten()

    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]

        # Calculate correlation for point estimate
        point_lr_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr']))[0, 1]
        axes_lr[i].plot(
            env_data['utd'],
            env_data['best_lr'],
            'o-',
            label=f'Point estimate (corr={point_lr_corr:.3f})',
        )

        # Add bootstrapped confidence intervals
        bootstrap_lr_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr_bootstrap_mean']))[0, 1]
        axes_lr[i].errorbar(
            env_data['utd'],
            env_data['best_lr_bootstrap_mean'],
            yerr=env_data['best_lr_bootstrap_std'],
            fmt='o-',
            capsize=5,
            alpha=0.4,
            label=f'Bootstrap (corr={bootstrap_lr_corr:.3f})',
        )

        # Add mean optimal learning rate
        lr_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr_bsmean']))[0, 1]
        axes_lr[i].errorbar(
            env_data['utd'],
            env_data['best_lr_bsmean'],
            yerr=env_data['best_lr_bsmean_std'],
            fmt='o-',
            capsize=5,
            alpha=0.4,
            label=f'Mean across BSs (corr={lr_corr:.3f})',
        )

        # Add mean learning rate averaged across batch sizes and bootstrap intervals
        lr_corr_all = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr_bootstrap_bsmean']))[0, 1]
        axes_lr[i].errorbar(
            env_data['utd'],
            env_data['best_lr_bootstrap_bsmean'],
            yerr=env_data['best_lr_bootstrap_bsmean_std'],
            fmt='o-',
            capsize=5,
            alpha=0.4,
            label=f'Bootstrap Mean across BSs (corr={lr_corr_all:.3f})',
        )

        axes_lr[i].set_xscale('log')
        axes_lr[i].set_yscale('log')
        axes_lr[i].set_xlabel('UTD')
        axes_lr[i].set_title(f'{env}\nCorr: {lr_corr:.3f}')
        axes_lr[i].grid(True)
        axes_lr[i].legend()

    # Remove empty subplots from second figure
    for j in range(i + 1, len(axes_lr)):
        fig_lr.delaxes(axes_lr[j])
    plt.suptitle(r'$\eta^*$: Best learning rate')
    plt.tight_layout()
    plt.show()

    # Plot 2: Optimal batch size vs UTD with bootstrap CIs and mean optimal batch size
    fig_bs, axes_bs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)
    axes_bs = axes_bs.flatten()

    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]

        # Calculate correlation for point estimate
        point_bs_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs']))[0, 1]
        axes_bs[i].errorbar(
            env_data['utd'],
            env_data['best_bs'],
            yerr=None,
            fmt='o-',
            label=f'Point estimate (corr={point_bs_corr:.3f})',
        )

        # Add bootstrapped confidence intervals
        bootstrap_bs_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs_bootstrap_mean']))[0, 1]
        axes_bs[i].errorbar(
            env_data['utd'],
            env_data['best_bs_bootstrap_mean'],
            yerr=env_data['best_bs_bootstrap_std'],
            fmt='o-',
            capsize=5,
            alpha=0.4,
            label=f'Bootstrap (corr={bootstrap_bs_corr:.3f})',
        )

        # Add mean optimal batch size
        bs_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs_lrmean']))[0, 1]
        axes_bs[i].errorbar(
            env_data['utd'],
            env_data['best_bs_lrmean'],
            yerr=env_data['best_bs_lrmean_std'],
            fmt='o-',
            capsize=5,
            alpha=0.4,
            label=f'Mean across LRs (corr={bs_corr:.3f})',
        )

        # Add mean batch size averaged across learning rates and bootstrap intervals
        bs_corr_all = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs_bootstrap_lrmean']))[0, 1]
        axes_bs[i].errorbar(
            env_data['utd'],
            env_data['best_bs_bootstrap_lrmean'],
            yerr=env_data['best_bs_bootstrap_lrmean_std'],
            fmt='o-',
            capsize=5,
            alpha=0.4,
            label=f'Bootstrap Mean across LRs (corr={bs_corr_all:.3f})',
        )

        axes_bs[i].set_xscale('log')
        axes_bs[i].set_yscale('log')
        axes_bs[i].set_xlabel('UTD')
        axes_bs[i].set_title(f'{env}')
        axes_bs[i].grid(True)
        axes_bs[i].legend()

    # Remove empty subplots from first figure
    for j in range(i + 1, len(axes_bs)):
        fig_bs.delaxes(axes_bs[j])
    plt.suptitle(r'$B^*$: Best batch size')
    plt.tight_layout()
    plt.show()
