import numpy as np
import matplotlib.pyplot as plt

from qscaled.core.preprocessing import get_envs, get_utds
from qscaled.utils import power_law


def plot_closest_data_efficiency(df_grid, proposed_hparams):
    """
    Plot time to threshold for each environment using the existing
    hyperparameters closest to fit.
    """
    envs = get_envs(df_grid)
    utds = get_utds(df_grid)
    n_envs = len(envs)
    n_cols = 4
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()

    closest_data_efficiency_dict = {}

    for i, env in enumerate(envs):
        env_data = []
        for utd, e, lr, bs in zip(
            proposed_hparams['UTD'],
            proposed_hparams['Environment'],
            proposed_hparams['Learning Rate'],
            proposed_hparams['Batch Size'],
        ):
            utd = float(utd)
            if utd not in utds:  # Only consider UTDs in the data, not extrapolated
                continue
            if e == env:
                env_df = df_grid[df_grid['env_name'] == env]
                lr_diffs = np.abs(env_df['learning_rate'] - float(lr))
                bs_diffs = np.abs(env_df['batch_size'] - float(bs))
                utd_diffs = np.abs(env_df['utd'] - float(utd))
                closest_match = (lr_diffs + bs_diffs + utd_diffs).idxmin()
                env_data.append((utd, df_grid.loc[closest_match, 'crossings'][-1]))

        closest_data_efficiency_dict[env] = env_data

        if len(env_data) > 0:
            utds, times = zip(*env_data)
            axes[i].plot(utds, times, 'o-')
            axes[i].set_xlabel('Updates per Data point (UTD)')
            axes[i].set_ylabel('Time to Threshold')
            axes[i].set_title(f'{env}')
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
            axes[i].grid(True, alpha=0.3)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return closest_data_efficiency_dict


def plot_averaged_data_efficiency(closest_data_efficiency_dict):
    """
    Plot time to threshold for each environment on a single plot, and fit
    their median-normalized data efficiency.
    """
    plt.figure(figsize=(9, 6))

    median_times = np.array(
        [
            np.median(list(zip(*data))[1])
            for data in closest_data_efficiency_dict.values()
            if len(data) > 0
        ]
    )
    scaling = 1 / median_times

    # Store the normalized data for each environment
    normalized_times_all = []
    for i, (env, data) in enumerate(closest_data_efficiency_dict.items()):
        utds, times = zip(*data)
        normalized_times = np.array(times) * scaling[i]
        normalized_times_all.append(normalized_times)
        plt.plot(utds, normalized_times, '--', label=env, alpha=0.5)

    # Calculate and plot the average across all environments
    normalized_times_all = np.array(normalized_times_all)

    plt.plot(
        utds, np.mean(normalized_times_all, axis=0), 'ko', linewidth=3, label='Average', alpha=0.8
    )

    # Fit a line to log-transformed data
    log_utds = np.log(utds)
    log_mean_times = np.log(np.mean(normalized_times_all, axis=0))
    slope, intercept = np.polyfit(log_utds, log_mean_times, 1)

    # Plot fit line
    fit_x = np.array([min(utds), max(utds)])
    fit_y = np.exp(slope * np.log(fit_x) + intercept)
    plt.plot(fit_x, fit_y, 'k-', linewidth=2, label=f'y = {np.exp(intercept):.2f} * x^{slope:.2f}')

    # Plot fit curve
    mean_times = np.mean(normalized_times_all, axis=0)
    a, b, c = power_law.fit_powerlaw(utds, mean_times)
    x_smooth = np.logspace(np.log10(min(utds)), np.log10(max(utds)), 100)
    y_fitted_powerlaw = power_law.power_law_with_const(x_smooth, a, b, c)
    plt.plot(
        x_smooth,
        y_fitted_powerlaw,
        linewidth=2,
        label=f'y = {c:.2f} + (x/{b:.2f})^-{a:.2f}',
        color='blue',
    )

    plt.xlabel('Updates per Data point (UTD)')
    plt.ylabel('Normalized Time to Threshold')
    plt.title('Normalized Time to Threshold vs UTD Across Environments')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()
