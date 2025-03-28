import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Tuple
from rliable import plot_utils
from matplotlib.lines import Line2D

from qscaled.utils.power_law import fit_powerlaw, power_law_with_const
from qscaled.utils.plot_utils import make_smooth_x_range, ax_set_x_bounds_and_scale, ax_set_y_bounds_and_scale, COLORS


def compute_data_efficiency_per_env(df, envs):
    """Compute the data efficiency dictionary for each environment."""
    data_efficiency_dict = {}

    for env in envs:
        env_df = df[df['env_name'] == env]
        utds = sorted(env_df['utd'].values)
        times = [env_df[env_df['utd'] == utd]['crossings'].iloc[0][:] for utd in utds]
        data_efficiency_dict[env] = list(zip(utds, times))

    return data_efficiency_dict


def plot_data_efficiency_per_env(ours_data_efficiency_dict, baseline_data_efficiency_dict, envs):
    """Plot the number of environment steps taken to achieve each performance threshold."""
    n_envs = len(envs)
    fig, axes = plt.subplots(2, n_envs, figsize=(3.5 * n_envs, 2.5 * 2), sharex='col', sharey='col')
    fig.suptitle('Data Efficiency by Environment')

    def plot_helper(axes, data_efficiency_dict):
        axes = axes.flatten()
        for i, env in enumerate(envs):
            if env in data_efficiency_dict and len(data_efficiency_dict[env]) > 0:
                utds, times = zip(*data_efficiency_dict[env])
                axes[i].plot(utds, times, 'o-')
                axes[i].set_xlabel('UTD')
                axes[i].set_ylabel('Env steps to Threshold')
                axes[i].set_title(env)
                axes[i].set_xscale('log')
                axes[i].set_yscale('log')
                axes[i].grid(True, alpha=0.3)

    plot_helper(axes[0], ours_data_efficiency_dict)
    plot_helper(axes[1], baseline_data_efficiency_dict)

    fig.text(-0.01, 0.75, 'Ours', va='center', ha='center', fontsize=14, rotation=90)
    fig.text(-0.01, 0.25, 'Baseline', va='center', ha='center', fontsize=14, rotation=90)
    plt.tight_layout()
    plt.show()


def compute_normalized_times(data_efficiency_dict, envs):
    """
    Compute normalized times and scaling factors for each environment.
    Implements Appendix D of the paper.
    """
    median_times = np.array(
        [np.median(list(zip(*data_efficiency_dict[env]))[1]) for env in envs if len(data_efficiency_dict[env]) > 0]
    )
    median_median = np.median(median_times)
    scaling = 1 / median_times

    normalized_times_all = []
    for i, env in enumerate(envs):
        if len(data_efficiency_dict[env]) > 0:
            _, times = zip(*data_efficiency_dict[env])
            normalized_times = np.array(times) * scaling[i]
            normalized_times_all.append(normalized_times)

    mean_normalized_times = np.mean(normalized_times_all, axis=0)
    return np.array(normalized_times_all), mean_normalized_times, median_median


def plot_data_efficiency_averaged(ours_mean_normalized_times, baseline_mean_normalized_times, utds, thresholds):
    n_thresholds = len(thresholds)
    colors = sns.color_palette('viridis', n_colors=n_thresholds)

    def helper(ax, mean_normalized_times, title):
        for i in range(n_thresholds):
            ax.plot(
                utds,
                mean_normalized_times[:, i],
                'o',
                color=colors[i],
                label=f'{thresholds[i]}',
                markersize=8,
                alpha=0.8,
            )

        ax.set_title(title)
        ax.set_xlabel('Updates per Data point (UTD)')
        ax.set_ylabel('Normalized Time to Threshold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    helper(axes[0], ours_mean_normalized_times, 'Ours')
    helper(axes[1], baseline_mean_normalized_times, 'Baseline')

    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle('Normalized Time to Threshold vs UTD Across Environments', fontsize=14)

    try:
        plt.tight_layout()
    except ValueError as e:
        if 'Data has no positive values' in str(e):
            warnings.warn(
                'Matplotlib error coming. In the per-env data efficiency plot, check that each performance '
                'threshold is achieved for every UTD. If not, decrease your thresholds in the config, and '
                'call `bootstrap_crossings` with `use_cached=False`.',
                UserWarning,
            )
        raise e

    plt.show()


def make_data_pareto_fits(mean_normalized_times, utds, n_thresholds, output_dir=None, save_name=None):
    assert (output_dir is not None) == (save_name is not None), 'Both output_dir and save_name must be provided or not'

    fits = []
    for i in tqdm(range(n_thresholds)):
        params = fit_powerlaw(utds[:], mean_normalized_times[:, i])
        fits.append(params)

    if output_dir is not None:
        full_path = f'{output_dir}/data_pareto_fit/{save_name}.npy'
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        np.save(full_path, fits)

    return fits


def plot_utd_data_pareto_fits(
    ours_fits,
    ours_median_median,
    baseline_fits,
    baseline_median_median,
    utds,
    thresholds,
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    n_thresholds = len(thresholds)
    colors = sns.color_palette('viridis', n_colors=n_thresholds)
    x_smooth = make_smooth_x_range(utds)

    def helper(ax, fits, median_median, label: str):
        for i in range(n_thresholds):
            a, b, c = fits[i]
            data = power_law_with_const(x_smooth, a, b, c) * median_median
            ax.plot(x_smooth, data, '-', color=colors[i], label=thresholds[i])

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True)
        ax.set_xlabel('UTD Ratio')
        ax.set_ylabel('Mean Normalized Training Time')
        ax.set_title(label)

    helper(axes[0], ours_fits, ours_median_median, 'Ours')
    helper(axes[1], baseline_fits, baseline_median_median, 'Baseline')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle('Time vs UTD to reach performance thresholds')
    plt.tight_layout()
    plt.show()


def plot_clean_utd_data_pareto_fit(
    ours_fits,
    ours_normalized_times_all,
    ours_median_median,
    baseline_fits,
    baseline_normalized_times_all,
    baseline_median_median,
    utds,
    thresholds,
    threshold_idx=-1,
    show_baseline=True,
    ylim: Tuple[float, float] | None = None,
    yscale: str = '1',
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.set_size_inches(496.0 / 192 * 2, 369.6 / 192 * 2)

    print(f'Fits using threshold {thresholds[threshold_idx]}')

    def helper(median_median, fits, normalized_times_all, color, label):
        a, b, c = fits[threshold_idx]
        utd_line = make_smooth_x_range(utds)
        data_line = power_law_with_const(utd_line, a, b, c) * median_median
        plt.scatter(
            utds,
            np.mean(normalized_times_all[..., threshold_idx], axis=0) * median_median,
            marker='o',
            color=color,
            alpha=0.8,
            s=100,
        )
        fit_lines = plt.plot(utd_line, data_line, color, linewidth=3)
        asymptote_line = plt.axhline(y=c * median_median, color=color, linewidth=3, linestyle='--')

        print(f'{label}: D_J = {c * median_median:.2e} * (1 + (σ/{b:.2f})**(-{a:.2f}))')
        return fit_lines[0], asymptote_line

    dot_line = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10)
    ours_fit_line, ours_asymptote_line = helper(
        ours_median_median, ours_fits, ours_normalized_times_all, color=COLORS[1], label='Ours'
    )
    lines = [dot_line, ours_fit_line]
    labels = ['Empirical value', 'Ours $\mathcal{D}_J(\sigma)$']

    if show_baseline:
        baseline_fit_line, baseline_asymptote_line = helper(
            baseline_median_median, baseline_fits, baseline_normalized_times_all, color=COLORS[4], label='Baseline'
        )
        lines.append(baseline_fit_line)
        labels.append('Constant fit $\mathcal{D}_J(\sigma)$')
    else:
        lines.append(ours_asymptote_line)
        labels.append(r'Asymptote $\mathcal{D}^{\text{min}}$')

    plt.legend(lines, labels, prop={'size': 14}, frameon=False)

    plt.xscale('log')
    plt.yscale('log')

    plot_utils._annotate_and_decorate_axis(
        ax,
        xlabel='$\sigma$: UTD Ratio',
        ylabel='$\mathcal{D}_J$: Data until $J$',
        labelsize='xx-large',
        ticklabelsize='xx-large',
        grid_alpha=0.2,
        legend=False,
    )

    x = np.logspace(np.log10(min(utds)), np.log10(max(utds)), num=len(utds))
    x_ticks_labels = [round(x) if x >= 0.99 else f'{x:.2f}' for x in x]
    ax.set_xticks(x, x_ticks_labels)
    ax_set_y_bounds_and_scale(ax, ylim, yscale)

    plt.show()


def _plot_compute_vs_data(
    normalized_times_all,
    median_median,
    fits,
    utd_to_batch_size_fn,
    utds,
    model_size,
    threshold_idx,
    color,
    plot_points=True,
    plot_asymptote=True,
):
    a, b, c = fits[threshold_idx]
    utd_line = make_smooth_x_range(utds)
    data_line = power_law_with_const(utd_line, a, b, c) * median_median
    grad_steps_line = utd_line * data_line
    batch_sizes_line = utd_to_batch_size_fn(utd_line)
    compute_line = 10 * grad_steps_line * batch_sizes_line * model_size

    fit_lines = plt.plot(compute_line, data_line, color=color, linewidth=3)

    if plot_points:
        utds = np.array(utds)
        data = np.mean(normalized_times_all[..., threshold_idx], axis=0) * median_median
        grad_steps = utds * data
        batch_sizes = utd_to_batch_size_fn(utds)
        compute = 10 * grad_steps * batch_sizes * model_size

        plt.scatter(compute, data, marker='o', color=color, alpha=0.8, s=100)

    if plot_asymptote:
        asymptote_line = plt.axhline(y=c * median_median, color=color, linewidth=3, linestyle='--')
    else:
        asymptote_line = None

    return (fit_lines[0], asymptote_line), (compute_line, data_line, utd_line)


def plot_clean_compute_data_pareto_fit(
    ours_fits,
    ours_normalized_times_all,
    ours_median_median,
    baseline_fits,
    baseline_normalized_times_all,
    baseline_median_median,
    utd_to_batch_size_fn,
    model_size,
    utds,
    thresholds,
    threshold_idx=-1,
    show_baseline=True,
    xlim: Tuple[float, float] | None = None,
    xscale: str = '1',
    ylim: Tuple[float, float] | None = None,
    yscale: str = '1',
):
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    fig.set_size_inches(496.0 / 192 * 2, 369.6 / 192 * 2)

    print(f'Fits using threshold {thresholds[threshold_idx]}')

    dot_line = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10)
    (ours_fit_line, ours_asymptote_line), _ = _plot_compute_vs_data(
        ours_normalized_times_all,
        ours_median_median,
        ours_fits,
        utd_to_batch_size_fn,
        utds,
        model_size,
        threshold_idx,
        color=COLORS[1],
    )
    lines = [dot_line, ours_fit_line]
    labels = ['Empirical value', 'Ours $\mathcal{D}_J(\mathcal{C}_J)$']

    if show_baseline:
        (baseline_fit_line, baseline_asymptote_line), _ = _plot_compute_vs_data(
            baseline_normalized_times_all,
            baseline_median_median,
            baseline_fits,
            utd_to_batch_size_fn,
            utds,
            model_size,
            threshold_idx,
            color=COLORS[4],
        )
        lines.append(baseline_fit_line)
        labels.append('Constant fit $\mathcal{D}_J(\mathcal{C}_J)$')
    else:
        lines.append(ours_asymptote_line)
        labels.append(r'Asymptote $\mathcal{D}^{\text{min}}$')

    plt.legend(lines, labels, prop={'size': 14}, frameon=False)

    plt.xscale('log')
    plt.yscale('log')

    plot_utils._annotate_and_decorate_axis(
        axes,
        xlabel='$\mathcal{C}_J$: Compute until $J$',
        ylabel='$\mathcal{D}_J$: Data until $J$',
        labelsize='xx-large',
        ticklabelsize='xx-large',
        grid_alpha=0.2,
        legend=False,
    )

    ax_set_x_bounds_and_scale(axes, xlim, xscale)
    ax_set_y_bounds_and_scale(axes, ylim, yscale)

    plt.show()
