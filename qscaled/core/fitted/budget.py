import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
from rliable import plot_utils as rliable_plot_utils
from matplotlib.colors import LinearSegmentedColormap

from qscaled.utils import plot_utils
from qscaled.core.fitted.data_efficiency import _plot_compute_vs_data


def plot_compute_data_isoperformance(
    fits,
    normalized_times_all,
    median_median,
    thresholds,
    utd_to_batch_size_fn,
    utds,
    model_size,
    delta,
    xlim: Union[Tuple[float, float], None] = None,
    xticks: Union[List[float], None] = None,
    xscale: str = '1',
    ylim: Union[Tuple[float, float], None] = None,
    yticks: Union[List[float], None] = None,
    yscale: str = '1',
    show_isocost=False,
    isocost_xlim: Union[Tuple[float, float], None] = None,
    isocost_ylim: Union[Tuple[float, float], None] = None,
):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(496.0 / 192 * 2 * 1.27, 369.6 / 192 * 2)

    cmap = LinearSegmentedColormap.from_list(
        'custom_gradient', [plot_utils.COLORS[0], plot_utils.COLORS[1]]
    )
    n_colors = n_thresholds = len(thresholds)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    min_points_utd = []
    min_points_compute = []
    min_points_data = []

    for i in range(n_thresholds):
        _, (compute, data, utd) = _plot_compute_vs_data(
            normalized_times_all,
            median_median,
            fits,
            utd_to_batch_size_fn,
            utds,
            model_size,
            i,
            color=colors[i],
            plot_points=False,
            plot_asymptote=False,
        )

        min_idx = np.argmin(compute + delta * data)
        plt.plot(
            compute[min_idx],
            data[min_idx],
            'o',
            color=colors[i],
            markersize=10,
            label='Optimal budget' if i == n_thresholds - 1 else None,
            markeredgecolor='black',
        )

        # Store min points for fitting line
        min_points_compute.append(compute[min_idx])
        min_points_data.append(data[min_idx])
        min_points_utd.append(utd[min_idx])

    if show_isocost:
        assert isocost_xlim and isocost_ylim, 'isocost_xlim and isocost_ylim must be provided'
        compute_min_limit, compute_max_limit = isocost_xlim
        data_min_limit, data_max_limit = isocost_ylim
        compute_range = np.logspace(np.log10(compute_min_limit), np.log10(compute_max_limit), 1000)

        for i, (opt_compute, opt_data) in enumerate(zip(min_points_compute, min_points_data)):
            budget = opt_compute + delta * opt_data
            data_range = (budget - compute_range) / delta
            valid_points = (
                (compute_range >= compute_min_limit)
                & (compute_range <= compute_max_limit)
                & (data_range >= data_min_limit)
                & (data_range <= data_max_limit)
            )
            plt.plot(
                compute_range[valid_points],
                data_range[valid_points],
                '-',
                color='gray',
                alpha=0.3,
                linewidth=2,
                label='Iso-cost lines' if i == n_thresholds - 1 else None,
            )

    # UTD extrapolation line
    min_points_compute = np.array(min_points_compute)
    min_points_data = np.array(min_points_data)
    min_points_utd = np.array(min_points_utd)

    # Fit line in log space
    coeffs = np.polyfit(np.log10(min_points_compute), np.log10(min_points_data), 1)

    # Generate points for the fitted line
    x_range = np.log10(min(min_points_compute)), np.log10(max(min_points_compute))
    x_line = np.logspace(
        x_range[0] - 0.2 * (x_range[1] - x_range[0]),
        x_range[1] + 0.2 * (x_range[1] - x_range[0]),
        100,
    )
    y_line = 10 ** (coeffs[0] * np.log10(x_line) + coeffs[1])
    plt.plot(x_line, y_line, '--', color='gray', linewidth=3, alpha=0.8)

    plt.xscale('log')
    plt.yscale('log')
    ax.legend(prop={'size': 14}, ncol=1, frameon=False)

    rliable_plot_utils._annotate_and_decorate_axis(
        ax,
        xlabel='$\mathcal{C}_J$: Compute until $J$',
        ylabel='$\mathcal{D}_J$: Data until $J$',
        labelsize='xx-large',
        ticklabelsize='xx-large',
        grid_alpha=0.2,
        legend=False,
    )

    plot_utils.ax_set_x_bounds_and_scale(ax, xlim, xticks, xscale)
    plot_utils.ax_set_y_bounds_and_scale(ax, ylim, yticks, yscale)

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min(thresholds), vmax=max(thresholds))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('$J$: Performance level', size='xx-large')
    cbar.ax.tick_params(labelsize='xx-large')

    plt.show()

    return min_points_compute, min_points_data, min_points_utd


def plot_budget_extrapolation(
    min_points_compute,
    min_points_data,
    min_points_utd,
    delta,
    thresholds,
    n_extrapolate_points,
    xlim: Union[Tuple[float, float], None] = None,
    xticks: Union[List[float], None] = None,
    xscale: str = '1',
    ylim: Union[Tuple[float, float], None] = None,
    yticks: Union[List[float], None] = None,
    yscale: str = '1',
):
    """
    The optimal UTDs for the top `n_extrapolate_points` performance levels are
    extrapolated from the remaining ones.
    """
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(496.0 / 192 * 2, 369.6 / 192 * 2)

    cmap = LinearSegmentedColormap.from_list(
        'custom_gradient', [plot_utils.COLORS[0], plot_utils.COLORS[1]]
    )
    n_colors = len(thresholds)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    # Fit line in log space
    min_points_budget = min_points_compute + delta * min_points_data
    coeffs = np.polyfit(
        np.log10(min_points_budget[:-n_extrapolate_points]),
        np.log10(min_points_utd[:-n_extrapolate_points]),
        1,
    )

    # Generate points for the fitted line
    budget_range = np.log10(min(min_points_budget)), np.log10(max(min_points_budget))
    budget_line = np.logspace(
        budget_range[0] - 0.2 * (budget_range[1] - budget_range[0]),
        budget_range[1] + 0.2 * (budget_range[1] - budget_range[0]),
        100,
    )
    utd_line = 10 ** (coeffs[0] * np.log10(budget_line) + coeffs[1])

    plt.plot(
        budget_line, utd_line, color='gray', linewidth=3, alpha=0.8, label='$\sigma^*(\mathcal{F})$'
    )
    plt.scatter(
        min_points_budget[:-n_extrapolate_points],
        min_points_utd[:-n_extrapolate_points],
        marker='o',
        color=colors[:-n_extrapolate_points],
        label='$\sigma^*$',
        alpha=0.8,
        s=80,
    )
    plt.scatter(
        min_points_budget[-n_extrapolate_points:],
        min_points_utd[-n_extrapolate_points:],
        marker='x',
        color=colors[-n_extrapolate_points:],
        label='Extrapolation',
        alpha=0.8,
        s=80,
        linewidth=3,
    )

    plt.xscale('log')
    plt.yscale('log')
    ax.legend(prop={'size': 14}, ncol=1, frameon=False)

    rliable_plot_utils._annotate_and_decorate_axis(
        ax,
        xlabel='$\mathcal{F}_J$: Budget until $J$',
        ylabel='$\sigma^*$: optimal UTD',
        labelsize='xx-large',
        ticklabelsize='xx-large',
        grid_alpha=0.2,
        legend=False,
    )

    plot_utils.ax_set_x_bounds_and_scale(ax, xlim, xticks, xscale)
    plot_utils.ax_set_y_bounds_and_scale(ax, ylim, yticks, yscale, yfloat=True)

    # Print fitted function
    print(f'σ* = {10 ** coeffs[1]:.2e} × F_J^{coeffs[0]:.2f}')
