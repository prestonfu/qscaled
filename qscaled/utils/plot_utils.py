import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple

COLORS = [
    '#BBCC33',
    '#77AADD',
    '#44BB99',
    '#EEDD88',
    '#EE8866',
    '#FFAABB',
    '#99DDFF',
    '#AAAA00',
    '#DDDDDD',
]


def make_smooth_x_range(xs):
    return np.logspace(np.log10(min(xs)), np.log10(max(xs)), 100)


def set_theme(display_palette=False):
    plt.rcParams['text.usetex'] = False  # Let TeX do the typsetting
    plt.rcParams['text.latex.preamble'] = (
        r'\usepackage{sansmath} \sansmath'  # Force sans-serif math mode (for ax labels)
    )
    plt.rcParams['font.family'] = 'sans-serif'  # ... for regular text
    plt.rcParams['font.sans-serif'] = ['Helveta Nue']  # Choose a nice font here
    sns.set_style('whitegrid')

    if display_palette:
        plt.figure(figsize=(9, 2))
        for i, color in enumerate(COLORS):
            plt.bar(i, 1, color=color, label=color)
        plt.xticks([])
        plt.yticks([])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('Color Palette')
        plt.show()


def ax_set_x_bounds_and_scale(
    ax,
    xlim: Tuple[float, float] | None = None,
    xscale: str = '1',
):
    if xlim is not None:
        x_min, x_max = xlim
        xticks = np.logspace(np.log10(x_min), np.log10(x_max), num=5)
        scaled_xticks = [x / float(xscale) for x in xticks]
        xlabels = [int(x) if x >= 1 else round(x, 2) for x in scaled_xticks]
        ax.xaxis.set_major_locator(plt.FixedLocator(xticks))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(xticks, xlabels)

    if xscale != '1':
        plt.text(
            1.12,
            -0.14,
            f'×{xscale}',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize='x-large',
            alpha=0.8,
        )


def ax_set_y_bounds_and_scale(ax, ylim: Tuple[float, float] | None = None, yscale: str = '1'):
    if ylim is not None:
        y_min, y_max = ylim
        yticks = np.logspace(np.log10(y_min), np.log10(y_max), num=4)
        scaled_yticks = [y / float(yscale) for y in yticks]
        ylabels = [int(y) if y >= 1 else round(y, 2) for y in scaled_yticks]
        ax.yaxis.set_major_locator(plt.FixedLocator(yticks))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_yticks(yticks, ylabels)

    if yscale != '1':
        plt.text(
            -0.045,
            0.93,
            f'×{yscale}',
            transform=ax.transAxes,
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize='x-large',
            alpha=0.8,
        )
