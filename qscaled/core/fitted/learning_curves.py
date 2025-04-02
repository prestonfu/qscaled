import matplotlib.pyplot as plt

from qscaled.core.preprocessing import get_envs, get_utds


def plot_per_env_utd(ours_df, baseline_df, thresholds):
    envs = get_envs(ours_df)
    utds = get_utds(ours_df)
    n_envs = len(envs)
    n_utds = len(utds)

    fig, axes = plt.subplots(n_utds, n_envs, figsize=(3.5 * n_envs, 2.5 * n_utds))
    fig.suptitle('Learning Curves by Environment and UTD Ratio')

    def helper(axes, df, label, color):
        for i, env in enumerate(envs):
            for j, utd in enumerate(utds):
                subset = df[(df['env_name'] == env) & (df['utd'] == utd)]
                if subset.empty:
                    continue

                ax = axes[j, i]
                ax.set_title(f'{env} (UTD={utd})')

                for _, row in subset.iterrows():
                    # ax_label = f"{label}: bs={row['batch_size']}, lr={row['learning_rate']}"
                    ax.plot(row['training_step'], row['mean_return'], color=color, alpha=0.3)
                    ax.plot(
                        row['training_step'],
                        row['return_isotonic'],
                        color=color,
                        alpha=1,
                        label=label,
                    )

                    # use the crossings column to plot crossings
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        ax.plot(crossing_x, crossing_y, 'o', color=color)

                    # Plot crossing standard deviations as error bars
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        crossing_std = row['crossings_std'][k]
                        ax.errorbar(
                            crossing_x, crossing_y, xerr=crossing_std, capsize=3, color=color
                        )

                ax.set_xlabel('Steps')
                ax.set_ylabel('Return')
                ax.grid(True)
                ax.legend(frameon=False)

    helper(axes, ours_df, 'Ours', 'tab:blue')
    helper(axes, baseline_df, 'Baseline', 'tab:orange')
    plt.tight_layout()
    plt.show()
