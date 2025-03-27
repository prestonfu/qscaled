import numpy as np
import matplotlib.pyplot as plt


def plot_per_env_utd(ours_df, baseline_df, envs, n_utds, n_envs):
    fig, axs = plt.subplots(n_utds, n_envs, figsize=(3.5 * n_envs, 2.5 * n_utds))
    fig.suptitle('Learning Curves by Environment and UTD Ratio')

    def helper(axs, df, label, color):
        # Group data by environment and UTD ratio
        for i, env in enumerate(envs):
            env_data = df[df['env_name'] == env]
            for j, utd in enumerate(sorted(env_data['utd'].unique())):
                utd_data = env_data[env_data['utd'] == utd]

                ax = axs[j, i]
                ax.set_title(f'{env} (UTD={utd})')

                for _, row in utd_data.iterrows():
                    # ax_label = f"{label}: bs={row['batch_size']}, lr={row['learning_rate']}"
                    ax.plot(row['training_step'], row['mean_return'], color=color, alpha=0.3)
                    ax.plot(row['training_step'], row['return_isotonic'], color=color, alpha=1, label=label)

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
                        ax.errorbar(crossing_x, crossing_y, xerr=crossing_std, capsize=3, color=color)

                ax.set_xlabel('Steps')
                ax.set_ylabel('Return')
                ax.grid(True)
                ax.set_facecolor('#f0f0f0')
                ax.legend()

    helper(axs, ours_df, 'Ours', 'tab:blue')
    helper(axs, baseline_df, 'Baseline', 'tab:orange')
    plt.tight_layout()
    plt.show()
