import matplotlib.pyplot as plt
import seaborn as sns

from qscaled.core.preprocessing import get_envs, get_utds, get_batch_sizes, get_learning_rates


def _plot_per_lr_or_bs(df, thresholds, group_name):
    """
    For group_name = learning rate (batch size), plot learning curves grouped by
    environment, utd, and learning rate (batch size) across different
    values of batch size (learning rate).
    """
    assert group_name in ['lr', 'bs']

    utds = get_utds(df)
    envs = get_envs(df)
    batch_sizes = get_batch_sizes(df)
    learning_rates = get_learning_rates(df)
    n_utds = len(utds)
    n_envs = len(envs)

    if group_name == 'lr':
        group_key = 'learning_rate'
        param_key = 'batch_size'
        group_values = learning_rates
        param_values = batch_sizes
        description = 'Learning Rate'
    else:
        group_key = 'batch_size'
        param_key = 'learning_rate'
        group_values = batch_sizes
        param_values = learning_rates
        description = 'Batch Size'

    for group_value in group_values:
        colors = sns.color_palette('viridis', n_colors=len(param_values))
        fig, axes = plt.subplots(n_utds, n_envs, figsize=(3.5 * n_envs, 2.5 * n_utds))
        fig.suptitle(f'Learning Curves by Environment and UTD Ratio ({description} = {group_value})')

        lines = []
        labels = []
        config_colors = {}
        color_idx = 0

        # Filter data for current batch size
        df_filtered = df[df[group_key] == group_value]

        # Group data by environment and UTD ratio
        for i, env in enumerate(envs):
            env_data = df_filtered[df_filtered['env_name'] == env]

            # Create separate plots for each UTD value
            for j, utd in enumerate(utds):
                utd_data = env_data[env_data['utd'] == utd]
                # Sort by learning rate only since we're already filtering by UTD
                utd_data = utd_data.sort_values(param_key)

                ax = axes[j, i]
                ax.set_title(f'{env} (UTD={utd})')

                for _, row in utd_data.iterrows():
                    config = row[param_key]
                    if config not in config_colors:
                        config_colors[config] = color_idx
                        color_idx += 1

                    label = f'{param_key}={row[param_key]}'
                    line = ax.plot(
                        row['training_step'], row['mean_return'], alpha=0.3, color=colors[config_colors[config]]
                    )
                    line = ax.plot(
                        row['training_step'], row['return_isotonic'], alpha=1, color=colors[config_colors[config]]
                    )

                    # use the crossings column to plot crossings
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        ax.plot(crossing_x, crossing_y, 'o', color=colors[config_colors[config] % len(colors)])

                    # Plot crossing standard deviations as error bars
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        crossing_std = row['crossings_std'][k]
                        ax.errorbar(
                            crossing_x,
                            crossing_y,
                            xerr=crossing_std,
                            fmt='none',
                            color=colors[config_colors[config] % len(colors)],
                            capsize=3,
                        )

                    # Only add to legend if we haven't seen this combination before
                    if label not in labels:
                        lines.append(line[0])
                        labels.append(label)

                ax.set_xlabel('Steps')
                ax.set_ylabel('Return')
                ax.grid(True)
                ax.set_facecolor('#f0f0f0')

        # Sort labels by learning rate
        sorted_indices = [
            i for i, _ in sorted(enumerate(labels), key=lambda x: float(x[1].replace(f'{param_key}=', '')))
        ]
        lines = [lines[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        # Create a single legend outside all subplots
        fig.legend(lines, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=(len(labels)), fontsize=12)
        plt.tight_layout()
        plt.show()


def plot_per_learning_rate(df, thresholds):
    """
    Plot learning curves grouped by environment, utd, and learning rate
    across different values of batch size.
    """
    _plot_per_lr_or_bs(df, thresholds, 'lr')


def plot_per_batch_size(df, thresholds):
    """
    Plot learning curves grouped by environment, utd, and batch size
    across different values of learning rate.
    """
    _plot_per_lr_or_bs(df, thresholds, 'bs')
