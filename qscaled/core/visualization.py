import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from qscaled.core.preprocessing import (
    get_envs, 
    get_utds, 
    get_batch_sizes, 
    get_learning_rates
)
from qscaled.utils.fitting import fit_powerlaw, power_law_with_const


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
        colors = sns.color_palette("viridis", n_colors=len(param_values))
        fig, axs = plt.subplots(n_utds, n_envs, figsize=(3.5*n_envs, 2.5*n_utds))
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
                
                ax = axs[j, i]
                ax.set_title(f'{env} (UTD={utd})')
                
                for _, row in utd_data.iterrows():
                    config = row[param_key]
                    if config not in config_colors:
                        config_colors[config] = color_idx
                        color_idx += 1
                    
                    label = f"{param_key}={row[param_key]}"
                    line = ax.plot(row['training_step'], row['mean_return'], alpha=0.3, color=colors[config_colors[config]])
                    line = ax.plot(row['training_step'], row['return_isotonic'], alpha=1, color=colors[config_colors[config]])
                    
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
                        ax.errorbar(crossing_x, crossing_y, xerr=crossing_std, fmt='none', color=colors[config_colors[config] % len(colors)], capsize=3)

                    # Only add to legend if we haven't seen this combination before
                    if label not in labels:
                        lines.append(line[0])
                        labels.append(label)
                
                ax.set_xlabel('Steps')
                ax.set_ylabel('Return')
                ax.grid(True)
                ax.set_facecolor('#f0f0f0')

        # Sort labels by learning rate
        sorted_indices = [i for i, _ in sorted(enumerate(labels), key=lambda x: float(x[1].replace(f'{param_key}=', '')))]
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
    fig_lr, axes_lr = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharey=True)
    axes_lr = axes_lr.flatten()

    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
        
        # Calculate correlation for point estimate
        point_lr_corr = np.corrcoef(
            np.log10(env_data['utd']), np.log10(env_data['best_lr'])
        )[0, 1]
        axes_lr[i].plot(
            env_data['utd'],
            env_data['best_lr'],
            'o-',
            label=f'Point estimate (corr={point_lr_corr:.3f})',
        )

        # Add bootstrapped confidence intervals
        bootstrap_lr_corr = np.corrcoef(
            np.log10(env_data['utd']), np.log10(env_data['best_lr_bootstrap_mean'])
        )[0, 1]
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
        lr_corr = np.corrcoef(
            np.log10(env_data['utd']), np.log10(env_data['best_lr_bsmean'])
        )[0, 1]
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
        lr_corr_all = np.corrcoef(
            np.log10(env_data['utd']), np.log10(env_data['best_lr_bootstrap_bsmean'])
        )[0, 1]
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
    fig_bs, axes_bs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharey=True)
    axes_bs = axes_bs.flatten()

    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
        
        # Calculate correlation for point estimate
        point_bs_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs']))[0,1]
        axes_bs[i].errorbar(
            env_data['utd'], 
            env_data['best_bs'], 
            yerr=None,
            fmt='o-', 
            label=f'Point estimate (corr={point_bs_corr:.3f})'
        )
        
        # Add bootstrapped confidence intervals
        bootstrap_bs_corr = np.corrcoef(
            np.log10(env_data['utd']), np.log10(env_data['best_bs_bootstrap_mean'])
        )[0, 1]
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
        bs_corr = np.corrcoef(
            np.log10(env_data['utd']), np.log10(env_data['best_bs_lrmean'])
        )[0, 1]
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
        bs_corr_all = np.corrcoef(
            np.log10(env_data['utd']), np.log10(env_data['best_bs_bootstrap_lrmean'])
        )[0, 1]
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

 
def plot_closest_data_efficiency(df, proposed_lr_values, proposed_bs_values):
    """
    Plot time to threshold for each environment using the existing
    hyperparameters closest to fit.
    """
    envs = get_envs(df)
    utds = get_utds(df)
    n_envs = len(envs)
    n_cols = 4
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 7))
    axes = axes.flatten()

    env_data_dict = {}

    for i, env in enumerate(envs):
        env_data = []
        for utd, e, lr, bs in zip(proposed_lr_values['UTD'],
                                  proposed_lr_values['Environment'], 
                                  proposed_lr_values['Learning Rate'],
                                  proposed_bs_values['Batch Size']):
            utd = float(utd)
            if utd not in utds:  # Only consider UTDs in the data, not extrapolated
                continue
            if e == env:
                env_df = df[df['env_name'] == env]
                lr_diffs = np.abs(env_df['learning_rate'] - float(lr))
                bs_diffs = np.abs(env_df['batch_size'] - float(bs))
                utd_diffs = np.abs(env_df['utd'] - float(utd))
                closest_match = (lr_diffs + bs_diffs + utd_diffs).idxmin()
                env_data.append((utd, df.loc[closest_match, 'crossings'][-1]))
        
        env_data_dict[env] = env_data
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
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    return env_data_dict


def plot_averaged_data_efficiency(env_data_dict):
    """
    Plot time to threshold for each environment on a single plot, and fit
    their median-normalized data efficiency.
    """
    plt.figure(figsize=(10, 6))
    
    median_times = np.array([
        np.median(list(zip(*data))[1]) for data in env_data_dict.values() if len(data) > 0
    ])
    scaling = 1 / median_times

    # Store the normalized data for each environment
    normalized_times_all = []
    for i, (env, data) in enumerate(env_data_dict.items()):
        utds, times = zip(*data)
        normalized_times = np.array(times) * scaling[i]
        normalized_times_all.append(normalized_times)
        plt.plot(utds, normalized_times, '--', label=env, alpha=0.5)

    # Calculate and plot the average across all environments    
    normalized_times_all = np.array(normalized_times_all)

    plt.plot(utds, np.mean(normalized_times_all, axis=0), 'ko', linewidth=3, label='Average', alpha=0.8)

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
    a, b, c = fit_powerlaw(utds, mean_times)    
    x_smooth = np.logspace(np.log10(min(utds)), np.log10(max(utds)), 100)
    y_fitted_powerlaw = power_law_with_const(x_smooth, a, b, c)
    plt.plot(x_smooth, y_fitted_powerlaw, linewidth=2, 
             label=f'y = {c:.2f} + (x/{b:.2f})^-{a:.2f}', color='blue')

    plt.xlabel('Updates per Data point (UTD)')
    plt.ylabel('Normalized Time to Threshold')
    plt.title('Normalized Time to Threshold vs UTD Across Environments')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_per_env_utd(ours_df, baseline_df, envs, n_utds, n_envs):
    fig, axs = plt.subplots(n_utds, n_envs, figsize=(3.5*n_envs, 2.5*n_utds))
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
    