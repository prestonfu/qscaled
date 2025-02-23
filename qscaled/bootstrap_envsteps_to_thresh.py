import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

from sklearn.linear_model import LinearRegression

script_dir = os.path.dirname(os.path.abspath(__file__))
outputs_dir = os.path.join(script_dir, '../outputs') 


def grid_best_uncertainty_lr(df, print_pivot=False):
    """
    Make and print a table with uncertainty-corrected best learning rate for 
    each environment, batch size, and UTD with environment as rows and 
    utd x batch size as columns.
    """

    # Group data by environment, batch size and UTD
    grouped = df.groupby(['env_name', 'batch_size', 'utd'])

    # Find best learning rate for each group
    results = []
    for (env, bs, utd), group in grouped:
        threshold_i = -1
        # Group by learning rate and get time to threshold
        lr_groups = group.groupby('learning_rate')
        time_to_threshold = lr_groups.apply(lambda x: x['crossings'].iloc[0][threshold_i], include_groups=False).dropna()
        
        # Get bootstrap samples
        time_to_threshold_bs = lr_groups.apply(lambda x: x['crossings_bootstrap'].iloc[0][:, threshold_i], include_groups=False)
        lrs = np.array(time_to_threshold_bs.index)
        times_bs = np.array(time_to_threshold_bs.tolist())
        
        # Find best learning rates
        times_bs_inf = np.where(np.isnan(times_bs), np.inf, times_bs)
        best_lr_bootstrap = lrs[np.argmin(times_bs_inf, axis=0)]
        best_times_bootstrap = np.min(times_bs_inf, axis=0)
        
        # Get point estimates
        try:
            best_lr = time_to_threshold.idxmin(skipna=True) if len(time_to_threshold) > 0 else None
            min_time = time_to_threshold[best_lr] if best_lr is not None else None
        except:
            best_bs = float('nan')
            min_time = float('inf')
        
        results.append({
            'env_name': env,
            'batch_size': bs,
            'utd': utd,
            'best_lr': best_lr,
            'time_to_threshold': min_time,
            'best_lr_bootstrap': best_lr_bootstrap, # run learning rate selection 100 times and record best learning rate
            'time_to_threshold_bootstrap': best_times_bootstrap, # run learning rate selection 100 times and record corresponding time to threshold
        })

    # Create DataFrame
    df_best_lr = pd.DataFrame(results)

    if print_pivot:
        # Pivot table to get environments as rows and utd x batch_size as columns
        pivot_df = df_best_lr.pivot_table(
            index='utd',
            columns=['env_name', 'batch_size'],
            values='best_lr',
            aggfunc='first'
        )

        # Print formatted table with scientific notation
        pd.set_option('display.float_format', '{:.1e}'.format)
        print("\nBest Learning Rates:")
        print(pivot_df.to_string())

        # now do the same for uncertainty-corrected best learning rates
        # Create pivot table for uncertainty-corrected best learning rates
        # Use mean of bootstrap samples as uncertainty-corrected estimate
        pivot_df_bootstrap = df_best_lr.pivot_table(
            index='utd',
            columns=['env_name', 'batch_size'],
            values='best_lr_bootstrap',
            aggfunc=lambda x: np.mean(np.stack(x))
        )

        # Print formatted table with scientific notation
        print("\nUncertainty-Corrected Best Learning Rates:")
        print(pivot_df_bootstrap.to_string())
    
    return df_best_lr


def grid_best_uncertainty_bs(df, print_pivot=False):
    """
    Make and print a table with uncertainty-corrected best batch size for 
    each environment, learning rate, and UTD with environment as rows and 
    utd x learning rate as columns.
    """

    # Group data by environment, UTD, and learning rate
    grouped = df.groupby(['env_name', 'utd', 'learning_rate'])

    # Find best batch size for each group
    results = []
    for (env, utd, lr), group in grouped:
        # Drop any rows where crossings[-1] is None
        group = group.dropna(subset=['crossings'])

        if len(group) > 0:
            # Get the last crossing time for each row and bootstrap samples
            group['last_crossing'] = group['crossings'].apply(lambda x: x[-1] if len(x) > 0 else float('inf'))
            group['last_crossing_bootstrap'] = group['crossings_bootstrap'].apply(lambda x: x[:, -1])

            # Get batch sizes and times for bootstrap analysis
            batch_sizes = np.array(group['batch_size'])
            times_bs = np.stack(group['last_crossing_bootstrap'])

            # Find best batch sizes
            times_bs_inf = np.where(np.isnan(times_bs), np.inf, times_bs)
            best_bs_bootstrap = batch_sizes[np.argmin(times_bs_inf, axis=0)]
            best_times_bootstrap = np.min(times_bs_inf, axis=0)

            # Get point estimates
            try:
                best_bs = group.loc[group['last_crossing'].idxmin(skipna=True), 'batch_size']
                min_time = group['last_crossing'].min()
            except:
                best_bs = float('nan')
                min_time = float('inf')

            results.append({
                'env_name': env,
                'utd': utd,
                'learning_rate': lr,
                'best_bs': best_bs,
                'time_to_threshold': min_time,
                'best_bs_bootstrap': best_bs_bootstrap,
                'time_to_threshold_bootstrap': best_times_bootstrap,
            })

    # Create DataFrame
    df_best_bs = pd.DataFrame(results)

    if print_pivot:
        # Pivot table to get environments as rows and utd x learning_rate as columns
        pivot_df = df_best_bs.pivot_table(
            index='utd',
            columns=['env_name', 'learning_rate'],
            values='best_bs',
            aggfunc='first'
        )

        # Print formatted table with scientific notation
        pd.set_option('display.float_format', '{:.1e}'.format)
        print("\nBest Batch Sizes:")
        print(pivot_df.to_string())

        # Create pivot table for uncertainty-corrected best batch sizes
        # Use mean of bootstrap samples as uncertainty-corrected estimate
        pivot_df_bootstrap = df_best_bs.pivot_table(
            index='utd',
            columns=['env_name', 'learning_rate'],
            values='best_bs_bootstrap',
            aggfunc=lambda x: np.mean(np.stack(x))
        )

        # Print formatted table with scientific notation
        print("\nUncertainty-Corrected Best Batch Sizes:")
        print(pivot_df_bootstrap.to_string())

    return df_best_bs


def get_bootstrap_optimal(group):
    """Get bootstrapped optimal batch sizes."""
    # Get time to threshold bootstrap array for all batch sizes
    batch_sizes = group['batch_size'].values
    lr_bootstrap = np.stack(group['best_lr_bootstrap'].values) # 100 bootstrap samples all have different optimal learning rates
    times_bootstrap = np.stack(group['time_to_threshold_bootstrap'].values) # 100 times to threshold corresponding to different bootstrap samples
    
    # Find optimal batch size index for each bootstrap sample
    # Replace nans with large values so they are never selected as minimum
    times_bootstrap = np.nan_to_num(times_bootstrap, nan=np.inf)
    optimal_indices_bootstrap = np.argmin(times_bootstrap, axis=0) # 100 indices of optimal batch sizes
    best_lr_bootstrap = lr_bootstrap[optimal_indices_bootstrap, np.arange(times_bootstrap.shape[1])] # for each bootstrap sample, get the learning rate corresponding to the optimal batch size
    best_times_bootstrap = times_bootstrap[optimal_indices_bootstrap, np.arange(times_bootstrap.shape[1])] # for each bootstrap sample, get the time to threshold corresponding to the optimal batch size
    best_bs_bootstrap = batch_sizes[optimal_indices_bootstrap] # for each bootstrap sample, get the optimal batch size
    
    # Get point estimate
    best_idx = group['time_to_threshold'].argmin()
    best_bs = group.iloc[best_idx]['batch_size']
    best_lr = group.iloc[best_idx]['best_lr']
    best_time = group.iloc[best_idx]['time_to_threshold']
    
    return pd.Series({
        'best_lr': best_lr,
        'best_bs': best_bs,
        'time_to_threshold': best_time,
        'best_lr_bootstrap': best_lr_bootstrap,
        'best_bs_bootstrap': best_bs_bootstrap,
        'time_to_threshold_bootstrap': best_times_bootstrap,
    })
    
    
def compute_bootstrap_averages(df_best_lr, df_best_bs, df_best_lr_bs):
    """Get optimal bs and lr averaged across lr and bs, respectively."""

    for env in df_best_lr_bs['env_name'].unique():
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
        mean_bs_all = np.mean(best_bs_bootstrap, axis=(1,2))
        std_bs_all = np.std(best_bs_bootstrap, axis=(1,2))
        df_best_lr_bs.loc[env_mask, 'best_bs_bootstrap_lrmean'] = [mean_bs_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']]
        df_best_lr_bs.loc[env_mask, 'best_bs_bootstrap_lrmean_std'] = [std_bs_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']]
        
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
        mean_lr_all = np.mean(best_lr_bootstrap, axis=(1,2))
        std_lr_all = np.std(best_lr_bootstrap, axis=(1,2))
        df_best_lr_bs.loc[env_mask, 'best_lr_bootstrap_bsmean'] = [mean_lr_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']]
        df_best_lr_bs.loc[env_mask, 'best_lr_bootstrap_bsmean_std'] = [std_lr_all[list(utd_groups.groups.keys()).index(utd)] for utd in env_data['utd']]
        
    return df_best_lr_bs


def plot_bootstrap_average_params(df_best_lr_bs):
    # plot hparams and performance for optimal bs, lr
    envs = df_best_lr_bs['env_name'].unique()
    n_envs = len(envs)

    # Create first figure for batch size plots
    n_cols = 4
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharey=True)
    axs1 = axs1.flatten()

    # Plot 1: Optimal batch size vs UTD with bootstrap CIs and mean optimal batch size
    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
        
        # Calculate correlation for point estimate
        point_bs_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs']))[0,1]
        axs1[i].plot(env_data['utd'], env_data['best_bs'], 'o-', label=f'Point estimate (corr={point_bs_corr:.3f})')
        
        # Add bootstrapped confidence intervals
        bootstrap_bs_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs_bootstrap_mean']))[0,1]
        axs1[i].errorbar(env_data['utd'], env_data['best_bs_bootstrap_mean'], 
                        yerr=env_data['best_bs_bootstrap_std'],
                        fmt='o-', capsize=5, alpha=0.4,
                        label=f'Bootstrap (corr={bootstrap_bs_corr:.3f})')
        
        # Add mean optimal batch size
        bs_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs_lrmean']))[0,1]
        axs1[i].errorbar(env_data['utd'], env_data['best_bs_lrmean'],
                        yerr=env_data['best_bs_lrmean_std'], 
                        fmt='o-', capsize=5, alpha=0.4,
                        label=f'Mean across LRs (corr={bs_corr:.3f})')
        
        # Add mean batch size averaged across learning rates and bootstrap intervals
        bs_corr_all = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_bs_bootstrap_lrmean']))[0,1]
        axs1[i].errorbar(env_data['utd'], env_data['best_bs_bootstrap_lrmean'],
                        yerr=env_data['best_bs_bootstrap_lrmean_std'],
                        fmt='o-', capsize=5, alpha=0.4,
                        label=f'Bootstrap Mean across LRs (corr={bs_corr_all:.3f})')
        
        axs1[i].set_xscale('log')
        axs1[i].set_yscale('log')
        axs1[i].set_xlabel('UTD')
        axs1[i].set_ylabel('Optimal Batch Size')
        axs1[i].set_title(f'{env}')
        axs1[i].grid(True)
        axs1[i].legend()

    # Remove empty subplots from first figure
    for j in range(i+1, len(axs1)):
        fig1.delaxes(axs1[j])

    plt.tight_layout()
    plt.show()

    # Create second figure for learning rate plots
    fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharey=True)
    axs2 = axs2.flatten()

    # Plot 2: Optimal learning rate vs UTD with bootstrap CIs and mean optimal learning rate
    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
        
        # Calculate correlation for point estimate
        point_lr_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr']))[0,1]
        axs2[i].plot(env_data['utd'], env_data['best_lr'], 'o-', label=f'Point estimate (corr={point_lr_corr:.3f})')
        
        # Add bootstrapped confidence intervals
        bootstrap_lr_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr_bootstrap_mean']))[0,1]
        axs2[i].errorbar(env_data['utd'], env_data['best_lr_bootstrap_mean'],
                        yerr=env_data['best_lr_bootstrap_std'],
                        fmt='o-', capsize=5, alpha=0.4,
                        label=f'Bootstrap (corr={bootstrap_lr_corr:.3f})')
        
        # Add mean optimal learning rate
        lr_corr = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr_bsmean']))[0,1]
        axs2[i].errorbar(env_data['utd'], env_data['best_lr_bsmean'],
                        yerr=env_data['best_lr_bsmean_std'],
                        fmt='o-', capsize=5, alpha=0.4,
                        label=f'Mean across BSs (corr={lr_corr:.3f})')
        
        # Add mean learning rate averaged across batch sizes and bootstrap intervals
        lr_corr_all = np.corrcoef(np.log10(env_data['utd']), np.log10(env_data['best_lr_bootstrap_bsmean']))[0,1]
        axs2[i].errorbar(env_data['utd'], env_data['best_lr_bootstrap_bsmean'],
                        yerr=env_data['best_lr_bootstrap_bsmean_std'],
                        fmt='o-', capsize=5, alpha=0.4,
                        label=f'Bootstrap Mean across BSs (corr={lr_corr_all:.3f})')
        
        axs2[i].set_xscale('log')
        axs2[i].set_yscale('log')
        axs2[i].set_xlabel('UTD')
        axs2[i].set_ylabel('Optimal Learning Rate')
        axs2[i].set_title(f'{env}\nCorr: {lr_corr:.3f}')
        axs2[i].grid(True)
        axs2[i].legend()

    # Remove empty subplots from second figure
    for j in range(i+1, len(axs2)):
        fig2.delaxes(axs2[j])

    plt.tight_layout()
    plt.show()


def linear_fit_separate(utds_to_predict, df_best_lr_bs, plot=False):
    """
    Plot linear fit for bootstrap averaged learning rate and batch size
    with a seprate slope per environment.
    """
    
    # Make linear fit for bs and lr
    envs = df_best_lr_bs['env_name'].unique()
    envs = [x for x in envs if 'merged' not in x]
    n_envs = len(envs)

    if plot:
        # Create first figure for batch size plots
        n_cols = 4
        n_rows = (n_envs + n_cols - 1) // n_cols
        fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True)
        axs1 = axs1.flatten()
        
    # Create table to store proposed values
    proposed_lr_values = {
        "Environment": [],
        "UTD": [],
        "Learning Rate": [],
        "Learning Rate x√2": [],
        "Learning Rate x√0.5": [],
    }

    # Plot 1: Optimal batch size vs UTD with bootstrap CIs and mean optimal batch size
    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
        
        # Fit linear regression in log-log space
        X = np.log10(env_data['utd'].values[:-1]).reshape(-1, 1)
        y = np.log10(env_data['best_bs_bootstrap_lrmean'].values[:-1])
        reg = LinearRegression().fit(X, y)
        
        # Plot the fit line
        utds = np.array(utds_to_predict)
        predictions = 10**(reg.predict(np.log10(utds).reshape(-1, 1)))

        # Store values in table
        for utd, pred in zip(utds, predictions):
            proposed_lr_values["Environment"].append(env)
            proposed_lr_values["UTD"].append(f"{utd.item():.2f}")
            proposed_lr_values["Learning Rate"].append(f"{pred.item():.2e}")
            proposed_lr_values["Learning Rate x√2"].append(f"{pred.item() * np.sqrt(2):.2e}")
            proposed_lr_values["Learning Rate x√0.5"].append(f"{pred.item() * np.sqrt(0.5):.2e}")
        
        if plot:
            axs1[i].errorbar(env_data['utd'], env_data['best_bs_bootstrap_lrmean'],
                            yerr=env_data['best_bs_bootstrap_lrmean_std'],
                            fmt='o-', capsize=5, alpha=0.4)
            axs1[i].plot(utds, predictions, label=f'Slope={reg.coef_[0]:.2f}, R²={reg.score(X,y):.2f}', color='black')
            axs1[i].set_xscale('log')
            axs1[i].set_yscale('log')
            axs1[i].set_xlabel('UTD')
            if i == 0:
                axs1[i].set_ylabel('Optimal Batch Size')
            axs1[i].set_title(f'{env}')
            axs1[i].grid(True)
            axs1[i].legend()

    if plot:
        # Remove empty subplots from first figure
        for j in range(i+1, len(axs1)):
            fig1.delaxes(axs1[j])

        plt.tight_layout()
        plt.show()

    if plot:
        # Create second figure for learning rate plots
        fig2, axs2 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True)
        axs2 = axs2.flatten()
        
    proposed_bs_values = {
        "Environment": [],
        "UTD": [],
        "Batch Size": [],
        "Batch Size x√2": [],
        "Batch Size x√0.5": [],
    }

    # Plot 2: Optimal learning rate vs UTD with bootstrap CIs and mean optimal learning rate
    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
        
        # Fit linear regression in log-log space
        X = np.log10(env_data['utd'].values[:-1]).reshape(-1, 1)
        y = np.log10(env_data['best_lr_bootstrap_bsmean'].values[:-1])
        reg = LinearRegression().fit(X, y)
        
        # Plot the fit line
        utds = np.array(utds_to_predict)
        predictions = 10**(reg.predict(np.log10(utds).reshape(-1, 1)))

        # Store values in table
        for utd, pred in zip(utds, predictions):
            proposed_bs_values["Environment"].append(env)
            proposed_bs_values["UTD"].append(f"{utd.item():.2f}")
            proposed_bs_values["Batch Size"].append(f"{pred.item():.0f}")
            proposed_bs_values["Batch Size x√2"].append(f"{pred.item() * np.sqrt(2):.0f}")
            proposed_bs_values["Batch Size x√0.5"].append(f"{pred.item() * np.sqrt(0.5):.0f}")

        if plot:
            axs2[i].errorbar(env_data['utd'], env_data['best_lr_bootstrap_bsmean'],
                            yerr=env_data['best_lr_bootstrap_bsmean_std'],
                            fmt='o-', capsize=5, alpha=0.4)
            axs2[i].plot(utds, predictions, label=f'Slope={reg.coef_[0]:.2f}, R²={reg.score(X,y):.2f}', color='black')
            axs2[i].set_xscale('log')
            axs2[i].set_yscale('log')
            axs2[i].set_xlabel('UTD')
            if i == 0:
                axs2[i].set_ylabel('Optimal Learning Rate')
            axs2[i].set_title(f'{env}')
            axs2[i].grid(True)
            axs2[i].legend()
            
    if plot:
        # Remove empty subplots from second figure
        for j in range(i+1, len(axs2)):
            fig2.delaxes(axs2[j])

        plt.tight_layout()
        plt.show()


def _fit_shared_slope_regression(df, envs, x_col='utd', y_col='best_lr_bootstrap_bsmean'):
    """
    Plot linear fit for bootstrap averaged learning rate and batch size
    with a shared slope per environment.
    """
    # Get data for all environments
    X_all = []
    y_all = []
    env_indices = []

    for i, env in enumerate(envs):
        env_data = df[df['env_name'] == env]    
        X_all.append(np.log10(env_data[x_col].values[:]))
        y_all.append(np.log10(env_data[y_col].values[:]))
        env_indices.extend([i] * len(env_data))
        
    X_all = np.concatenate(X_all).reshape(-1, 1)
    y_all = np.concatenate(y_all)
    env_indices = np.array(env_indices)

    # Create dummy variables for environment intercepts
    n_envs = len(envs)
    env_dummies = np.zeros((len(X_all), n_envs))
    for i in range(n_envs):
        env_dummies[env_indices == i, i] = 1

    # Combine UTD and environment dummies
    X_combined = np.hstack([X_all, env_dummies[:, 1:]])  # Drop first dummy to avoid collinearity

    # Fit regression
    reg_shared = LinearRegression(fit_intercept=True).fit(X_combined, y_all)

    # Extract shared slope and environment-specific intercepts
    shared_slope = reg_shared.coef_[0]
    env_intercepts = np.zeros(n_envs)
    env_intercepts[0] = reg_shared.intercept_
    env_intercepts[1:] = reg_shared.intercept_ + reg_shared.coef_[1:]
    
    return reg_shared, X_all, y_all, env_indices, n_envs, shared_slope, env_intercepts


def linear_fit_shared(utds_to_predict, df, df_best_lr_bs, envs, path, plot=False):
    utds_to_predict = np.array(utds_to_predict, dtype=float)
    n_envs = len(envs)

    # Best learning rate fit
    (
        reg_shared,
        X_all,
        y_all,
        env_indices,
        n_envs,
        lr_shared_slope,
        lr_env_intercepts,
    ) = _fit_shared_slope_regression(df_best_lr_bs, envs)

    if plot:
        # Create figures
        n_cols = 4
        n_rows = (n_envs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharex=True)
        axes = axes.flatten()

    # Create table to store proposed values
    proposed_lr_values = {
        "Environment": [],
        "UTD": [],
        "Learning Rate": [],
        "Learning Rate x√2": [],
        "Learning Rate x√0.5": [],
    }

    for i, env in enumerate(envs):
        print(f"{env}: lr ~ {10 ** lr_env_intercepts[i]:.6f} * UTD^{lr_shared_slope:.6f}")
        
        # Get predictions
        X_plot = np.linspace(np.log10(utds_to_predict.min()), np.log10(utds_to_predict.max()), 100).reshape(-1, 1)
        env_dummies_plot = np.zeros((len(X_plot), n_envs - 1))
        if i > 0:
            env_dummies_plot[:, i - 1] = 1
        X_combined_plot = np.hstack([X_plot, env_dummies_plot])
        y_plot = reg_shared.predict(X_combined_plot)

        utds = utds_to_predict.reshape(-1, 1)
        predictions = 10 ** y_plot[np.searchsorted(10 ** X_plot.flatten(), utds)]

        # Store values in table
        for utd, pred in zip(utds, predictions):
            proposed_lr_values["Environment"].append(env)
            proposed_lr_values["UTD"].append(f"{utd.item():.2f}")
            proposed_lr_values["Learning Rate"].append(f"{pred.item():.2e}")
            proposed_lr_values["Learning Rate x√2"].append(f"{pred.item() * np.sqrt(2):.2e}")
            proposed_lr_values["Learning Rate x√0.5"].append(f"{pred.item() * np.sqrt(0.5):.2e}")

        if plot:            
            mask = env_indices == i
            axes[i].scatter(10 ** X_all[mask], 10 ** y_all[mask], label="Bootstrap estimates", alpha=0.6)

            # Plot regression line
            axes[i].plot(10**X_plot, 10**y_plot, "-", color="black", label=f"Fit")

            # Plot x1.4 and x0.7 lines for each UTD point
            axes[i].scatter(utds, predictions * np.sqrt(2), marker="o", color="black", alpha=0.3)
            axes[i].scatter(utds, predictions * np.sqrt(0.5), marker="o", color="black", alpha=0.3)
            axes[i].scatter(utds, predictions, marker="o", color="black", alpha=0.6, label="Proposed values")

            # plot possible values of learning rate as lines
            utd_values = df[df["env_name"] == env]["utd"].unique()
            lr_values = df[df["env_name"] == env]["learning_rate"].unique()

            # Plot vertical lines for UTD values
            for utd in utd_values:
                axes[i].plot(
                    [utd, utd],
                    [min(lr_values), max(lr_values)],
                    color="black",
                    alpha=0.2,
                    linestyle="--",
                    label="Grid search values" if utd == utd_values[0] else None,
                )

            # Plot horizontal lines for learning rate values
            for lr in lr_values:
                axes[i].plot(
                    [min(utd_values), max(utd_values)],
                    [lr, lr],
                    color="black",
                    alpha=0.2,
                    linestyle="--",
                )

            axes[i].set_xlabel("UTD")
            axes[i].set_title(env)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale("log")
            axes[i].set_yscale("log")
            axes[i].yaxis.set_major_locator(plt.FixedLocator(lr_values))
            axes[i].yaxis.set_minor_locator(plt.NullLocator())
            axes[i].set_xticks(utds_to_predict, utds_to_predict)
            axes[i].set_yticks(lr_values, lr_values)
            if i % n_cols == 0:
                axes[i].set_ylabel(r"$\eta^*$: Best learning rate")

    if plot:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Create a single legend for all plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.show()

    # Best batch size fit
    (
        reg_shared,
        X_all,
        y_all,
        env_indices,
        n_envs,
        bs_shared_slope,
        bs_env_intercepts,
    ) = _fit_shared_slope_regression(df_best_lr_bs, envs, y_col="best_bs_bootstrap_lrmean")
    
    if plot:
        # Create figures
        n_cols = 4
        n_rows = (n_envs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharey=True)
        axes = axes.flatten()

    # Create table to store proposed batch size values
    proposed_bs_values = {
        "Environment": [],
        "UTD": [],
        "Batch Size": [],
        "Batch Size x√2": [],
        "Batch Size x√0.5": [],
    }

    for i, env in enumerate(envs):
        print(f"{env}: batch size ~ {10 ** bs_env_intercepts[i]:.6f} * UTD^{bs_shared_slope:.6f}")
        
        X_plot = np.linspace(np.log10(utds_to_predict.min()), np.log10(utds_to_predict.max()), 100).reshape(-1, 1)
        env_dummies_plot = np.zeros((len(X_plot), n_envs - 1))
        if i > 0:
            env_dummies_plot[:, i - 1] = 1
        X_combined_plot = np.hstack([X_plot, env_dummies_plot])
        y_plot = reg_shared.predict(X_combined_plot)

        # Plot data points for this environment
        utds = utds_to_predict.reshape(-1, 1)
        predictions = 10 ** y_plot[np.searchsorted(10 ** X_plot.flatten(), utds)]

        # Store values in table
        for utd, pred in zip(utds, predictions):
            proposed_bs_values["Environment"].append(env)
            proposed_bs_values["UTD"].append(f"{utd.item():.2f}")
            proposed_bs_values["Batch Size"].append(f"{pred.item():.0f}")
            proposed_bs_values["Batch Size x√2"].append(f"{pred.item() * np.sqrt(2):.0f}")
            proposed_bs_values["Batch Size x√0.5"].append(f"{pred.item() * np.sqrt(0.5):.0f}")

        if plot:
            mask = env_indices == i
            axes[i].scatter(10 ** X_all[mask], 10 ** y_all[mask], label="Bootstrap estimates", alpha=0.6)

            # Plot regression line
            axes[i].plot(10**X_plot, 10**y_plot, "-", color="black", label=f"Fit")

            # Plot x1.4 and x0.7 lines for each UTD point
            axes[i].scatter(utds, predictions * np.sqrt(2), marker="o", color="black", alpha=0.3)
            axes[i].scatter(utds, predictions * np.sqrt(0.5), marker="o", color="black", alpha=0.3)
            axes[i].scatter(utds, predictions, marker="o", color="black", alpha=0.6, label="Proposed values")

            # plot possible values of batch size as lines
            utd_values = df[df["env_name"] == env]["utd"].unique()
            bs_values = df[df["env_name"] == env]["batch_size"].unique()

            # Plot vertical lines for UTD values
            for utd in utd_values:
                axes[i].plot(
                    [utd, utd],
                    [min(bs_values), max(bs_values)],
                    color="black",
                    alpha=0.2,
                    linestyle="--",
                    label="Grid search values" if utd == utd_values[0] else None,
                )

            # Plot horizontal lines for batch size values
            for bs in bs_values:
                axes[i].plot(
                    [min(utd_values), max(utd_values)],
                    [bs, bs],
                    color="black",
                    alpha=0.2,
                    linestyle="--",
                )

            axes[i].set_xlabel("UTD")
            axes[i].set_title(env)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xscale("log")
            axes[i].set_yscale("log")
            axes[i].set_xticks(utds_to_predict, utds_to_predict)
            axes[i].yaxis.set_major_locator(plt.FixedLocator(bs_values))
            axes[i].yaxis.set_minor_locator(plt.NullLocator())
            axes[i].set_yticks(bs_values, bs_values)
            if i % n_cols == 0:
                axes[i].set_ylabel(r"$B^*$: Best batch size")

    if plot:
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        # Create a single legend for all plots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0)
        plt.tight_layout()
        plt.show()
        
    lr_export = np.column_stack([np.full(n_envs, lr_shared_slope), lr_env_intercepts])
    bs_export = np.column_stack([np.full(n_envs, bs_shared_slope), bs_env_intercepts])
    np.save(os.path.join(outputs_dir, f'grid_proposed_fits/{path}_lr_fit.npy'), lr_export)
    np.save(os.path.join(outputs_dir, f'grid_proposed_fits/{path}_bs_fit.npy'), bs_export)

    return (
        proposed_lr_values,
        proposed_bs_values,
        lr_shared_slope,
        lr_env_intercepts,
        bs_shared_slope,
        bs_env_intercepts,
    )


def tabulate_proposed_params(envs, utds_to_predict, proposed_lr_values, proposed_bs_values, path, verbose=False):
    utds_to_predict = sorted(utds_to_predict)

    # Display merged table of proposed values
    if verbose:
        print("\nProposed Values:")
        print("-" * 160)
        print(f"{'Environment':<30} {'UTD':<10} {'Learning Rate':<15} {'Learning Rate x√2':<15} {'Learning Rate x√0.5':<15} {'Batch Size':<15} {'Batch Size x√2':<15} {'Batch Size x√0.5':<15}")
        print("-" * 160)

    proposed_values_formatted = []

    for env in envs:
        for utd in utds_to_predict:
            utd = f"{utd:.2f}"
            # Find indices for this env/UTD combination
            lr_idx = next(
                (
                    i for i in range(len(proposed_lr_values["Environment"]))
                    if proposed_lr_values["Environment"][i] == env and proposed_lr_values["UTD"][i] == utd
                ),
                None,
            )
            bs_idx = next(
                (
                    i for i in range(len(proposed_bs_values["Environment"]))
                    if proposed_bs_values["Environment"][i] == env and proposed_bs_values["UTD"][i] == utd
                ),
                None,
            )

            if lr_idx is not None and bs_idx is not None:
                if verbose:
                    print(
                        f"{env:<30} {utd:<10} "
                        f"{proposed_lr_values['Learning Rate'][lr_idx]:<15} "
                        f"{proposed_lr_values['Learning Rate x√2'][lr_idx]:<15} "
                        f"{proposed_lr_values['Learning Rate x√0.5'][lr_idx]:<15} "
                        f"{proposed_bs_values['Batch Size'][bs_idx]:<15} "
                        f"{proposed_bs_values['Batch Size x√2'][bs_idx]:<15} "
                        f"{proposed_bs_values['Batch Size x√0.5'][bs_idx]:<15}"
                    )

                proposed_values_formatted.append(
                    {
                        "Environment": env,
                        "UTD": utd,
                        "Learning Rate": proposed_lr_values["Learning Rate"][lr_idx],
                        "Learning Rate x√2": proposed_lr_values["Learning Rate x√2"][lr_idx],
                        "Learning Rate x√0.5": proposed_lr_values["Learning Rate x√0.5"][lr_idx],
                        "Batch Size": proposed_bs_values["Batch Size"][bs_idx],
                        "Batch Size x√2": proposed_bs_values["Batch Size x√2"][bs_idx],
                        "Batch Size x√0.5": proposed_bs_values["Batch Size x√0.5"][bs_idx],
                    }
                )
                
    proposed_values_df = pd.DataFrame(proposed_values_formatted).astype(
        {
            "Environment": str,
            "UTD": float,
            "Learning Rate": float,
            "Learning Rate x√2": float,
            "Learning Rate x√0.5": float,
            "Batch Size": int,
            "Batch Size x√2": int,
            "Batch Size x√0.5": int,
        }
    )

    for c in proposed_values_df.columns:
        if "Batch Size" in c:
            proposed_values_df[f"{c}(rounded)"] = (np.round(proposed_values_df[c] / 16) * 16).astype(int)

    outfile = os.path.join(outputs_dir, 'grid_proposed_hparams', f'{path}_fitted.csv')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    proposed_values_df.to_csv(outfile, index=False)
    return proposed_values_df


def tabulate_baseline_params(df, utds, utds_to_predict, n_envs, path):
    """
    For a fixed UTD (geometric mean among predicted UTDs), find the best
    (batch size, learning rate) pair. Then run this across many UTDs. The
    output csv details additional experiments to be run.
    """
    middle_utd = np.prod(utds_to_predict) ** (1/len(utds_to_predict))  # Geometric mean of predicted UTDs
    snap_utd = min(utds, key=lambda x: abs(np.log(x) - np.log(middle_utd)))  # Snap to nearest UTD in grid search
    print('Baseline based on UTD', snap_utd)
    
    utd_data = df[df['utd'] == snap_utd]
    utd_data['last_crossing'] = utd_data['crossings'].apply(lambda x: x[-1])
    idx = utd_data.groupby(['env_name'])['last_crossing'].idxmin()
    baseline_values_df = utd_data.loc[idx]
    
    res_dict = {}
    for _, row in baseline_values_df.iterrows():
        res_dict[row['env_name']] = {'batch_size': row['batch_size'], 'learning_rate': row['learning_rate']}
    
    baseline_values_df['utd'] = [utds_to_predict] * n_envs
    baseline_values_df = baseline_values_df.explode('utd').reset_index(drop=True)
    baseline_values_df = baseline_values_df[['env_name', 'utd', 'learning_rate', 'batch_size']].rename(columns={
        'env_name': 'Environment', 
        'utd': 'UTD', 
        'learning_rate': 'Learning Rate', 
        'batch_size': 'Batch Size'
    })

    hparam_dir = os.path.join(outputs_dir, 'grid_proposed_hparams')
    os.makedirs(hparam_dir, exist_ok=True)
    base_fname = f'{path}_baseline_utd{snap_utd}'
    # baseline_values_df.query(f'UTD in {utds}').to_csv(f'{hparam_dir}/{base_fname}_existing.csv', index=False)
    baseline_values_df.query(f'UTD not in {utds}').to_csv(f'{hparam_dir}/{base_fname}_new.csv', index=False)
    
    return baseline_values_df
