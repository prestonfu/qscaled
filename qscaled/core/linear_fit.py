import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from qscaled.core.preprocessing import get_envs, get_utds, get_batch_sizes, get_learning_rates

# TODO: clean up this whole file
# one method for each type of fit (should just depend on df_best_lr_bs?)
# one method for plotting either lr/bs given the fit (should work for both fits; here you pass in utds_to_predict)
# one method for plotting lr given the fit
# one method for plotting bs given the fit

def linear_fit_separate(utds_to_predict, df, df_best_lr_bs, outputs_dir=None, save_path=None, plot=False):
    """
    Plot linear fit for bootstrap averaged learning rate and batch size
    with a seprate slope per environment.
    """
    utds = get_utds(df)
    envs = get_envs(df)
    n_envs = len(envs)

    print('Separate slope fits:')

    # Plot 1: learning rates
    if plot:
        n_cols = 4
        n_rows = (n_envs + n_cols - 1) // n_cols
        fig_lr, axes_lr = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharey=True)
        axes_lr = axes_lr.flatten()
        r2_str = r'$R^2$'

    proposed_lr_values = {
        'Environment': [],
        'UTD': [],
        'Batch Size': [],
        'Batch Size x√2': [],
        'Batch Size x√0.5': [],
    }
    
    lr_env_slopes = []
    lr_env_intercepts = []

    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]

        # Fit linear regression in log-log space
        X = np.log10(env_data['utd'].values[:-1]).reshape(-1, 1)
        y = np.log10(env_data['best_lr_bootstrap_bsmean'].values[:-1])
        reg = LinearRegression().fit(X, y)
        env_slope = reg.coef_[0]
        env_intercept = reg.intercept_
        lr_env_slopes.append(env_slope)
        lr_env_intercepts.append(env_intercept)
        print(f'{env}: learning rate ~ {10 ** env_intercept:.6f} * UTD^{env_slope:.6f}')

        # Plot the fit line
        utds_to_predict = np.array(utds_to_predict)
        predictions = 10 ** (reg.predict(np.log10(utds_to_predict).reshape(-1, 1)))
        subset_predictions = 10 ** (reg.predict(np.log10(utds).reshape(-1, 1)))

        # Store values in table
        for utd, pred in zip(utds_to_predict, predictions):
            proposed_lr_values['Environment'].append(env)
            proposed_lr_values['UTD'].append(f'{utd.item():.2f}')
            proposed_lr_values['Batch Size'].append(f'{pred.item():.0f}')
            proposed_lr_values['Batch Size x√2'].append(f'{pred.item() * np.sqrt(2):.0f}')
            proposed_lr_values['Batch Size x√0.5'].append(f'{pred.item() * np.sqrt(0.5):.0f}')

        if plot:
            axes_lr[i].scatter(env_data['utd'], env_data['best_lr_bootstrap_bsmean'], label='Bootstrap estimates', color='black')
            axes_lr[i].plot(utds, subset_predictions, label='Fit', color='black')
            axes_lr[i].scatter(utds_to_predict, predictions, marker='o', color='lightblue', label='Proposed values')
            axes_lr[i].scatter(utds_to_predict, predictions * np.sqrt(2), marker='o', color='lightblue')
            axes_lr[i].scatter(utds_to_predict, predictions * np.sqrt(0.5), marker='o', color='lightblue')
            
            # plot possible values of learning rate as lines
            utd_values = df[df['env_name'] == env]['utd'].unique()
            lr_values = df[df['env_name'] == env]['learning_rate'].unique()
            
            # Plot vertical lines for UTD values
            for utd in utd_values:
                axes_lr[i].plot(
                    [utd, utd],
                    [min(lr_values), max(lr_values)],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                    label='Grid search values' if utd == utd_values[0] else None,
                )

            # Plot horizontal lines for learning rate values
            for lr in lr_values:
                axes_lr[i].plot(
                    [min(utd_values), max(utd_values)],
                    [lr, lr],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                )

            axes_lr[i].set_xscale('log')
            axes_lr[i].set_yscale('log')
            axes_lr[i].set_xlabel('UTD')
            axes_lr[i].set_title(f"{env}\nSlope={reg.coef_[0]:.2f}, {r2_str}={reg.score(X,y):.2f}")
            axes_lr[i].grid(True)

    if plot:
        # Remove empty subplots from second figure
        for j in range(i + 1, len(axes_lr)):
            fig_lr.delaxes(axes_lr[j])

        plt.suptitle(r'$\eta^*$: Best learning rate')
        plt.tight_layout()
        plt.show()
    else:
        print()

    
    # Plot 2: batch sizes
    if plot:
        fig_bs, axes_bs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=True)
        axes_bs = axes_bs.flatten()

    # Create table to store proposed values
    proposed_bs_values = {
        'Environment': [],
        'UTD': [],
        'Learning Rate': [],
        'Learning Rate x√2': [],
        'Learning Rate x√0.5': [],
    }
    
    bs_env_slopes = []
    bs_env_intercepts = []

    for i, env in enumerate(envs):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]

        # Fit linear regression in log-log space
        X = np.log10(env_data['utd'].values[:-1]).reshape(-1, 1)
        y = np.log10(env_data['best_bs_bootstrap_lrmean'].values[:-1])
        reg = LinearRegression().fit(X, y)
        env_slope = reg.coef_[0]
        env_intercept = reg.intercept_
        bs_env_slopes.append(env_slope)
        bs_env_intercepts.append(env_intercept)
        print(f'{env}: batch size ~ {10 ** env_intercept:.6f} * UTD^{env_slope:.6f}')

        # Plot the fit line
        utds_to_predict = np.array(utds_to_predict)
        predictions = 10 ** (reg.predict(np.log10(utds_to_predict).reshape(-1, 1)))
        subset_predictions = 10 ** (reg.predict(np.log10(utds).reshape(-1, 1)))

        # Store values in table
        for utd, pred in zip(utds_to_predict, predictions):
            proposed_bs_values['Environment'].append(env)
            proposed_bs_values['UTD'].append(f'{utd.item():.2f}')
            proposed_bs_values['Learning Rate'].append(f'{pred.item():.2e}')
            proposed_bs_values['Learning Rate x√2'].append(f'{pred.item() * np.sqrt(2):.2e}')
            proposed_bs_values['Learning Rate x√0.5'].append(f'{pred.item() * np.sqrt(0.5):.2e}')

        if plot:
            axes_bs[i].scatter(env_data['utd'], env_data['best_bs_bootstrap_lrmean'], label='Bootstrap estimates', color='black')
            axes_bs[i].plot(utds, subset_predictions, label='Fit', color='black')
            axes_bs[i].scatter(utds_to_predict, predictions, marker='o', color='lightblue', label='Proposed values')
            axes_bs[i].scatter(utds_to_predict, predictions * np.sqrt(2), marker='o', color='lightblue')
            axes_bs[i].scatter(utds_to_predict, predictions * np.sqrt(0.5), marker='o', color='lightblue')
            
            # plot possible values of batch size as lines
            utd_values = df[df['env_name'] == env]['utd'].unique()
            bs_values = df[df['env_name'] == env]['batch_size'].unique()

            # Plot vertical lines for UTD values
            for utd in utd_values:
                axes_bs[i].plot(
                    [utd, utd],
                    [min(bs_values), max(bs_values)],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                    label='Grid search values' if utd == utd_values[0] else None,
                )

            # Plot horizontal lines for batch size values
            for bs in bs_values:
                axes_bs[i].plot(
                    [min(utd_values), max(utd_values)],
                    [bs, bs],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                )

            axes_bs[i].set_xscale('log')
            axes_bs[i].set_yscale('log')
            axes_bs[i].set_xlabel('UTD')
            axes_bs[i].set_title(f"{env}\nSlope={reg.coef_[0]:.2f}, {r2_str}={reg.score(X,y):.2f}")
            axes_bs[i].grid(True)

    if plot:
        # Remove empty subplots from first figure
        for j in range(i + 1, len(axes_bs)):
            fig_bs.delaxes(axes_bs[j])
        handles, labels = axes_bs[0].get_legend_handles_labels()
        fig_bs.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0, frameon=False)
        plt.suptitle(r'$B^*$: Best batch size')
        plt.tight_layout()
        plt.show()
    else:
        print()
        
    lr_export = np.column_stack([lr_env_slopes, lr_env_intercepts])
    bs_export = np.column_stack([bs_env_slopes, bs_env_intercepts])
    if save_path:
        os.makedirs(os.path.join(outputs_dir, 'grid_proposed_fits'), exist_ok=True)
        np.save(os.path.join(outputs_dir, f'grid_proposed_fits/{save_path}_lr_fit.npy'), lr_export)
        np.save(os.path.join(outputs_dir, f'grid_proposed_fits/{save_path}_bs_fit.npy'), bs_export)

    return (
        proposed_lr_values,
        proposed_bs_values,
        np.array(lr_env_slopes),
        np.array(lr_env_intercepts),
        np.array(bs_env_slopes),
        np.array(bs_env_intercepts)
    )


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


def _get_plot_predictions(reg_shared, utds, n_envs, i):
    X = np.log10(utds).reshape(-1, 1)
    env_dummies_plot = np.zeros((len(utds), n_envs - 1))
    if i > 0:
        env_dummies_plot[:, i - 1] = 1
    X_combined = np.hstack([X, env_dummies_plot])
    predictions = reg_shared.predict(X_combined)
    return 10 ** predictions


def linear_fit_shared(utds_to_predict, df, df_best_lr_bs, outputs_dir=None, save_path=None, plot=False):
    utds_to_predict = np.array(utds_to_predict)
    utds = np.array(get_utds(df))
    envs = get_envs(df)
    n_envs = len(envs)
        
    print('Shared slope fits:')

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
        n_cols = 4
        n_rows = (n_envs + n_cols - 1) // n_cols
        fig_lr, axes_lr = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharex=True)
        axes_lr = axes_lr.flatten()

    proposed_lr_values = {
        'Environment': [],
        'UTD': [],
        'Learning Rate': [],
        'Learning Rate x√2': [],
        'Learning Rate x√0.5': [],
    }

    for i, env in enumerate(envs):
        print(f'{env}: lr ~ {10 ** lr_env_intercepts[i]:.6f} * UTD^{lr_shared_slope:.6f}')
        predictions = _get_plot_predictions(reg_shared, utds_to_predict, n_envs, i)      
        predictions_subset = _get_plot_predictions(reg_shared, utds, n_envs, i)

        # Store values in table
        for utd, pred in zip(utds_to_predict, predictions):
            proposed_lr_values['Environment'].append(env)
            proposed_lr_values['UTD'].append(f'{utd.item():.2f}')
            proposed_lr_values['Learning Rate'].append(f'{pred.item():.2e}')
            proposed_lr_values['Learning Rate x√2'].append(f'{pred.item() * np.sqrt(2):.2e}')
            proposed_lr_values['Learning Rate x√0.5'].append(f'{pred.item() * np.sqrt(0.5):.2e}')

        if plot:
            mask = env_indices == i
            axes_lr[i].scatter(10 ** X_all[mask], 10 ** y_all[mask], label='Bootstrap estimates', color='black')

            # Plot regression line
            axes_lr[i].plot(utds, predictions_subset, '-', color='black', label=f'Fit')

            # Plot x1.4 and x0.7 lines for each UTD point
            axes_lr[i].scatter(utds_to_predict, predictions * np.sqrt(2), marker='o', color='lightblue')
            axes_lr[i].scatter(utds_to_predict, predictions * np.sqrt(0.5), marker='o', color='lightblue')
            axes_lr[i].scatter(utds_to_predict, predictions, marker='o', color='lightblue', label='Proposed values')

            # plot possible values of learning rate as lines
            utd_values = df[df['env_name'] == env]['utd'].unique()
            lr_values = df[df['env_name'] == env]['learning_rate'].unique()

            # Plot vertical lines for UTD values
            for utd in utd_values:
                axes_lr[i].plot(
                    [utd, utd],
                    [min(lr_values), max(lr_values)],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                    label='Grid search values' if utd == utd_values[0] else None,
                )

            # Plot horizontal lines for learning rate values
            for lr in lr_values:
                axes_lr[i].plot(
                    [min(utd_values), max(utd_values)],
                    [lr, lr],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                )

            axes_lr[i].set_xlabel('UTD')
            axes_lr[i].set_title(env)
            axes_lr[i].grid(True, alpha=0.3)
            axes_lr[i].set_xscale('log')
            axes_lr[i].set_yscale('log')
            axes_lr[i].yaxis.set_major_locator(plt.FixedLocator(lr_values))
            axes_lr[i].yaxis.set_minor_locator(plt.NullLocator())
            axes_lr[i].set_xticks(utds_to_predict, utds_to_predict)
            axes_lr[i].set_yticks(lr_values, lr_values)

    if plot:
        for j in range(i + 1, len(axes_lr)):
            fig_lr.delaxes(axes_lr[j])

        # Create a single legend for all plots
        handles, labels = axes_lr[0].get_legend_handles_labels()
        fig_lr.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0, frameon=False)
        plt.suptitle(r'$\eta^*$: Best learning rate')
        plt.tight_layout()
        plt.show()
    else:
        print()

    # Best batch size fit
    (
        reg_shared,
        X_all,
        y_all,
        env_indices,
        n_envs,
        bs_shared_slope,
        bs_env_intercepts,
    ) = _fit_shared_slope_regression(df_best_lr_bs, envs, y_col='best_bs_bootstrap_lrmean')
    
    if plot:
        # Create fig_bsures
        n_cols = 4
        n_rows = (n_envs + n_cols - 1) // n_cols
        fig_bs, axes_bs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows), sharey=True)
        axes_bs = axes_bs.flatten()

    # Create table to store proposed batch size values
    proposed_bs_values = {
        'Environment': [],
        'UTD': [],
        'Batch Size': [],
        'Batch Size x√2': [],
        'Batch Size x√0.5': [],
    }

    for i, env in enumerate(envs):
        print(f'{env}: batch size ~ {10 ** bs_env_intercepts[i]:.6f} * UTD^{bs_shared_slope:.6f}')
        predictions = _get_plot_predictions(reg_shared, utds_to_predict, n_envs, i)      
        predictions_subset = _get_plot_predictions(reg_shared, utds, n_envs, i)

        # Store values in table
        for utd, pred in zip(utds_to_predict, predictions):
            proposed_bs_values['Environment'].append(env)
            proposed_bs_values['UTD'].append(f'{utd.item():.2f}')
            proposed_bs_values['Batch Size'].append(f'{pred.item():.0f}')
            proposed_bs_values['Batch Size x√2'].append(f'{pred.item() * np.sqrt(2):.0f}')
            proposed_bs_values['Batch Size x√0.5'].append(f'{pred.item() * np.sqrt(0.5):.0f}')

        if plot:
            mask = env_indices == i
            axes_bs[i].scatter(10 ** X_all[mask], 10 ** y_all[mask], label='Bootstrap estimates', color='black')

            # Plot regression line
            axes_bs[i].plot(utds, predictions_subset, '-', color='black', label=f'Fit')

            # Plot x1.4 and x0.7 lines for each UTD point
            axes_bs[i].scatter(utds_to_predict, predictions * np.sqrt(2), marker='o', color='lightblue')
            axes_bs[i].scatter(utds_to_predict, predictions * np.sqrt(0.5), marker='o', color='lightblue')
            axes_bs[i].scatter(utds_to_predict, predictions, marker='o', color='lightblue', label='Proposed values')

            # plot possible values of batch size as lines
            utd_values = df[df['env_name'] == env]['utd'].unique()
            bs_values = df[df['env_name'] == env]['batch_size'].unique()

            # Plot vertical lines for UTD values
            for utd in utd_values:
                axes_bs[i].plot(
                    [utd, utd],
                    [min(bs_values), max(bs_values)],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                    label='Grid search values' if utd == utd_values[0] else None,
                )

            # Plot horizontal lines for batch size values
            for bs in bs_values:
                axes_bs[i].plot(
                    [min(utd_values), max(utd_values)],
                    [bs, bs],
                    color='black',
                    alpha=0.2,
                    linestyle='--',
                )

            axes_bs[i].set_xlabel('UTD')
            axes_bs[i].set_title(env)
            axes_bs[i].grid(True, alpha=0.3)
            axes_bs[i].set_xscale('log')
            axes_bs[i].set_yscale('log')
            axes_bs[i].set_xticks(utds_to_predict, utds_to_predict)
            axes_bs[i].yaxis.set_major_locator(plt.FixedLocator(bs_values))
            axes_bs[i].yaxis.set_minor_locator(plt.NullLocator())
            axes_bs[i].set_yticks(bs_values, bs_values)

    if plot:
        for j in range(i + 1, len(axes_bs)):
            fig_bs.delaxes(axes_bs[j])

        # Create a single legend for all plots
        handles, labels = axes_bs[0].get_legend_handles_labels()
        fig_bs.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
        plt.suptitle(r'$B^*$: Best batch size')
        plt.tight_layout()
        plt.show()
    else:
        print()

    lr_export = np.column_stack([np.full(n_envs, lr_shared_slope), lr_env_intercepts])
    bs_export = np.column_stack([np.full(n_envs, bs_shared_slope), bs_env_intercepts])
    if save_path:
        os.makedirs(os.path.join(outputs_dir, 'grid_proposed_fits'), exist_ok=True)
        np.save(os.path.join(outputs_dir, f'grid_proposed_fits/{save_path}_lr_fit.npy'), lr_export)
        np.save(os.path.join(outputs_dir, f'grid_proposed_fits/{save_path}_bs_fit.npy'), bs_export)

    return (
        proposed_lr_values,
        proposed_bs_values,
        lr_shared_slope,
        lr_env_intercepts,
        bs_shared_slope,
        bs_env_intercepts
    )
