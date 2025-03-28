import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from qscaled.core.preprocessing import get_envs, get_utds


def make_linear_fit_separate_slope(df_best_lr_bs, outputs_dir=None, save_path=None):
    assert (outputs_dir is None) == (save_path is None), 'Both outputs_dir and save_path must be provided or neither'

    def helper(x_col, y_col, label):
        regs = {}
        env_slopes = {}
        env_intercepts = {}
        envs = get_envs(df_best_lr_bs)

        max_env_len = max(len(env) for env in envs)
        max_label_len = len(label)

        # Fit linear regression in log-log space
        for i, env in enumerate(envs):
            env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
            X = np.log10(env_data[x_col].values[:-1]).reshape(-1, 1)
            y = np.log10(env_data[y_col].values[:-1])
            reg = LinearRegression().fit(X, y)
            env_slope = reg.coef_[0]
            env_intercept = reg.intercept_
            regs[env] = reg
            env_slopes[env] = env_slope
            env_intercepts[env] = env_intercept

            formatted_env = f'{env}:'.ljust(max_env_len + 1)
            formatted_label = label.ljust(max_label_len)
            print(f'  {formatted_env}  {formatted_label} ~ {10**env_intercept:.6f} * UTD^{env_slope:.6f}')

        print()
        return regs, env_slopes, env_intercepts

    print('Separate slope fits:')
    lr_regs, lr_env_slopes, lr_env_intercepts = helper('utd', 'best_lr_bootstrap_bsmean', label='learning rate')
    bs_regs, bs_env_slopes, bs_env_intercepts = helper('utd', 'best_bs_bootstrap_lrmean', label='batch size')

    lr_export = [lr_env_slopes, lr_env_intercepts]
    bs_export = [bs_env_slopes, bs_env_intercepts]

    if outputs_dir is not None:
        save_dir = os.path.join(outputs_dir, 'grid_proposed_fits', save_path)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'separate_lr_fit.npy'), lr_export)
        np.save(os.path.join(save_dir, 'separate_bs_fit.npy'), bs_export)

    print('Saved separate slope fits to:', save_dir, '\n')
    return lr_regs, bs_regs


def get_separate_slope_prediction(regs, utds_to_predict, envs):
    predictions_per_env = {}
    for env in envs:
        reg = regs[env]
        prediction = 10 ** (reg.predict(np.log10(utds_to_predict).reshape(-1, 1)))
        predictions_per_env[env] = prediction
    return predictions_per_env


def get_shared_slope_predictions(reg_shared, utds_to_predict, envs):
    n_envs = len(envs)
    predictions_per_env = {}
    for i, env in enumerate(envs):
        X = np.log10(utds_to_predict).reshape(-1, 1)
        env_dummies_plot = np.zeros((len(utds_to_predict), n_envs - 1))
        if i > 0:
            env_dummies_plot[:, i - 1] = 1
        X_combined = np.hstack([X, env_dummies_plot])
        predictions = reg_shared.predict(X_combined)
        predictions_per_env[env] = 10**predictions
    return predictions_per_env


def make_linear_fit_shared_slope(df_best_lr_bs, outputs_dir=None, save_path=None):
    envs = get_envs(df_best_lr_bs)

    def helper(df_best_lr_bs, x_col, y_col, label):
        """
        Plot linear fit for bootstrap averaged learning rate and batch size
        with a shared slope per environment.
        """
        n_envs = len(envs)
        X_all = []
        y_all = []
        env_indices = []

        for i, env in enumerate(envs):
            env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
            X_all.append(np.log10(env_data[x_col].values[:]))
            y_all.append(np.log10(env_data[y_col].values[:]))
            env_indices.extend([i] * len(env_data))

        X_all = np.concatenate(X_all).reshape(-1, 1)
        y_all = np.concatenate(y_all)
        env_indices = np.array(env_indices)

        # Create dummy variables for environment intercepts
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
        env_intercepts = {env: intercept for env, intercept in zip(envs, env_intercepts)}

        max_env_len = max(len(env) for env in envs)
        max_label_len = len(label)

        for i, env in enumerate(envs):
            formatted_env = f'{env}:'.ljust(max_env_len + 1)
            formatted_label = label.ljust(max_label_len)
            print(f'  {formatted_env}  {formatted_label} ~ {10 ** env_intercepts[env]:.6f} * UTD^{shared_slope:.6f}')

        print()
        return reg_shared, shared_slope, env_intercepts

    assert (outputs_dir is None) == (save_path is None), 'Both outputs_dir and save_path must be provided or neither'

    print('Shared slope fits:')
    lr_reg_shared, lr_shared_slope, lr_env_intercepts = helper(
        df_best_lr_bs, 'utd', 'best_lr_bootstrap_bsmean', label='learning rate'
    )
    bs_reg_shared, bs_shared_slope, bs_env_intercepts = helper(
        df_best_lr_bs, 'utd', 'best_bs_bootstrap_lrmean', label='batch size'
    )

    lr_shared_slope_dict = {env: lr_shared_slope for env in envs}
    bs_shared_slope_dict = {env: bs_shared_slope for env in envs}

    lr_export = [lr_shared_slope_dict, lr_env_intercepts]
    bs_export = [bs_shared_slope_dict, bs_env_intercepts]

    if outputs_dir is not None:
        save_dir = os.path.join(outputs_dir, 'grid_proposed_fits', save_path)
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'shared_lr_fit.npy'), lr_export)
        np.save(os.path.join(save_dir, 'shared_bs_fit.npy'), bs_export)

    print('Saved shared slope fits to:', save_dir, '\n')
    return lr_reg_shared, bs_reg_shared


def load_fits(directory, name, fit_type):
    assert fit_type in ['separate', 'shared']
    lr_path = os.path.join(directory, 'grid_proposed_fits', name, f'{fit_type}_lr_fit.npy')
    bs_path = os.path.join(directory, 'grid_proposed_fits', name, f'{fit_type}_bs_fit.npy')
    lr_env_slopes, lr_env_intercepts = np.load(lr_path, allow_pickle=True)
    bs_env_slopes, bs_env_intercepts = np.load(bs_path, allow_pickle=True)
    return lr_env_slopes, lr_env_intercepts, bs_env_slopes, bs_env_intercepts


def get_fit_mean_batch_size(envs, directory, name, fit_type):
    _, _, bs_env_slopes, bs_env_intercepts = load_fits(directory, name, fit_type)

    def helper(utd):
        batch_sizes = []
        for env in envs:
            batch_sizes.append(10 ** (bs_env_slopes[env] * np.log10(utd) + bs_env_intercepts[env]))
        batch_sizes = np.stack(batch_sizes)
        return np.mean(batch_sizes, axis=0)

    return helper


def _plot_fits_against_grid(
    df_grid, df_best_lr_bs, utds_to_predict, og_predictions, desired_predictions, grid_y_col, bootstrap_y_col, title
):
    """
    og_predictions: Predictions per env for utds in grid search
    desired_predictions: Predictions per env for utds to predict

    Plot includes:
    * Bootstrap estimates
    * Linear fit
    * Proposed values
    * Grid search values
    """
    envs = get_envs(df_grid)
    n_envs = len(envs)
    n_cols = 4
    n_rows = (n_envs + n_cols - 1) // n_cols
    r2_str = r'$R^2$'
    utds_to_predict = np.array(utds_to_predict)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharey=True)
    axes = axes.flatten()

    def r2_score(y, yhat):
        rss = np.sum((y - yhat) ** 2)
        tss = np.sum((y - np.mean(y)) ** 2)
        return 1 - (rss / tss)

    for i, (env, ax) in enumerate(zip(envs, axes)):
        env_data = df_best_lr_bs[df_best_lr_bs['env_name'] == env]
        utd_values = env_data['utd']
        bootstrap_estimates = env_data[bootstrap_y_col]
        grid_y_values = df_grid[df_grid['env_name'] == env][grid_y_col].unique()

        # Plot bootstrap estimates
        ax.scatter(utd_values, bootstrap_estimates, label='Bootstrap estimates', color='black')

        # Plot linear fit for UTDs within support
        ax.plot(utd_values, og_predictions[env], label='Fit', color='black')
        r2 = r2_score(np.log10(bootstrap_estimates), np.log10(og_predictions[env]))

        # Plot proposed values
        ax.scatter(utds_to_predict, desired_predictions[env], marker='o', color='lightblue', label='Proposed values')

        # Plot grid search values
        grid_kw = dict(color='black', alpha=0.2, linestyle='--')
        for utd in utd_values:
            ax.plot(
                [utd, utd],
                [min(grid_y_values), max(grid_y_values)],
                **grid_kw,
                label='Grid search values' if utd == utd_values.iloc[0] else None,
            )
        for grid_y in grid_y_values:
            ax.plot([min(utd_values), max(utd_values)], [grid_y, grid_y], **grid_kw)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('UTD')
        ax.set_title(f'{env} ({r2_str}={r2:.2f})')
        ax.grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc='center left', borderaxespad=0, frameon=False)
    plt.suptitle(title, size=14)
    plt.tight_layout()
    plt.show()


def plot_fits_against_grid_separate_slope(df_grid, df_best_lr_bs, utds_to_predict, lr_regs, bs_regs):
    envs = get_envs(df_grid)
    utds = get_utds(df_grid)
    lr_predictions_og_utds = get_separate_slope_prediction(lr_regs, utds, envs)
    lr_predictions_desired_utds = get_separate_slope_prediction(lr_regs, utds_to_predict, envs)
    _plot_fits_against_grid(
        df_grid,
        df_best_lr_bs,
        utds_to_predict,
        lr_predictions_og_utds,
        lr_predictions_desired_utds,
        'learning_rate',
        'best_lr_bootstrap_bsmean',
        r'$\eta^*$: Best learning rate',
    )

    bs_predictions_og_utds = get_separate_slope_prediction(bs_regs, utds, envs)
    bs_predictions_desired_utds = get_separate_slope_prediction(bs_regs, utds_to_predict, envs)
    _plot_fits_against_grid(
        df_grid,
        df_best_lr_bs,
        utds_to_predict,
        bs_predictions_og_utds,
        bs_predictions_desired_utds,
        'batch_size',
        'best_bs_bootstrap_lrmean',
        r'$B^*$: Best batch size',
    )


def plot_fits_against_grid_shared_slope(df_grid, df_best_lr_bs, utds_to_predict, lr_reg_shared, bs_reg_shared):
    envs = get_envs(df_grid)
    utds = get_utds(df_grid)

    lr_predictions_og_utds = get_shared_slope_predictions(lr_reg_shared, utds, envs)
    lr_predictions_desired_utds = get_shared_slope_predictions(lr_reg_shared, utds_to_predict, envs)
    _plot_fits_against_grid(
        df_grid,
        df_best_lr_bs,
        utds_to_predict,
        lr_predictions_og_utds,
        lr_predictions_desired_utds,
        'learning_rate',
        'best_lr_bootstrap_bsmean',
        r'$\eta^*$: Best learning rate',
    )

    bs_predictions_og_utds = get_shared_slope_predictions(bs_reg_shared, utds, envs)
    bs_predictions_desired_utds = get_shared_slope_predictions(bs_reg_shared, utds_to_predict, envs)
    _plot_fits_against_grid(
        df_grid,
        df_best_lr_bs,
        utds_to_predict,
        bs_predictions_og_utds,
        bs_predictions_desired_utds,
        'batch_size',
        'best_bs_bootstrap_lrmean',
        r'$B^*$: Best batch size',
    )


def _tabulate_proposed_hparams(
    utds_to_predict, lr_predictions, bs_predictions, envs, outputs_dir=None, save_path=None, verbose=False
):
    assert (outputs_dir is None) == (save_path is None), 'Both outputs_dir and save_path must be provided or neither'

    proposed_hparams = {'Environment': [], 'UTD': [], 'Learning Rate': [], 'Batch Size': []}

    for env in envs:
        for utd, lr_pred, bs_pred in zip(utds_to_predict, lr_predictions[env], bs_predictions[env]):
            proposed_hparams['Environment'].append(env)
            proposed_hparams['UTD'].append(utd)
            proposed_hparams['Learning Rate'].append(lr_pred)
            proposed_hparams['Batch Size'].append(bs_pred)

    proposed_hparams_df = pd.DataFrame(proposed_hparams)

    proposed_hparams_df['Learning Rate x√2'] = proposed_hparams_df['Learning Rate'] * np.sqrt(2)
    proposed_hparams_df['Learning Rate x√0.5'] = proposed_hparams_df['Learning Rate'] * np.sqrt(0.5)
    proposed_hparams_df['Batch Size x√2'] = proposed_hparams_df['Batch Size'] * np.sqrt(2)
    proposed_hparams_df['Batch Size x√0.5'] = proposed_hparams_df['Batch Size'] * np.sqrt(0.5)

    for col in proposed_hparams_df.columns:
        if 'Learning Rate' in col:
            proposed_hparams_df[col] = proposed_hparams_df[col].apply(lambda x: f'{x:.2e}').astype(float)
        if 'Batch Size' in col:
            proposed_hparams_df[col] = proposed_hparams_df[col].astype(int)
            proposed_hparams_df[f'{col}(rounded)'] = (np.round(proposed_hparams_df[col] / 16) * 16).astype(int)

    if verbose:
        pd.options.display.float_format = '{:.2e}'.format
        print(proposed_hparams_df, '\n')

    if outputs_dir is not None:
        full_path = os.path.join(outputs_dir, 'grid_proposed_hparams', f'{save_path}_fitted.csv')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        proposed_hparams_df.to_csv(full_path, index=False)

    print('Saved proposed hyperparameters to:', full_path, '\n')
    return proposed_hparams_df


def tabulate_proposed_hparams_separate_slope(
    df_grid, utds_to_predict, lr_regs, bs_regs, outputs_dir=None, save_path=None, verbose=False
):
    envs = get_envs(df_grid)
    lr_predictions = get_separate_slope_prediction(lr_regs, utds_to_predict, envs)
    bs_predictions = get_separate_slope_prediction(bs_regs, utds_to_predict, envs)
    return _tabulate_proposed_hparams(
        utds_to_predict, lr_predictions, bs_predictions, envs, outputs_dir, f'{save_path}/separate', verbose
    )


def tabulate_proposed_hparams_shared_slope(
    df_grid, utds_to_predict, lr_reg_shared, bs_reg_shared, outputs_dir=None, save_path=None, verbose=False
):
    envs = get_envs(df_grid)
    lr_predictions = get_shared_slope_predictions(lr_reg_shared, utds_to_predict, envs)
    bs_predictions = get_shared_slope_predictions(bs_reg_shared, utds_to_predict, envs)
    return _tabulate_proposed_hparams(
        utds_to_predict, lr_predictions, bs_predictions, envs, outputs_dir, f'{save_path}/shared', verbose
    )


def tabulate_baseline_hparams(
    df_grid, utds_to_predict, utd_at='middle', outputs_dir=None, save_path=None, verbose=False
):
    """
    For a fixed `utd_at`, find the optimal (batch size, learning rate) pair,
    and run the same configuration across multiple UTDs.

    If `utd_at == 'middle'`, the utd_at is set to the geometric mean of the grid search.
    """
    assert (outputs_dir is None) == (save_path is None), 'Both outputs_dir and save_path must be provided or neither'

    utds = get_utds(df_grid)
    envs = get_envs(df_grid)
    n_envs = len(envs)
    if utd_at == 'middle':
        middle_utd = np.prod(utds_to_predict) ** (1 / len(utds_to_predict))  # Geometric mean of predicted UTDs
        utd_at = min(utds, key=lambda x: abs(np.log(x) - np.log(middle_utd)))  # Snap to nearest utd_at in grid search
    utd_at = float(utd_at)
    assert utd_at in utds, f'utd={utd_at} not found in grid search'
    print('Baseline based optimal hyperparamers at UTD', utd_at, '\n')

    utd_data = df_grid[df_grid['utd'] == utd_at]
    utd_data['last_crossing'] = utd_data['crossings'].apply(lambda x: x[-1])
    idx = utd_data.groupby(['env_name'])['last_crossing'].idxmin()
    baseline_values_df = utd_data.loc[idx]

    baseline_values_df['utd'] = [utds_to_predict] * n_envs
    baseline_values_df = baseline_values_df.explode('utd').reset_index(drop=True)
    baseline_values_df = baseline_values_df[['env_name', 'utd', 'learning_rate', 'batch_size']].rename(
        columns={
            'env_name': 'Environment',
            'utd': 'UTD',
            'learning_rate': 'Learning Rate',
            'batch_size': 'Batch Size',
        }
    )
    baseline_values_df['is_new_experiment'] = ~baseline_values_df['UTD'].isin(utds)

    if verbose:
        pd.options.display.float_format = '{:.2e}'.format
        print(baseline_values_df, '\n')

    if outputs_dir is not None:
        full_path = os.path.join(outputs_dir, 'grid_proposed_hparams', f'{save_path}/baseline_utd{utd_at}.csv')
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        baseline_values_df.to_csv(full_path, index=False)

    print('Saved baseline hyperparameters to:', full_path, '\n')
    return baseline_values_df
