import os
import warnings
import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression

from qscaled.constants import QSCALED_PATH

np.random.seed(42)


def bootstrap_crossings(df, thresholds, filename: str, use_cached=True):
    bootstrap_cache_file = os.path.join(QSCALED_PATH, 'bootstrap_results', f'{filename}.pkl')

    # Isotonic regression
    iso_reg_results = []

    for _, row in df.iterrows():
        ir = IsotonicRegression(out_of_bounds='clip')
        x = row['training_step']
        y = row['mean_return']
        ir.fit(x, y)
        y_iso = ir.predict(x)
        iso_reg_results.append(y_iso)

    df['return_isotonic'] = iso_reg_results
    crossings_array = []

    for _, row in df.iterrows():
        row_crossings = []
        for threshold in thresholds:
            # Get crossing from isotonic regression
            crossing_idx = np.where(row['return_isotonic'] > threshold)[0]
            row_crossings.append(row['training_step'][crossing_idx[0]] if len(crossing_idx) > 0 else np.nan)

        crossings_array.append(row_crossings)

    df['crossings'] = crossings_array

    if use_cached and os.path.exists(bootstrap_cache_file):
        with open(bootstrap_cache_file, 'rb') as f:
            results = pkl.load(f)
            iso_reg = results['iso_reg']
            iso_reg_stds = results['iso_reg_stds']
            crossings = results['crossings']
            crossings_std = results['crossings_std']
    else:

        def _compute_nanstd(sample_crossings):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always', category=RuntimeWarning)  # Catch all RuntimeWarnings
                crossings_std = np.nanstd(sample_crossings, axis=0)
                for warning in w:
                    if 'Degrees of freedom <= 0 for slice' in str(warning.message):
                        warnings.warn(
                            'It is probable that some environments do not reach every performance threshold '
                            'for every UTD. This can cause the standard deviation to be zero. '
                            'Consider decreasing your thresholds in the config, and call `bootstrap_crossings` '
                            'with `use_cached=False`.',
                            UserWarning,
                        )
                    print(warning.message)

                return crossings_std

        iso_reg = []
        iso_reg_stds = []
        crossings = []
        crossings_std = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            n_bootstrap = 100  # Number of bootstrap samples. 100 seems enough for std to converge
            ir = IsotonicRegression(out_of_bounds='clip')
            x = row['training_step']
            y_iso_samples = []
            sample_crossings = []
            for _ in range(n_bootstrap):
                # Sample with replacement
                sample_indices = np.random.randint(0, row['return'].shape[1], size=row['return'].shape[1])
                y = np.mean(row['return'][:, sample_indices], axis=1)  # Average the bootstrap samples

                # Fit isotonic regression on this bootstrap sample
                ir.fit(x, y)
                y_iso = ir.predict(x)
                y_iso_samples.append(y_iso)

                # For each bootstrap sample, find threshold crossings
                sample_crossing = []
                for threshold in thresholds:
                    crossing_idx = np.where(y_iso > threshold)[0]
                    crossing = row['training_step'][crossing_idx[0]] if len(crossing_idx) > 0 else np.nan
                    sample_crossing.append(crossing)
                sample_crossings.append(sample_crossing)

            # Store mean prediction, crossing statistics, and isotonic std
            iso_reg.append(y_iso_samples)
            crossings.append(np.array(sample_crossings))
            crossings_std.append(_compute_nanstd(sample_crossings))
            iso_reg_stds.append(np.std(y_iso_samples, axis=0))

        # Save results to cache
        results = {
            'iso_reg': iso_reg,
            'iso_reg_stds': iso_reg_stds,
            'crossings': crossings,
            'crossings_std': crossings_std,
        }
        os.makedirs(os.path.dirname(bootstrap_cache_file), exist_ok=True)
        with open(bootstrap_cache_file, 'wb') as f:
            pkl.dump(results, f)

    df['return_isotonic_bootstrap'] = iso_reg
    df['crossings_bootstrap'] = crossings
    df['crossings_std'] = crossings_std
    df['return_isotonic_std'] = iso_reg_stds

    mean_std = np.nanmean(np.array(crossings_std))
    print(f'Average standard deviation across all conditions: {mean_std:.2f}')
    return df


def select_middle_bs_lr(df):
    """
    In some cases, we ran our fit with five (batch size, learning rate) settings:
      * (B*(sigma), eta^*(sigma))
      * (B*(sigma) * 0.7, eta^*(sigma))
      * (B*(sigma) * 1.4, eta^*(sigma))
      * (B*(sigma), eta^*(sigma) * 0.7)
      * (B*(sigma), eta^*(sigma) * 1.4)
    This filters a dataframe to include only (B*(sigma), eta^*(sigma)).
    """
    envs = sorted(df['env_name'].unique())
    utds = sorted(df['utd'].unique())

    filtered_rows = []
    for env in envs:
        env_data = df[df['env_name'] == env]

        for utd in utds:
            utd_data = env_data[env_data['utd'] == utd]
            if len(utd_data) > 0:
                lrs = sorted(utd_data['learning_rate'].unique())
                mid_lr = lrs[len(lrs) // 2]
                batch_sizes = sorted(utd_data['batch_size'].unique())
                mid_bs = batch_sizes[len(batch_sizes) // 2]

                row = utd_data[(utd_data['learning_rate'] == mid_lr) & (utd_data['batch_size'] == mid_bs)]

                if len(row) > 0:
                    filtered_rows.append(row.iloc[0])

    df = pd.DataFrame(filtered_rows)
    return df


def get_envs(df):
    return sorted(df['env_name'].unique().tolist())


def get_utds(df):
    return sorted(df['utd'].unique().tolist())


def get_batch_sizes(df):
    return sorted(df['batch_size'].unique().tolist())


def get_learning_rates(df):
    return sorted(df['learning_rate'].unique().tolist())
