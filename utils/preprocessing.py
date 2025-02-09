import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from zipfile import ZipFile
from sklearn.isotonic import IsotonicRegression

np.random.seed(42)


def load_zip(zip_path, max_returns, thresholds):
    records = []
    iso_reg_results = []

    with ZipFile(zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            # Sample filename: gym_sweep/utd_1/Ant-v4/episode.return/bs_128_lr_0.0001.npy
            if filename.endswith('.npy'):
                if filename.startswith('__MACOSX'):
                    continue
                _, utd_param, env_name, name, params = os.path.splitext(filename)[0].split('/')
                if name != 'episode.return': 
                    continue
                utd = float(utd_param[len('utd_'):])
                params = params.split('_')
                batch_size = int(params[1])
                learning_rate = float(params[3])
                with zip_ref.open(filename) as f:
                    arr = np.load(f)
                arr *= 1000 / max_returns[env_name]  # Normalize returns
                    
                records.append({ 
                    'env_name': env_name,
                    'batch_size': batch_size,
                    'utd': utd,
                    'learning_rate': learning_rate,
                    'training_step': arr[:, 0],
                    'return': arr[:, 1:],
                    'mean_return': np.mean(arr[:, 1:], axis=1),
                    'std_return': np.std(arr[:, 1:], axis=1) / np.sqrt(arr.shape[1] - 1)  # standard error of mean over multiple seeds
                })

    df = pd.DataFrame(records)
    
    envs = sorted(df['env_name'].unique())
    utds = sorted(df['utd'].unique())
    batch_sizes = sorted(df['batch_size'].unique())
    learning_rates = sorted(df['learning_rate'].unique())
    
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

    return df, (envs, utds, batch_sizes, learning_rates)


def bootstrap(df, thresholds):
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
        crossings_std.append(np.nanstd(sample_crossings, axis=0))
        iso_reg_stds.append(np.std(y_iso_samples, axis=0))

    df['return_isotonic_bootstrap'] = iso_reg
    df['crossings_bootstrap'] = crossings
    df['crossings_std'] = crossings_std
    df['return_isotonic_std'] = iso_reg_stds

    mean_std = np.nanmean(np.array(crossings_std))
    print(f"Average standard deviation across all conditions: {mean_std:.2f}")
    
    return df

