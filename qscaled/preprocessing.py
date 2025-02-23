import os
import numpy as np
import pandas as pd
import pickle as pkl

from tqdm import tqdm
from zipfile import ZipFile
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Tuple

np.random.seed(42)


class ZipLoader:
    """A flexible base class for loading numpy data from zip."""
    
    def __init__(self, max_returns: Dict[str, float], return_key: str):
        """
        max_returns: dict mapping environment names to maximum returns
        return_key: str used for episode returns in Wandb runs
        """
        self.max_returns = max_returns
        self.return_key = return_key
    
    def parse_filename(self, filename: str) -> Tuple:
        """
        Return (env_name, utd, batch_size, learning_rate) corresponding to filename.
        If filename should be skipped, returns None.
        """
        raise NotImplementedError('Implemented in subclasses')
    
    def load(self, zip_path: str, manual_step=None) -> pd.DataFrame:
        """
        Loads data from zip file to a DataFrame.
        If `manual_step is None`, it will load the training steps as is.
        Otherwise, it will 
        """
        records = []
        
        with ZipFile(zip_path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                if filename.endswith('.npy'):
                    if filename.startswith('__MACOSX'):
                        continue
                    result = self.parse_filename(filename)
                    if result is None:
                        continue
                    env_name, utd, batch_size, learning_rate = result
                    with zip_ref.open(filename, 'r') as f:
                        arr = np.load(f)
                    
                    # normalize returns
                    if env_name in self.max_returns:
                        arr *= 1000 / self.max_returns[env_name]
                        
                    record = {
                        'env_name': env_name,
                        'batch_size': batch_size,
                        'utd': utd,
                        'learning_rate': learning_rate,
                    }
                    if manual_step is None:
                        record.update({
                            'training_step': arr[:, 0],
                            'return': arr[:, 1:],
                            'mean_return': np.mean(arr[:, 1:], axis=1),
                            'std_return': np.std(arr[:, 1:], axis=1) / np.sqrt(arr.shape[1] - 1)  # standard error of mean over multiple seeds
                        })
                    else:
                        record.update({
                            'training_step': np.arange(1, arr.shape[0] + 1) * manual_step,
                            'return': arr,
                            'mean_return': np.mean(arr, axis=1),
                            'std_return': np.std(arr, axis=1) / np.sqrt(arr.shape[1])  # standard error of mean over multiple seeds
                        })
                    records.append(record)
                        
        
        df = pd.DataFrame(records)
        
        envs = sorted(df['env_name'].unique().tolist())
        utds = sorted(df['utd'].unique().tolist())
        batch_sizes = sorted(df['batch_size'].unique().tolist())
        learning_rates = sorted(df['learning_rate'].unique().tolist())
        
        return df, (envs, utds, batch_sizes, learning_rates)


class UTDGroupedLoader(ZipLoader):
    """
    If you followed the same workflow involving `save_data` in `create_zip.py`,
    you can use this loader to load the data.
    
    Examples: 
    * gym_sweep/utd_1/Ant-v4/episode.return/bs_128_lr_0.0001.npy
    * dmc_ours/utd_0.5/cartpole-swingup/online_returns/bs_528_lr_0.000902_reset_True.npy
    """
    def __init__(self, max_returns, return_key):
        super().__init__(max_returns, return_key)
        
    def parse_filename(self, filename):
        _, utd_param, env_name, name, params = os.path.splitext(filename)[0].split('/')
        if name != self.return_key: 
            return None
        utd = float(utd_param[len('utd_'):])
        params = params.split('_')  # Example: ['bs', 128, 'lr', 0.0001]
        batch_size = int(params[1])
        learning_rate = float(params[3])
        return env_name, utd, batch_size, learning_rate
    
    
class FullGroupedLoaderUnlabeled(ZipLoader):
    """
    Assumes the data corresponds to returns.
    Example: dmc_baseline/BRO_256_0.5_3e-4/cartpole-swingup.npy
    """
    def __init__(self, max_returns):
        super().__init__(max_returns, return_key=None)
    
    def parse_filename(self, filename):
        _, params, env_name = os.path.splitext(filename)[0].split('/')
        _, batch_size, utd, learning_rate = params.split('_')  # Example: ['BRO', '256', '0.5', '3e-4']
        batch_size = int(batch_size)
        utd = float(utd)
        learning_rate = float(learning_rate)
        return env_name, utd, batch_size, learning_rate


def bootstrap_crossings(df, thresholds, bootstrap_cache_file: str):
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
    
    # Bootstrapping
    if os.path.exists(bootstrap_cache_file):
        with open(bootstrap_cache_file, 'rb') as f:
            results = pkl.load(f)
            iso_reg = results['iso_reg']
            iso_reg_stds = results['iso_reg_stds'] 
            crossings = results['crossings']
            crossings_std = results['crossings_std']
    else:    
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
            
        # Save results to cache
        results = {
            'iso_reg': iso_reg,
            'iso_reg_stds': iso_reg_stds,
            'crossings': crossings, 
            'crossings_std': crossings_std
        }
        os.makedirs(os.path.dirname(bootstrap_cache_file), exist_ok=True)
        with open(bootstrap_cache_file, 'wb') as f:
            pkl.dump(results, f)

    df['return_isotonic_bootstrap'] = iso_reg
    df['crossings_bootstrap'] = crossings
    df['crossings_std'] = crossings_std
    df['return_isotonic_std'] = iso_reg_stds

    mean_std = np.nanmean(np.array(crossings_std))
    print(f"Average standard deviation across all conditions: {mean_std:.2f}")
    
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
                mid_lr = lrs[len(lrs)//2]
                batch_sizes = sorted(utd_data['batch_size'].unique())
                mid_bs = batch_sizes[len(batch_sizes)//2]
                
                row = utd_data[
                    (utd_data['learning_rate'] == mid_lr) & 
                    (utd_data['batch_size'] == mid_bs)
                ]
                
                if len(row) > 0:
                    filtered_rows.append(row.iloc[0])

    df = pd.DataFrame(filtered_rows)
    return df