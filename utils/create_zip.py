import os
import numpy as np
import pandas as pd
import subprocess
from functools import reduce

from utils.configs import Config
from utils.wandb_utils import BaseRunCollector


script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, '../cache') 

def replace_slashes(s: str):
    return s.replace("/", ".")

def save_and_load(config: Config):
    assert config.wandb_collector is not None or os.path.exists(f'{cache_dir}/zip/{config.name}.zip'), \
        "Either wandb_collector must be provided or zip file must exist."
    if config.wandb_collector is not None:
        os.makedirs(f"{cache_dir}/zip", exist_ok=True)
        save_loop(config.wandb_collector, config.return_key, config.name, config.logging_freq)
        subprocess.run(
            f'zip -r {config.name}.zip {config.name} && mv {config.name}.zip ../zip',
            shell=True,
            check=True,
            cwd=f'{cache_dir}/prezip',
        )
    zip_loader = config.zip_load_cls(config.max_returns, replace_slashes(config.return_key))
    return zip_loader.load(f"{cache_dir}/zip/{config.name}.zip")

def get_data(collector: BaseRunCollector, env_name, varname, utd, logging_freq=None):
    """
    Merges data from multiple seeds into pd.DataFrame.
    If `logging_freq` is not None, it will round the step to the nearest multiple of `logging_freq`.
    """
    all_data = collector.filter(env=env_name, utd=utd)
    data_dict = {}
    for key, summaries in all_data.items():
        bs = key[collector.category_index['batch_size']]
        lr = key[collector.category_index['learning_rate']]
        name = (env_name, utd, bs, lr)
        short_summaries = []
        for i, df in enumerate(summaries):
            short_summaries.append(df[['_step', varname]].rename(columns={varname: f'seed{i}/{varname}'}))
        merged_df = reduce(
            lambda l, r: pd.merge(l, r, on='_step', how='outer'),
            short_summaries
        )
        
        if logging_freq:  # Resolve non-uniform logging
            merged_df['rounded_step'] = (merged_df['_step'] // logging_freq) * logging_freq
        agg_dict = {
            '_step': 'first', 
            **{col: 'mean' for col in merged_df.columns if col.startswith('seed')}
        }
        result_df = merged_df.groupby('rounded_step').agg(agg_dict).dropna().reset_index(drop=True)
        
        data_dict[name] = result_df.to_numpy()
    
    return data_dict


def save_data(collector, path, env_name, varname, utd, logging_freq=None):
    """Creates a directory and saves aggregate data for zipping."""
    dirname = f'{cache_dir}/prezip/{path}/utd_{utd}/{env_name}/{replace_slashes(varname)}'
    os.makedirs(dirname, exist_ok=True)
    data_dict = get_data(collector, env_name, varname, utd, logging_freq)
    for key, value in data_dict.items():
        env_name, utd, bs, lr = key
        name = f'bs_{bs}_lr_{lr}'
        np.save(f'{dirname}/{name}', value)


def save_loop(collector, return_key, path, logging_freq=None):
    for env in collector.get_unique('env'):
        for utd in collector.get_unique('utd', env=env):
            for varname in [return_key]:
                save_data(collector, path, env, varname, utd, logging_freq)
