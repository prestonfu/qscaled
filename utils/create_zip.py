import os
import numpy as np
import pandas as pd
from functools import reduce

from utils.wandb_utils import BaseRunCollector


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
        
        if logging_freq:
            merged_df['rounded_step'] = (merged_df['_step'] // logging_freq) * logging_freq  # Resolve non-uniform logging
        agg_dict = {
            '_step': 'first', 
            **{col: 'mean' for col in merged_df.columns if col.startswith('seed')}
        }
        result_df = merged_df.groupby('rounded_step').agg(agg_dict).dropna().reset_index(drop=True)
        
        data_dict[name] = result_df.to_numpy()
    
    return data_dict


def save_data(collector, path, env_name, varname, utd, logging_freq=None):
    dirname = f'cache/data/{path}/utd_{utd}/{env_name}/{varname.replace("/", ".")}'
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