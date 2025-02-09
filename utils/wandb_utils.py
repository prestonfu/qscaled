import os
import abc
import numpy as np
import wandb
import pandas as pd
import time

from functools import reduce
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

np.random.seed(42)

api = wandb.Api(timeout=120)
DUMMY_VALUE = 'None'


class BaseRunCollector(abc.ABC):
    def __init__(self, project: str):
        self.project = project
        self.data = defaultdict(list)
        self._set_category_index()
        self._set_wandb_keys()
        
    @abc.abstractmethod
    def _set_category_index(self):
        """Specifies run categories and their order."""
        self.categories = None
        self.num_categories = None
        self.category_index = None
        raise NotImplementedError
        
    @abc.abstractmethod
    def _set_wandb_keys(self):
        """Specifies keys to fetch from Wandb."""
        self.wandb_keys = None
        raise NotImplementedError
    
    @abc.abstractmethod
    def generate_key(self, run):
        raise NotImplementedError
    
    @abc.abstractmethod
    def insert(self, run):
        raise NotImplementedError
    
    @abc.abstractmethod
    def load_state(self, path):
        raise NotImplementedError

    @abc.abstractmethod
    def save_state(self, path):
        raise NotImplementedError

    def sample(self):
        """Returns a sample key and value"""
        keys = list(self.data.keys())
        sampled_key = keys[np.random.choice(len(keys))]
        return sampled_key, self.data[sampled_key]

    def get_unique(self, category, *args, **kw):
        """Find unique keys in category subject to constraints (default none)"""
        if category not in self.category_index:
            raise ValueError(
                f"Invalid category: {category}. Must be one of {list(self.category_index.keys())}."
            )
        idx = self.category_index[category]
        unique = set()
        for key in self.filter(*args, **kw):
            unique.add(key[idx])
        return sorted(list(unique))
        
    def _dummy_key_factory(self, *args, **kw):
        """Create a dummy key with DUMMY_VALUE for unspecified categories."""
        params = [DUMMY_VALUE] * self.num_categories
        for i, value in enumerate(args):
            params[i] = value
        for cat, value in kw.items():
            assert self.category_index[cat] >= len(args)
            params[self.category_index[cat]] = value
        return tuple(params)
        
    def _check_key(self, check_key: tuple, *args, **kw):
        """Check whether check_key satisfies the constraints specified by env, args, kw."""
        template_key = self._dummy_key_factory(*args, **kw)
        for check, template in zip(check_key, template_key):
            if template not in (DUMMY_VALUE, check):
                return False
        return True

    def filter(self, *args, **kw) -> Dict[Tuple[Any], List[Dict[str, Any]]]:
        return {key: summaries for key, summaries in self.data.items() if self._check_key(key, *args, **kw)}

    def trim(self, num_seeds, verbose=False):
        """Trims the collector to only the best `num_seeds` runs for each key."""
        for key, summaries in self.data.items():
            # summaries = [summary for summary in summaries if not summary.empty and 'episode/return' in summary.columns] # TODO: remove this line
            if len(summaries) > num_seeds:
                final_train_returns = [(i, summary['episode/return'].mean()) for i, summary in enumerate(summaries)]
                final_train_returns = sorted(final_train_returns, key=lambda x: -x[1])
                idx = [x[0] for x in final_train_returns]
                self.data[key] = [summaries[idx[j]] for j in range(num_seeds)]
            elif len(summaries) < num_seeds and verbose:
                print(f'Warning: key {key} contains {len(summaries)} < {num_seeds} runs')

    def merge(self, other):
        """Merge with another RunCollector object"""
        for k, v in other.data.items():
            self.data[k].extend(v)
        return self
    
    def wandb_fetch(self, run, num_tries=5):
        """Returns full run history"""
        df = None
        for _ in range(num_tries):
            try:
                df = run.history(samples=10000)
                df = df[['_step', *self.wandb_keys]]
                return None if len(df) < 10 else df
            except:
                time.sleep(1)
                pass
        return df
    
    def create(self, load: bool, tags: List[str] = [], path: str = '', parallel=True):
        """
        Creates a new RunCollector object.
        
        If `load == True` and `path` exists, load the state from `path`.
        Otherwise, fetch all runs with tag in `tags` and insert them into the collector.
        """
        collector = self.__class__(self.project)
        if load and os.path.exists(path):
            collector.load_state(path)
        else:
            runs = api.runs(collector.project, {"tags": {"$in": tags}})
            insert = lambda run: collector.insert(run)
            if parallel:
                with ThreadPool() as pool:
                    list(tqdm(pool.imap(insert, runs), total=len(runs)))
            else:
                for run in tqdm(runs, total=len(runs)):
                    insert(run)
            collector.save_state(path)
        return collector
    
    def add_tag(self, tags: List[str], new_tag: str, *args, parallel=True, **kw):
        """
        Updates tags of all runs with tags in `tags` that pass the (args, kw) filter
        by adding `new_tag`.
        """
        runs = api.runs(self.project, {"tags": {"$in": tags}})
        def update(run):
            if self._check_key(self.generate_key(run), *args, **kw):
                if new_tag not in run.tags:
                    run.tags.append(new_tag)
                    run.update()
        if parallel:
            with ThreadPool() as pool:
                list(tqdm(pool.imap(update, runs), total=len(runs)))
        else:
            for run in tqdm(runs, total=len(runs)):
                update(run)


class CRLRunCollector(BaseRunCollector):
    """An example RunCollector implementation."""
    
    def __init__(self, project: str):
        super().__init__(project)
        
    def _set_category_index(self):
        """Specifies categories and their order."""
        self.categories = ['env', 'utd', 'batch_size', 'learning_rate']
        self.num_categories = len(self.categories)
        self.category_index = {cat: i for i, cat in enumerate(self.categories)}
        
    def _set_wandb_keys(self):
        self.wandb_keys = [
            'episode/return', 
            'evaluation/return',
            'training/critic_loss',
            'training/new_data_critic_loss',
            'val/critic_loss',
            'training/policy_data_ce',
            'training/critic_pnorm_l2',
            'training/critic_gnorm_l2',
        ]
        
    def generate_key(self, run):
        env = run.config["env_name"]
        utd = run.config["utd_ratio"]
        batch_size = run.config["batch_size"]
        learning_rate = run.config["agent.critic_lr"]
        key = (env, utd, batch_size, learning_rate)  # key is given in same order as `self.categories`
        return key
        
    def insert(self, run, verbose=False):
        if run.state != 'finished' and verbose:
            print(f'{run.name} skipped with status {run.state}')
            return       

        df = self.wandb_fetch(run)
        if df is None and verbose:
            print(f'Failed to fetch {run.name} from Wandb')
        else:
            self.data[self.generate_key(run)].append(df)
    
    def load_state(self, path):
        self.data.update(np.load(path, allow_pickle=True).item())

    def save_state(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.data)  # pickle/gzip on the dictionary is very slow

    def get_all(self, metric, *args, **kw):
        res_dict = defaultdict(list)
        for key, summaries in self.filter(*args, **kw).items():
            for summary in summaries:
                res_dict[key].append(summary[['_step', metric]])
        return res_dict
    
    def get_agg(self, metric, *args, **kw):
        """Make a single Dataframe for each key with the mean and standard error of the metric across seeds."""
        all_data = self.get_all(metric, *args, **kw)
        res_dict = {}
        for key, summaries in all_data.items():
            merged_df = reduce(
                lambda l, r: pd.merge(l, r, on='_step', how='outer', suffixes=('', '**')),
                summaries
            )
            merged_df = merged_df.sort_values('_step').dropna().reset_index()
            metric_cols = [c for c in merged_df.columns if metric in c]
            merged_df[f'{metric}_mean'] = merged_df[metric_cols].mean(axis=1, skipna=True)
            merged_df[f'{metric}_std'] = merged_df[metric_cols].std(axis=1, skipna=True) / np.sqrt(len(metric_cols))
            res_dict[key] = merged_df[['_step', f'{metric}_mean', f'{metric}_std']]
        return res_dict
        
    def remove_short(self, thresh=0.95):
        """Removes runs that are run for less than `thresh` times the length of the longest run"""
        for key, summaries in self.data.items():
            # summaries = [summary for summary in summaries if not summary.empty] # TODO: remove this line
            step_counts = [summary['_step'].iloc[-1] for summary in summaries]
            max_step_count = max(step_counts)
            for i, step_count in reversed(list(enumerate(step_counts))):
                if step_count < thresh * max_step_count:
                    summaries.pop(i)
