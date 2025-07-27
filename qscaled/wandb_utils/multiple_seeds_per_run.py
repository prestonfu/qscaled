import numpy as np
import pandas as pd
import re
from functools import reduce
from typing import Dict, Tuple, List, Union, Any
from collections import defaultdict
from copy import deepcopy

from qscaled.wandb_utils.base_collector import BaseCollector
from qscaled.wandb_utils import flatten_dict, get_wandb_run_history, get_dict_value


class MultipleSeedsPerRunCollector(BaseCollector):
    """
    An example implementation supporting multiple seeds per run.
    These metrics have keys like `seed0/return`, `seed1/return`, etc.
    """

    def __init__(
        self,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: Union[List[str], str] = [],
        use_cached: bool = True,
        parallel: bool = True,
    ):
        super().__init__(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags,
            use_cached=use_cached,
            parallel=parallel,
        )

    def _set_hparams(self):
        self._hparams = ['env', 'utd', 'batch_size', 'learning_rate']
        
    def _combine_metadatas(self, metadatas):
        """
        Combine metadata from multiple runs. This function is called when
        flattening the data.
        """
        combined_metadata = defaultdict(list)
        for metadata in metadatas:
            for key, value in metadata.items():
                combined_metadata[key].append(value)

        combined_metadata['num_seeds'] = sum(metadata['num_seeds'] for metadata in metadatas)
        combined_metadata['runtime_mins'] = (
            sum(metadata['runtime_mins'] * metadata['num_seeds'] for metadata in metadatas)
            / combined_metadata['num_seeds']
        )
        combined_metadata['last_step'] = (
            sum(metadata['last_step'] * metadata['num_seeds'] for metadata in metadatas)
            / combined_metadata['num_seeds']
        )

        return combined_metadata

    def flatten(self, forward_fill_metrics=None, subset_logging_freq=None, round_logging_freq=None):
        """
        Concatenates the dataframes corresponding to multiple runs for each key, 
        so that each metadatas and rundatas has length 1.

        For example, one run may have 5 seeds (seed0/ .. seed4/), and another may
        have 10 seeds (seed0/ .. seed9/). The resulting dataframe will have 15
        seeds (seed0/ .. seed14/).
        
        Applies forward fill to metrics listed in `forward_fill_metrics` after 
        the last valid index.
        
        `subset_logging_freq` only includes rows where the environment step is a multiple of 
        this value.
        
        `round_logging_freq` rounds the environment step to the nearest multiple of this value.

        Returns a new instance of the class with the flattened data.
        """
        new_collector = self.__class__(
            wandb_entity=self._wandb_entity,
            wandb_project=self._wandb_project,
            wandb_tags=None,
        )
        new_collector.copy_state(self)

        merge_fn = lambda l, r: pd.merge(l, r, on=self._env_step_key, how='outer')
        
        def ffill_after_last_valid(s):
            last_valid = s.last_valid_index()
            if last_valid is None:
                return s
            before = s.loc[:last_valid]
            after = s.loc[last_valid:].ffill()
            return pd.concat([before, after[1:]])

        for key in self.keys():
            metadatas = self._metadatas[key]
            rundatas = self._rundatas[key]
            j = 0
            combined_metadata = []
            relabeled_rundatas = []

            for metadata, rundata in zip(metadatas, rundatas):
                combined_metadata.append(metadata)
                num_seeds = metadata['num_seeds']
                renamer = {}
                allowed_cols = [self._env_step_key]

                for col in rundata.columns:
                    if col.startswith('seed'):
                        seed_part, metric_name = col.split('/')
                        seed_num = int(seed_part[4:])
                        renamer[col] = f'seed{j + seed_num}/{metric_name}'
                        allowed_cols.append(col)

                relabeled_rundata = rundata[allowed_cols].rename(columns=renamer)
                if subset_logging_freq is not None:
                    relabeled_rundata = relabeled_rundata[
                        relabeled_rundata[self._env_step_key] % subset_logging_freq == 0
                    ]
                if round_logging_freq is not None:
                    relabeled_rundata = self._resolve_logging_freq(relabeled_rundata, round_logging_freq)
                relabeled_rundatas.append(relabeled_rundata)
                j += num_seeds

            merged_rundatas = reduce(merge_fn, relabeled_rundatas)
            if forward_fill_metrics is not None:
                for metric in forward_fill_metrics:
                    metric_cols = [col for col in merged_rundatas.columns if re.search(rf'^seed\d+/{metric}$', col)]
                    assert len(metric_cols) > 0, f'Metric {metric} not found in columns: {merged_rundatas.columns}.'
                    merged_rundatas[metric_cols] = merged_rundatas[metric_cols].apply(ffill_after_last_valid)
            
            merged_rundatas = merged_rundatas.dropna(
                how='all', subset=merged_rundatas.columns.difference([self._env_step_key])
            )
            new_collector._metadatas[key] = combined_metadata
            new_collector._rundatas[key] = [merged_rundatas]

        return new_collector
    
    def sample_seeds(self, p):
        """Sample fraction p of seeds from each run."""
        for key in self.keys():
            metadatas, rundatas = self._metadatas[key], self._rundatas[key]
            for i, (metadata, rundata) in enumerate(zip(metadatas, rundatas)):
                num_seeds = metadata['num_seeds']
                if num_seeds <= 1:
                    continue
                num_samples = max(1, int(num_seeds * p))
                sampled_seeds = np.random.choice(range(num_seeds), num_samples, replace=False)
                sampled_seeds = {seed_num: i for i, seed_num in enumerate(sorted(sampled_seeds))}
                sampled_cols = []
                sampled_cols_renamed = []
                for col in rundata.columns:
                    if col.startswith('seed'):
                        seed_num = int(col.split('/')[0][4:])
                        if seed_num in sampled_seeds:
                            sampled_cols.append(col)
                            sampled_cols_renamed.append(
                                f'seed{sampled_seeds[seed_num]}/{col.split("/")[1]}'
                            )
                    else:
                        sampled_cols.append(col)
                        sampled_cols_renamed.append(col)
                new_rundata = rundata[sampled_cols]
                new_rundata.columns = sampled_cols_renamed
                metadatas[i]['num_seeds'] = num_samples
                rundatas[i] = new_rundata

    def prepare_zip_export_data(self, metric, logging_freq=None) -> Dict[Any, np.ndarray]:
        data_dict = {}
        flattened_collector = self.flatten(logging_freq=logging_freq)

        for env in self.get_unique('env'):
            for utd in self.get_unique('utd', filter_str=f'env=="{env}"'):
                filtered_rundatas = flattened_collector.get_filtered_rundatas(
                    f'env=="{env}" and utd=={utd}'
                )
                for key, rundatas in filtered_rundatas.items():
                    rundata = rundatas[0]
                    bs = key[self.hparam_index['batch_size']]
                    lr = key[self.hparam_index['learning_rate']]
                    save_key = (env, utd, bs, lr)
                    subset = [self._env_step_key] + [
                        col for col in rundata.columns if metric in col
                    ]
                    data_dict[save_key] = rundata[subset].to_numpy()

        return data_dict

    def drop_bad_seeds(self, thresh=0.9, q=0.75):
        """
        Drop everything with final performance worse than `thresh` times the run
        with quantile `q`.
        """
        for key in self.keys():
            metadatas, rundatas = self._metadatas[key], self._rundatas[key]
            empty = True
            for i, (metadata, rundata) in enumerate(zip(metadatas, rundatas)):
                if len(rundata) == 0:
                    continue
                empty = False
                return_cols = [c for c in rundata if re.match(r'^seed\d+/return$', c)]
                returns = np.nanmax(rundata[return_cols].values, axis=0)
                return_thresh = np.quantile(returns, q) * thresh
                good_seeds = {x: i for i, x in enumerate(sorted(np.where(returns >= return_thresh)[0]))}
                num_good_seeds = len(good_seeds)
                good_cols, good_cols_renamed = [], []
                for c in rundata.columns:
                    if c.startswith('seed'):
                        seed_num = int(c.split('/')[0][4:])
                        if seed_num in good_seeds:
                            good_cols.append(c)
                            good_cols_renamed.append(f'seed{good_seeds[seed_num]}/{c.split("/")[1]}')
                    else:
                        good_cols.append(c)
                        good_cols_renamed.append(c)
                metadata['num_seeds'] = num_good_seeds
                rundatas[i] = rundata[good_cols].rename(columns=dict(zip(good_cols, good_cols_renamed)))
            if empty:
                self._metadatas.pop(key)
                self._rundatas.pop(key)


class ExampleMultipleSeedsPerRunCollector(MultipleSeedsPerRunCollector):
    def __init__(
        self,
        wandb_entity: str,
        wandb_project: str,
        wandb_tags: Union[List[str], str] = [],
        use_cached: bool = True,
        parallel: bool = True,
    ):
        super().__init__(
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_tags=wandb_tags,
            use_cached=use_cached,
            parallel=parallel,
        )

    def _set_wandb_metrics(self):
        """seed{i}/{metric_name}"""
        self._wandb_metrics = [
            'return',
            'critic_loss',
            'rolling_new_data_critic_loss',
            'rolling_validation_data_critic_loss',
            'critic_pnorm',
            'critic_gnorm',
            'critic_agnorm',
        ]

    def _generate_key(self, run):
        config = flatten_dict(run.config)
        env = config['env_name']
        utd = config['utd_ratio']
        batch_size = config['batch_size']
        learning_rate = config['agent.critic_lr']
        key = (env, utd, batch_size, learning_rate)  # key is given in same order as `self._hparams`
        return key

    def wandb_fetch(self, run) -> Union[Tuple[Dict[str, Any], pd.DataFrame], None]:
        """Returns run metadata and history. If fails, returns None."""
        last_step = run.summary[self._env_step_key]
        if last_step < 50e3:
            return None
        config = flatten_dict(run.config)
        result = get_wandb_run_history(run)
        if result is None:
            return None
        num_seeds = config['num_seeds']
        keys = [self._env_step_key] + [
            f'seed{i}/{k}' for k in self._wandb_metrics for i in range(num_seeds)
        ]
        df = result[keys]
        
        metadata = {
            # must be present
            'last_step': last_step,
            'num_seeds': config['num_seeds'],
            'metadata': run.metadata,
            'config': run.config,
            
            # for debugging
            'id': run.id,
            'name': get_dict_value(config, ['logging.exp_name', 'exp_name']),
            'group': get_dict_value(config, ['logging.group', 'wandb_group']),
            'host': run.metadata['host'],
            'runtime_mins': float(run.summary['_runtime'] / 60),
        }
        return metadata, df
