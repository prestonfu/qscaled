import numpy as np
import pandas as pd
from functools import reduce
from typing import Dict, Tuple, List, Union, Any

from qscaled.wandb_utils import flatten_dict, get_dict_value, get_wandb_run_history
from qscaled.wandb_utils.base_collector import BaseCollector


class OneSeedPerRunCollector(BaseCollector):
    """An example implementation supporting one seed per run."""

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

    def prepare_zip_export_data(self, metric, logging_freq=None) -> Dict[Any, np.ndarray]:
        data_dict = {}
        merge_fn = lambda l, r: pd.merge(l, r, on=self._env_step_key, how='outer')

        for env in self.get_unique('env'):
            for utd in self.get_unique('utd', filter_str=f'env=="{env}"'):
                filtered_rundatas = self.get_filtered_rundatas(f'env=="{env}" and utd=={utd}')

                for key, rundatas in filtered_rundatas.items():
                    bs = key[self.hparam_index['batch_size']]
                    lr = key[self.hparam_index['learning_rate']]
                    save_key = (env, utd, bs, lr)
                    dfs = []

                    for i, df in enumerate(rundatas):
                        df = df[[self._env_step_key, metric]].rename(
                            columns={metric: f'seed{i}/{metric}'}
                        )
                        if logging_freq is not None:
                            df = self._resolve_logging_freq(df, logging_freq)
                        dfs.append(df)

                    result_df = reduce(merge_fn, dfs).dropna()
                    data_dict[save_key] = result_df.to_numpy()

        return data_dict


class ExampleOneSeedPerRunCollector(OneSeedPerRunCollector):
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
        self._wandb_metrics = [
            'episode/return',
            'evaluation/return',
            'training/critic_loss',
            'training/new_data_critic_loss',
            'val/critic_loss',
            'training/policy_data_ce',
            'training/critic_pnorm_l2',
            'training/critic_gnorm_l2',
        ]

    def _generate_key(self, run):
        config_dict = flatten_dict(run.config)
        env = config_dict['env_name']
        utd = config_dict['utd_ratio']
        batch_size = config_dict['batch_size']
        learning_rate = config_dict['agent.critic_lr']
        key = (env, utd, batch_size, learning_rate)  # key is given in same order as `self._hparams`
        return key

    def wandb_fetch(self, run) -> Tuple[Dict[str, Any], pd.DataFrame]:
        last_step = run.summary[self._env_step_key]
        if last_step < 50e3:
            return None
        config = flatten_dict(run.config)
        result = get_wandb_run_history(run)
        if result is None:
            return None
        df = result[[self._env_step_key, *self._wandb_metrics]]
        metadata = {
            'id': run.id,
            'name': get_dict_value(config, ['logging.exp_name', 'exp_name']),
            'group': get_dict_value(config, ['logging.group', 'wandb_group']),
            'runtime_mins': float(run.summary['_runtime'] / 60),
            'last_step': run.summary[self._env_step_key],
        }
        return metadata, df
