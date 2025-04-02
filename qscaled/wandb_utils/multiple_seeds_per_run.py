import numpy as np
import pandas as pd
import re
from functools import reduce
from typing import Dict, Tuple, List, Union, Any

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

    def prepare_zip_export_data(self, metric, logging_freq=None) -> Dict[Any, np.ndarray]:
        data_dict = {}
        merge_fn = lambda l, r: pd.merge(l, r, on=self._env_step_key, how='outer')

        for env in self.get_unique('env'):
            for utd in self.get_unique('utd', env=env):
                filtered_rundatas = self.get_filtered_rundatas(env=env, utd=utd)

                for key, rundatas in filtered_rundatas.items():
                    bs = key[self._hparam_index['batch_size']]
                    lr = key[self._hparam_index['learning_rate']]
                    save_key = (env, utd, bs, lr)
                    dfs = []
                    j = 0

                    for df in rundatas:
                        metric_cols = [
                            col for col in df.columns if re.match(f'^seed\d+/{metric}$', col)
                        ]
                        num_seeds = len(metric_cols)
                        df = df[[self._env_step_key] + metric_cols]
                        df.columns = [self._env_step_key] + [
                            f'seed{j + k}/{metric}' for k in range(num_seeds)
                        ]
                        if logging_freq is not None:
                            df = self._resolve_logging_freq(df, logging_freq)
                        dfs.append(df)
                        j += num_seeds

                    result_df = reduce(merge_fn, dfs).dropna()
                    data_dict[save_key] = result_df.to_numpy()

        return data_dict


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
        self._wandb_seed_metrics = [
            'return',
            'critic_loss',
            'rolling_new_data_critic_loss',
            'rolling_validation_data_critic_loss',
            'critic_pnorm',
            'critic_gnorm',
            'critic_agnorm',
        ]
        self._wandb_global_metrics = []

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
        config = flatten_dict(run.config)
        result = get_wandb_run_history(run)
        if result is None or len(result) < 10:
            return None
        df = result
        num_seeds = config['num_seeds']
        seed_keys = [f'seed{i}/{k}' for k in self._wandb_seed_metrics for i in range(num_seeds)]
        keys = [self._env_step_key] + seed_keys + self._wandb_global_metrics
        df = df[keys]
        metadata = {
            'id': run.id,
            'name': get_dict_value(config, ['logging.exp_name', 'exp_name']),
            'group': get_dict_value(config, ['logging.group', 'wandb_group']),
            'runtime_mins': float(run.summary['_runtime'] / 60),
            'last_step': df[self._env_step_key].iloc[-1],
        }
        return metadata, df
