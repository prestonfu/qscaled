import numpy as np
import pandas as pd
import re
from functools import reduce
from typing import Dict, Any, Tuple

from qscaled.wandb_utils.base_collector import BaseCollector
from qscaled.wandb_utils import flatten_dict, retry, get_dict_value


class MultipleSeedsPerRunCollector(BaseCollector):
    """
    An example implementation supporting multiple seeds per run.
    These metrics have keys like `seed0/return`, `seed1/return`, etc.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def _set_hparams(self):
        self._hparams = ['env', 'utd', 'batch_size', 'learning_rate']

    def prepare_zip_export_data(self, metric, logging_freq=None) -> Dict[Any, np.typing.NDArray]:
        """Prepares data to export in a format compatible with UTDGroupedLoader."""
        merge_fn = lambda l, r: pd.merge(l, r, on='_step', how='outer')

        for env in self.get_unique('env'):
            for utd in self.get_unique('utd', env=env):
                filtered_rundatas = self.get_filtered_rundatas(env=env, utd=utd)
                data_dict = {}

                for key, rundatas in filtered_rundatas.items():
                    bs = key[self._hparam_index['batch_size']]
                    lr = key[self._hparam_index['learning_rate']]
                    save_key = (env, utd, bs, lr)
                    j = 0
                    summaries = []

                    for i, df in enumerate(rundatas):
                        metric_cols = [col for col in df.columns if re.match(f'^seed\d+/{metric}$', col)]
                        num_seeds = len(metric_cols)
                        df = df[['_step'] + metric_cols]
                        df.columns = ['_step'] + [f'seed{j + k}/{metric}' for k in range(num_seeds)]
                        summaries.append(df)
                        j += num_seeds

                    merged_df = reduce(merge_fn, summaries)
                    result_df = self._resolve_logging_freq(merged_df, logging_freq) if logging_freq else merged_df
                    data_dict[save_key] = result_df.to_numpy()

        return data_dict


class ExampleMultipleSeedsPerRunCollector(MultipleSeedsPerRunCollector):
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

    def wandb_fetch(self, run, num_tries=5) -> Tuple[Dict[str, Any], pd.DataFrame] | None:
        """Returns run metadata and history. If fails, returns None."""

        @retry(num_tries)
        def helper(config):
            num_seeds = config['num_seeds']
            df = run.history(samples=10000)
            seed_keys = [f'seed{i}/{k}' for k in self._wandb_seed_metrics for i in range(num_seeds)]
            keys = ['_step'] + seed_keys + self._wandb_global_metrics
            df = df[keys]
            metadata = {
                'id': run.id,
                'name': get_dict_value(config, ['logging.exp_name', 'exp_name']),
                'group': get_dict_value(config, ['logging.group', 'wandb_group']),
                'runtime_mins': float(run.summary['_runtime'] / 60),
                'last_step': df['_step'].iloc[-1],
            }
            if len(df) >= 10:
                return metadata, df
            else:
                return None

        return helper(flatten_dict(run.config))
