import pandas as pd
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
        
    def _set_wandb_metrics(self):
        self._wandb_seed_metrics = [
            'return',
            'critic_loss',
            'rolling_new_data_critic_loss',
            'rolling_validation_data_critic_loss',
            'critic_pnorm',
            'critic_gnorm',
            'critic_agnorm'
        ]
        self._wandb_global_metrics = []
        
    def _generate_key(self, run):
        config = flatten_dict(run.config)
        env = config["env_name"]
        utd = config["utd_ratio"]
        batch_size = config["batch_size"]
        learning_rate = config["agent.critic_lr"]
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
                'runtime_mins': float(run.summary["_runtime"] / 60),
                'last_step': df['_step'].iloc[-1]
            }
            if len(df) >= 10:
                return metadata, df
            else:
                return None
        
        return helper(flatten_dict(run.config))
