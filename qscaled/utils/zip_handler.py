import os
import numpy as np
import pandas as pd
import subprocess
from functools import reduce

from qscaled import QSCALED_PATH
from qscaled.utils.configs import BaseConfig
from qscaled.wandb_utils.base_collector import BaseCollector


def replace_slash_with_period(s: str | None):
    if s is None:
        return s
    return s.replace("/", ".")

import os
import numpy as np
import pandas as pd

from zipfile import ZipFile
from typing import Dict, Tuple

np.random.seed(42)


class ZipHandler:
    """Handles saving and loading data to/from zip files."""
    
    def __init__(self, config: BaseConfig):
        self._config = config
    
    def save_and_load(self):
        collector = self._config.wandb_collector
        
        assert collector is not None \
            or os.path.exists(f'{QSCALED_PATH}/zip/{self._config.name}.zip'), \
            "Either wandb_collector must be provided or zip file must exist."
        
        # Save data to zip using wandb collector
        if self._config.wandb_collector is not None:
            os.makedirs(f"{QSCALED_PATH}/zip", exist_ok=True)
            (self._config.wandb_collector, self._config.return_key, self._config.name, self._config.logging_freq)
            subprocess.run(
                f'zip -r {self._config.name}.zip {self._config.name} && mv {self._config.name}.zip {QSCALED_PATH}/zip',
                shell=True,
                check=True,
                cwd=f'{QSCALED_PATH}/prezip',
            )

        zip_loader = self._config.zip_load_cls(self._config.max_returns, replace_slash_with_period(self._config.return_key))
        return zip_loader.load(f"{QSCALED_PATH}/zip/{self._config.name}.zip", self._config.env_step_freq, self._config.env_step_start)


    def save_data(collector, path, varname, logging_freq=None):
        """Creates a directory and saves aggregate data for zipping."""
        dirname = f'{QSCALED_PATH}/prezip/{path}/utd_{utd}/{env_name}/{replace_slash_with_period(varname)}'
        os.makedirs(dirname, exist_ok=True)
        data_dict = 
        for key, value in data_dict.items():
            env_name, utd, bs, lr = key
            name = f'bs_{bs}_lr_{lr}'
            np.save(f'{dirname}/{name}', value)


    def save_returns_as_zip(collector, return_key, path, logging_freq=None):
        for env in collector.get_unique('env'):
            for utd in collector.get_unique('utd', env=env):
                for varname in [return_key]:
                    save_data(collector, path, env, utd, varname, logging_freq)
                    
    def parse_filename(self, filename):
        """
        Parses previously-saved filenames.
        
        Example input 1:
          gym_sweep/utd_1/Ant-v4/episode.return/bs_128_lr_0.0001.npy
        Example output 1:
          ('Ant-v4', 1.0, 128, 0.0001)
          
        Example input 2:
          dmc_ours/utd_0.5/cartpole-swingup/online_returns/bs_528_lr_0.000902_reset_True.npy
        Example output 2:
          ('cartpole-swingup', 0.5, 528, 0.000902)
        """
        _, utd_param, env_name, name, params = os.path.splitext(filename)[0].split('/')
        if name != self.return_key: 
            return None
        utd = float(utd_param[len('utd_'):])
        params = params.split('_')  # Example: ['bs', 128, 'lr', 0.0001]
        batch_size = int(params[1])
        learning_rate = float(params[3])
        return env_name, utd, batch_size, learning_rate
    
    def load(self, zip_path: str, manual_step_freq=None, manual_step_start=None) -> pd.DataFrame:
        """
        Loads data from zip file to a DataFrame.
        
        * If `manual_step is None`, it will load the training steps from the first
          column of the data. 
        * Otherwise, it uses manual_step_start + k * manual_step_freq
          for k >= 0. 
        """
        records = []
        assert (manual_step_start is None) == (manual_step_freq is None), \
            "Both or neither of manual_step_start and manual_step_freq should be provided."
        
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
                    
                    # Normalize returns to [0, 1000]
                    if env_name in self.max_returns:
                        arr *= 1000 / self.max_returns[env_name]
                        
                    record = {
                        'env_name': env_name,
                        'batch_size': batch_size,
                        'utd': utd,
                        'learning_rate': learning_rate,
                    }
                    
                    if manual_step_freq is None:
                        step_data = arr[:, 0]
                        run_data = arr[:, 1:]
                    else:
                        step_data = np.arange(arr.shape[0]) * manual_step_freq + manual_step_start
                        run_data = arr

                    record.update({
                        'training_step': step_data,
                        'return': run_data,
                        'mean_return': np.mean(run_data, axis=1),
                        'std_return': np.std(run_data, axis=1) / np.sqrt(run_data.shape[1])  # Standard error
                    })
                    records.append(record)
        
        return pd.DataFrame(records)
