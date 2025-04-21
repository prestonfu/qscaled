from __future__ import annotations

import os
import numpy as np
import pandas as pd
import subprocess
from zipfile import ZipFile
from typing import Union, TYPE_CHECKING

from qscaled.constants import QSCALED_PATH
from qscaled.utils.state import remove_with_prompt

if TYPE_CHECKING:
    from qscaled.utils.configs import BaseConfig


def fetch_zip_data(config: BaseConfig, use_cached=True, quiet=True) -> pd.DataFrame:
    """
    If `use_cached==True` and zip file exists, loads from the zip file directly.
    Otherwise writes data to zip using the wandb collector, and then loads from the zip file.
    """
    handler = config.zip_handler_cls(config)
    prezip_dir = os.path.join(handler._prezip_path, handler._config.name)
    zip_path = os.path.join(handler._zip_path, f'{handler._config.name}.zip')

    if not use_cached or not os.path.exists(zip_path):
        collector = handler._config.wandb_collector
        assert collector is not None, (
            'Wandb collector must be provided if cache is unused or does not exist.'
        )
        remove_with_prompt(zip_path, prezip_dir)
        handler.save_prezip()
        handler.save_zip(quiet)

    return handler.load_df_from_zip()


class BaseZipHandler:
    """Base class for handling wandb collector offline returns data to/from zip files."""

    def __init__(self, config: BaseConfig):
        self._config = config
        self._prezip_path = f'{QSCALED_PATH}/prezip'
        self._zip_path = f'{QSCALED_PATH}/zip'

    def save_zip(self, quiet=True):
        """Saves prezip folder to zip."""
        os.makedirs(self._zip_path, exist_ok=True)
        if quiet:
            quiet_kw = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            quiet_kw = {}
        subprocess.run(
            f'zip -r {self._config.name}.zip {self._config.name} && mv {self._config.name}.zip {self._zip_path} && rm -rf {self._config.name}',
            check=True,
            shell=True,
            cwd=self._prezip_path,
            **quiet_kw
        )

    def _rename_wandb_metric(s: Union[str, None]):
        if s is None:
            return s
        return s.replace('/', '.')


class UTDZipHandler(BaseZipHandler):
    """Handles saving and loading wandb collector offline returns data to/from zip files."""

    def __init__(self, config: BaseConfig):
        super().__init__(config)

    def save_prezip(self):
        """Saves offline returns data to prezip folder using wandb collector."""
        collector = self._config.wandb_collector
        data_dict = collector.prepare_zip_export_data(
            self._config.returns_key, self._config.logging_freq
        )
        save_returns_key = UTDZipHandler._rename_wandb_metric(self._config.returns_key)

        for key, data in data_dict.items():
            env, utd, batch_size, learning_rate = key
            filename = f'{self._config.name}/utd_{utd}/{env}/{save_returns_key}/bs_{batch_size}_lr_{learning_rate}.npy'
            full_path = os.path.join(self._prezip_path, filename)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            np.save(full_path, data)

    def parse_filename(self, filename):
        """
        Parses previously-saved filenames.

        Example input:
          gym_sweep/utd_1/Ant-v4/episode.return/bs_128_lr_0.0001.npy
        Example output:
          ('Ant-v4', 1.0, 128, 0.0001)
        """
        _, utd_param, env_name, name, params = os.path.splitext(filename)[0].split('/')
        if name != UTDZipHandler._rename_wandb_metric(self._config.returns_key):
            return None
        utd = float(utd_param[len('utd_') :])
        params = params.split('_')  # Example: ['bs', 128, 'lr', 0.0001]
        batch_size = int(params[1])
        learning_rate = float(params[3])
        return env_name, utd, batch_size, learning_rate

    def load_df_from_zip(self) -> pd.DataFrame:
        """Loads data from zip file to a DataFrame."""
        full_path = os.path.join(self._zip_path, f'{self._config.name}.zip')
        records = []

        with ZipFile(full_path, 'r') as zip_ref:
            for filename in zip_ref.namelist():
                if filename.endswith('.npy') and not filename.startswith('__MACOSX'):
                    parsed_result = self.parse_filename(filename)
                    if parsed_result is None:
                        continue
                    env_name, utd, batch_size, learning_rate = parsed_result

                    with zip_ref.open(filename, 'r') as f:
                        arr = np.load(f)

                    step_data = arr[:, 0]
                    returns_data = arr[:, 1:]
                    if env_name in self._config.max_returns:
                        returns_data *= 1000 / self._config.max_returns[env_name]

                    record = {
                        'env_name': env_name,
                        'batch_size': batch_size,
                        'utd': utd,
                        'learning_rate': learning_rate,
                        'training_step': step_data,
                        'return': returns_data,
                        'mean_return': np.mean(returns_data, axis=1),
                        'std_return': np.std(returns_data, axis=1)
                        / np.sqrt(returns_data.shape[1]),  # Standard error
                    }
                    records.append(record)

        if len(records) == 0:
            raise ValueError('No data found in zip file.')

        return pd.DataFrame(records)
