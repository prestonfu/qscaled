import os
import numpy as np
import pandas as pd
import subprocess
from functools import reduce
from zipfile import ZipFile

from qscaled import QSCALED_PATH
from qscaled.utils.configs import BaseConfig
from qscaled.wandb_utils.base_collector import BaseCollector

np.random.seed(42)


def replace_slash_with_period(s: str | None):
    if s is None:
        return s
    return s.replace("/", ".")


def fetch_zip_data(config: BaseConfig, use_cached=True) -> pd.DataFrame:
    """
    If `use_cached==True` and zip file exists, loads from the zip file.
    Otherwise saves data to zip and returns it.
    """
    handler = ZipHandler(config)
    
    if not use_cached or not os.path.exists(f'{handler._zip_path}/{handler._config.name}.zip'):
        collector = handler._config.wandb_collector
        assert collector is not None
        handler.save_prezip()
        handler.save_zip()

    return handler.load_df_from_zip()


class ZipHandler:
    """Handles saving and loading wandb collector offline returns data to/from zip files."""

    def __init__(self, config: BaseConfig):
        self._config = config
        self._prezip_path = f"{QSCALED_PATH}/prezip"
        self._zip_path = f"{QSCALED_PATH}/zip"

    def save_prezip(self):
        """Saves offline returns data to prezip folder using wandb collector."""
        full_path = os.path.join(self._prezip_path, self._config.name)
        os.makedirs(full_path, exist_ok=True)
        collector = self._config.wandb_collector
        data_dict = collector.prepare_zip_export_data(self._config.returns_key, self._config.env_step_freq)
        save_returns_key = replace_slash_with_period(self._config.returns_key)

        for key, data in data_dict.items():
            env, utd, batch_size, learning_rate = key
            filename = f"{self._config.name}/utd_{utd}/{env}/{save_returns_key}/bs_{batch_size}_lr_{learning_rate}.npy"
            np.save(os.path.join(full_path, filename), data)

    def save_zip(self):
        """Saves prezip folder to zip."""
        os.makedirs(self._zip_path, exist_ok=True)
        subprocess.run(
            f"zip -r {self._zip_path}/{self._config.name}.zip {self._config.name}",
            shell=True,
            check=True,
            cwd=os.dirname(self._prezip_path),
        )

    def parse_filename(self, filename):
        """
        Parses previously-saved filenames.

        Example input:
          gym_sweep/utd_1/Ant-v4/episode.return/bs_128_lr_0.0001.npy
        Example output:
          ('Ant-v4', 1.0, 128, 0.0001)
        """
        _, utd_param, env_name, name, params = os.path.splitext(filename)[0].split("/")
        if name != replace_slash_with_period(self._config.returns_key):
            return None
        utd = float(utd_param[len("utd_") :])
        params = params.split("_")  # Example: ['bs', 128, 'lr', 0.0001]
        batch_size = int(params[1])
        learning_rate = float(params[3])
        return env_name, utd, batch_size, learning_rate

    def load_df_from_zip(self) -> pd.DataFrame:
        """
        Loads data from zip file to a DataFrame.

        * If `self._config.env_step_freq is None`, it will load the training
          steps from the first column of the data.
        * Otherwise, it uses env_step_start + k * env_step_freq for k >= 0.
        """
        assert (self._config.env_step_start is None) == (self._config.env_step_freq is None), \
            "Both or neither of env_step_start and env_step_freq should be provided."

        full_path = os.path.join(self._zip_path, f"{self._config.name}.zip")
        records = []

        with ZipFile(full_path, "r") as zip_ref:
            for filename in zip_ref.namelist():
                if filename.endswith(".npy") and not filename.startswith("__MACOSX"):
                    parsed_result = self.parse_filename(filename)
                    if parsed_result is None: continue
                    env_name, utd, batch_size, learning_rate = parsed_result
                    
                    with zip_ref.open(filename, "r") as f:
                        arr = np.load(f)
                    if self._config.env_step_freq is None:
                        step_data = arr[:, 0]
                        returns_data = arr[:, 1:]
                    else:
                        step_data = np.arange(arr.shape[0]) * self._config.env_step_freq + self._config.env_step_start
                        returns_data = arr

                    # Normalize returns to [0, 1000]
                    if env_name in self._config.max_returns:
                        returns_data *= 1000 / self._config.max_returns[env_name]

                    record = {
                        "env_name": env_name,
                        "batch_size": batch_size,
                        "utd": utd,
                        "learning_rate": learning_rate,
                        "training_step": step_data,
                        "return": returns_data,
                        "mean_return": np.mean(returns_data, axis=1),
                        "std_return": np.std(returns_data, axis=1) / np.sqrt(returns_data.shape[1]),  # Standard error
                    }
                    records.append(record)

        return pd.DataFrame(records)
