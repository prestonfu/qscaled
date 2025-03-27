from dataclasses import dataclass
from typing import ClassVar, Dict, List, Type, Optional

from qscaled.utils.load_from_zip import ZipLoader, DefaultZipLoader
from qscaled.wandb_utils.base_collector import BaseCollector


@dataclass(kw_only=True)
class BaseConfig:
    name: str  # Name of the experiment, used for filenames
    max_returns: Dict[str, float]  # Maximum returns per environment
    return_key: str  # Logging key for episode returns
    thresholds: ClassVar[List[int]] = [100, 200, 300, 400, 500, 600, 700, 800]  # Return thresholds out of 1000
    wandb_collector: Optional[BaseCollector] = None  # Wandb run collector; None if loading from zip directly
    zip_load_cls: Type[ZipLoader] = DefaultZipLoader  # Default zip loading class
    env_step_freq: Optional[int] = None  # Takes Wandb data every n steps
    env_step_start: Optional[int] = None  # Takes Wandb data starting from this step


@dataclass(kw_only=True)
class SweepConfig(BaseConfig):
    utds_to_predict: List[float]  # UTDs to predict


@dataclass(kw_only=True)
class FittedConfig(BaseConfig):
    model_size: int  # Number of critic parameters
