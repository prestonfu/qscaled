from dataclasses import dataclass
from typing import ClassVar, Dict, List, Type, Optional

from qscaled.wandb_utils.base_collector import BaseCollector


@dataclass(kw_only=True)
class BaseConfig:
    name: str  # Name of the experiment, used for zip filename
    max_returns: Dict[str, float]  # Maximum returns per environment
    returns_key: str  # Logging key for offline returns
    thresholds: ClassVar[List[int]] = [100, 200, 300, 400, 500, 600, 700, 800]  # Return thresholds out of 1000
    wandb_collector: Optional[BaseCollector] = None  # Wandb run collector; None if loading from zip directly
    env_step_freq: Optional[int] = None  # Takes Wandb data every n steps
    env_step_start: Optional[int] = None  # Takes Wandb data starting from this step


@dataclass(kw_only=True)
class SweepConfig(BaseConfig):
    utds_to_predict: List[float]  # UTDs to predict


@dataclass(kw_only=True)
class FittedConfig(BaseConfig):
    model_size: int  # Number of critic parameters
