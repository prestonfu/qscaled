from dataclasses import dataclass, field
from typing import Dict, List, Optional

from qscaled.wandb_utils.base_collector import BaseCollector


@dataclass(kw_only=True)
class BaseConfig:
    name: str  # Name of the experiment, used for zip filename
    max_returns: Dict[str, float]  # Maximum returns per environment
    returns_key: str  # Logging key for returns
    thresholds: List[int] = field(
        default_factory=lambda: [100, 200, 300, 400, 500, 600, 700, 800]
    )  # Return thresholds out of 1000
    wandb_collector: Optional[BaseCollector] = None  # Wandb run collector; None if loading from zip directly
    env_step_freq: Optional[int] = None  # Takes Wandb data every n steps
    env_step_start: Optional[int] = None  # Takes Wandb data starting from this step


@dataclass(kw_only=True)
class SweepConfig(BaseConfig):
    utds_to_predict: List[float]  # UTDs to predict hyperparams for
    baseline_utd_at: float | str = (
        'middle'  # UTD to use for baseline hyperparams; 'middle' approximates geo mean of utds_to_predict
    )


@dataclass(kw_only=True)
class FittedConfig(BaseConfig):
    sweep_name: str  # Copied from sweep config
    sweep_slope_type: str  # 'separate' or 'shared'
    model_size: int  # Number of critic parameters
    budget_delta: float  # TODO: figure out what to put here
    budget_extrapolate_top_k: int  # Number of performance thresholds to extrapolate optimal UTD
