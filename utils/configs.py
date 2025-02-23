from dataclasses import dataclass
from typing import ClassVar, Dict, List, Type, Optional

from qscaled.preprocessing import ZipLoader, UTDGroupedLoader
from utils.wandb_utils import BaseRunCollector

@dataclass
class Config:
    name: str  # Name of the experiment, used for filenames
    max_returns: Dict[str, float]  # Maximum returns per environment
    return_key: str  # Logging key for episode returns
    utds_to_predict: List[float]  # UTDs to predict
    thresholds: ClassVar[List[int]] = [100, 200, 300, 400, 500, 600, 700, 800]  # Return thresholds out of 1000
    wandb_collector: Optional[BaseRunCollector] = None  # Wandb run collector; None if loading from zip directly
    logging_freq: Optional[int] = None  # Takes Wandb data every n steps
    zip_load_cls: Type[ZipLoader] = UTDGroupedLoader  # Default zip loading class
