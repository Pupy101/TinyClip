from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional


class TypeUsing(Enum):
    TRAIN = 'train'
    EVALUATION = 'evaluation'


class DatasetType(Enum):
    DEFAULT = 'default'
    CACHED = 'cached'
    URL = 'url'


@dataclass
class TrainValidParameters:
    train: Any
    valid: Any


@dataclass
class CLIPDatasets(TrainValidParameters):
    """
    Paths to csv with pairs image - description
    """
    train: Path
    valid: Optional[Path] = None


@dataclass
class LoaderParameters:
    """
    Parameters for torch.utils.data.DataLoader
    """
    batch_size: int
    shuffle: bool
    num_workers: int = 2


@dataclass
class CLIPLoaders(TrainValidParameters):
    train: LoaderParameters
    valid: Optional[LoaderParameters] = None


@dataclass
class ModelWeight:
    save: Path
    pretrained: Optional[Path] = None


@dataclass
class InferenceParameters:
    image_dir: Path
    prediction_dir: Path
    classes: List[str]
