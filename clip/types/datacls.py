from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from .typings import Device, PathLike, Scheduler


@dataclass
class DownloadFile:
    url: str
    dir: PathLike


@dataclass
class Embeddings:
    image: Tensor
    text: Tensor


@dataclass
class Logits:
    image: Tensor
    text: Tensor


@dataclass
class CLIPTrainOutput:
    embeddings: Embeddings
    logits: Logits


@dataclass
class CLIPInferenceOutput:
    classes: Tensor
    embeddings: Embeddings


@dataclass
class ConvNeXtConfig:
    in_channels: int
    out_channels: int
    drop_path_rate: float
    depths: List[int]
    dims: List[int]
    use_dw_conv: bool = False

    def __post_init__(self) -> None:
        assert len(self.depths) > 1, "Count dims must be more 1"
        assert len(self.depths) == len(self.dims), "Count depths must equal dims"


@dataclass
class DataLoaders:
    train: DataLoader
    valid: DataLoader
    test: DataLoader


@dataclass
class DatasetsPaths:
    train: PathLike
    valid: PathLike
    test: PathLike


@dataclass
class BatchSizes:
    train: int
    valid: int
    test: int


@dataclass
class SplitSizes:
    train: float
    valid: float
    test: float

    def __post_init__(self) -> None:
        assert self.train + self.valid + self.test == 1, "Sum of train/valid/test is equal 1"


@dataclass
class MultiTaskProportions:
    clip: float
    image: float
    text: float

    def __post_init__(self) -> None:
        assert self.clip >= 0
        assert self.image >= 0
        assert self.text >= 0


@dataclass
class MultiTaskCriterions:
    clip: nn.Module
    image: nn.Module
    text: nn.Module


@dataclass
class MultiTaskDataLoaders:
    clip: DataLoaders
    image: DataLoaders
    text: DataLoaders


@dataclass
class TrainConfig:  # pylint: disable=too-many-instance-attributes
    n_epochs: int
    optimizer: optim.Optimizer
    criterion: MultiTaskCriterions
    coefficients: MultiTaskProportions
    device: Device
    save_dir: PathLike
    count_accumukating_steps: int
    scheduler: Optional[Scheduler] = None
    seed: int = 0xFEED

    def __post_init__(self) -> None:
        assert self.count_accumulated_batches >= 1, "Set 1 or more count_accumulated_batches"
        assert self.n_epochs >= 1, "Set 1 or more count training epochs"


__all__ = [
    "DownloadFile",
    "Embeddings",
    "Logits",
    "CLIPTrainOutput",
    "CLIPInferenceOutput",
    "ConvNeXtConfig",
    "DataLoaders",
    "DatasetsPaths",
    "BatchSizes",
    "SplitSizes",
    "MultiTaskProportions",
    "MultiTaskCriterions",
    "MultiTaskDataLoaders",
    "TrainConfig",
]
