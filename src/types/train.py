from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .data import MultiTaskDataLoaders

if TYPE_CHECKING:
    from ..engine.engine import Engine


@dataclass
class MultiTaskProportion:
    clip: float
    image: float
    text: float

    def __post_init__(self) -> None:
        assert 0 <= self.clip
        assert 0 <= self.image
        assert 0 <= self.text


@dataclass
class TrainingParameters:
    n_epochs: int
    engine: "Engine"
    dataloaders: MultiTaskDataLoaders
    save_dir: Path
    coefficients: MultiTaskProportion
