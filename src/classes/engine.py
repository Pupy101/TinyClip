from dataclasses import dataclass


@dataclass
class OneEpochResults:
    mean_loss: float
    recall: float
    precision: float
    f1: float
