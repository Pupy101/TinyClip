from dataclasses import dataclass
from math import exp
from typing import Any, Tuple


@dataclass
class BaseMetric:
    loss: float = 0.0
    count: int = 1

    def overall(self) -> Any:
        raise NotImplementedError


@dataclass
class CLIPMetrics(BaseMetric):
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0

    def update(
        self,
        tp: float,
        fp: float,
        fn: float,
        loss: float,
        count: int,
    ) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.loss += loss
        self.count += count

    def overall(self) -> Tuple[float, float, float, float]:
        recall = self.tp / (self.tp + self.fp)
        precision = self.tp / (self.tp + self.fn)
        f1 = 2 * precision * recall / (precision + recall)
        loss = self.loss / self.count
        return recall, precision, f1, loss

    def __str__(self) -> str:
        recall, precision, f1, loss = self.overall()
        return (
            f"\tLoss:       {loss:.3f}\n"
            f"\tRecall:     {recall:.3f}\n"
            f"\tPrecision:  {precision:.3f}\n"
            f"\tF1:         {f1:.3f}"
        )


@dataclass
class ClassificationMetrics(BaseMetric):
    right_classificated_top1: int = 0
    right_classificated_top5: int = 0

    def update(self, top1: int, top5: int, loss: float, count: int) -> None:
        self.right_classificated_top1 += top1
        self.right_classificated_top5 += top5
        self.loss += loss
        self.count += count

    def overall(self) -> Tuple[float, float, float]:
        accuracy_top1 = self.right_classificated_top1 / self.count
        accuracy_top5 = self.right_classificated_top5 / self.count
        loss = self.loss / self.count
        return accuracy_top1, accuracy_top5, loss

    def __str__(self) -> str:
        acc_top1, acc_top5, loss = self.overall()
        return f"\tLoss: {loss:.3f}\n" f"\tAccuracy top 1: {acc_top1:.3f}\n" f"\tAccuracy top 1: {acc_top5:.3f}"


@dataclass
class MaskedLMMetric(BaseMetric):
    right_classificated_top1: int = 0
    right_classificated_top5: int = 0
    perplexity: float = 0.0

    def update(self, top1: int, top5: int, loss: float, count: int) -> None:
        self.right_classificated_top1 += top1
        self.right_classificated_top5 += top5
        self.perplexity += exp(loss)
        self.loss += loss
        self.count += count

    def overall(self) -> Tuple[float, float, float, float]:
        accuracy_top1 = self.right_classificated_top1 / self.count
        accuracy_top5 = self.right_classificated_top5 / self.count
        perplexity = self.perplexity / self.count
        loss = self.loss / self.count
        return accuracy_top1, accuracy_top5, perplexity, loss

    def __str__(self) -> str:
        acc_top1, acc_top5, perplexity, loss = self.overall()
        return (
            f"\tLoss:           {loss:.3f}\n"
            f"\tAccuracy top 1: {acc_top1:.3f}\n"
            f"\tAccuracy top 5: {acc_top5:.3f}\n"
            f"\tPerplexity:     {perplexity:.3f}"
        )
