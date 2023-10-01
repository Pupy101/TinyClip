import sys
from typing import Any, Dict, Optional

import pytest
import torch
from packaging.version import Version
from pkg_resources import get_distribution
from pytest import MarkDecorator

from tests.helpers.package_available import _IS_WINDOWS, _SH_AVAILABLE, _TENSORBOARD_AVAILABLE, _WANDB_AVAILABLE


class RunIf:  # pylint: disable=too-few-public-methods
    def __new__(  # type: ignore
        cls,
        min_gpus: int = 0,
        min_torch: Optional[str] = None,
        max_torch: Optional[str] = None,
        min_python: Optional[str] = None,
        skip_windows: bool = False,
        sh: bool = False,
        wandb: bool = False,
        tensorboard: bool = False,
        **kwargs: Dict[Any, Any],
    ) -> MarkDecorator:
        conditions = []
        reasons = []

        if min_gpus:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs>={min_gpus}")

        if min_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) < Version(min_torch))
            reasons.append(f"torch>={min_torch}")

        if max_torch:
            torch_version = get_distribution("torch").version
            conditions.append(Version(torch_version) >= Version(max_torch))
            reasons.append(f"torch<{max_torch}")

        if min_python:
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            conditions.append(Version(py_version) < Version(min_python))
            reasons.append(f"python>={min_python}")

        if skip_windows:
            conditions.append(_IS_WINDOWS)
            reasons.append("does not run on Windows")

        if sh:
            conditions.append(not _SH_AVAILABLE)
            reasons.append("sh")

        if wandb:
            conditions.append(not _WANDB_AVAILABLE)
            reasons.append("wandb")

        if tensorboard:
            conditions.append(not _TENSORBOARD_AVAILABLE)
            reasons.append("tensorboard")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
