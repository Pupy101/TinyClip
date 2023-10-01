import platform

import pkg_resources


def _package_available(package_name: str) -> bool:
    try:
        return pkg_resources.require(package_name) is not None
    except pkg_resources.DistributionNotFound:
        return False


_IS_WINDOWS = platform.system() == "Windows"

_SH_AVAILABLE = not _IS_WINDOWS and _package_available("sh")

_WANDB_AVAILABLE = _package_available("wandb")
_TENSORBOARD_AVAILABLE = _package_available("tensorboard")
