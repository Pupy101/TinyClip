from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, TypeVar, Union

Item = TypeVar("Item")


@dataclass
class DownloadFile:
    url: str
    file_path: Union[str, Path]
