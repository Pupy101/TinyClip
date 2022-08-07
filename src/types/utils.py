from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

Item = TypeVar("Item")


@dataclass
class DownloadFile:
    url: str
    file_path: Path
