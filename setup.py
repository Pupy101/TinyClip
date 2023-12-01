from pathlib import Path
from typing import List

from setuptools import find_packages, setup

REPO_DIR = Path(__file__).parent


def find_requirements() -> List[str]:
    requirements: List[str] = []
    with open(REPO_DIR / "requirements.txt") as fp:
        for line in fp:
            line = line.strip()
            if line:
                requirements.append(line.strip())
    return requirements


setup(
    name="src",
    version="0.0.1",
    description="TinyClip",
    url="https://github.com/Pupy101/TinyClip",
    install_requires=find_requirements(),
    packages=find_packages(),
    entry_points={"console_scripts": ["train_command = src.train:main", "eval_command = src.eval:main"]},
)
