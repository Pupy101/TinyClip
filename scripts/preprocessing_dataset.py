import argparse
import json
import re

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

# this scrip is write for dataset from kaggle:
# https://www.kaggle.com/mrviswamitrakaushik/image-captioning-data
# https://www.kaggle.com/ashish2001/original-flickr8k-dataset
# https://www.kaggle.com/adityajn105/flickr30k


def check_exist(path: Union[str, Path]) -> bool:
    """Check file exists."""
    if isinstance(path, str):
        path = Path(path)
    return path.exists()


def create_dataset(data: Dict[str, Any], directories: List[Path]) -> pd.DataFrame:
    """
    Create csv with 2 columns - path to image and text description.

    Args:
        data (Dict[str, Any]): json data about dataframe
        directory (Dict[str, str]): directories with images

    Returns:
        pd.DataFrame: preprocessed coco dataset
    """
    assert "images" in data, "Strange json"
    path_to_images = []
    text_descriptions = []
    parts = []
    print("Preprocess COCO dataset")
    for image in data["images"]:
        image_name = image["filename"]
        text_description = [_["raw"] for _ in image["sentences"]]
        part = image["split"]
        path = None
        for directory in directories:
            if check_exist(directory / image_name):
                path = directory / image_name
                break
        if path is not None:
            path_to_images.extend([str(path)] * len(text_description))
            parts.extend([part] * len(text_description))
            text_descriptions.extend(text_description)
    return pd.DataFrame(
        {"image": path_to_images, "text": text_descriptions, "part": parts}
    )


def open_json(json_path: Union[str, Path]) -> Dict[str, Any]:
    assert check_exist(json_path), f"File {json_path} doesn't exists"
    with open(json_path) as file:
        data = json.load(file)
    return data


def union_jsons(jsons: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Union many jsons into one."""
    assert len(jsons) > 0, "Get one or more jsons"
    copy_json = deepcopy(jsons[0])
    for data in jsons[1:]:
        copy_json["images"].append(data["images"])
        copy_json["dataset"] += "/" + data["dataset"]
    return copy_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsons", type=str, help="Directory with jsons")
    parser.add_argument(
        "--dirs", type=str, help="Directories with images separated whitespace"
    )
    parser.add_argument("--target", type=str, help="Directory for resulting .csv")
    args = parser.parse_args()

    target_dir = Path(args.target)
    assert not check_exist(target_dir) or target_dir.is_dir()

    # find labels and create dataset
    jsons_dir = Path(args.jsons)
    jsons = list(jsons_dir.glob("*.json"))

    images_dirs = [Path(d) for d in re.findall(r"[^\s]+", args.dirs) if check_exist(d)]

    all_json = union_jsons([open_json(j) for j in jsons])

    dataframe = create_dataset(all_json, images_dirs)
    print(f"Overall count of pairs image-text after filter is {dataframe.shape[0]}")

    # train/valid/test split
    train_index = dataframe.part == "train"
    valid_index = dataframe.part == "valid"
    test_index = dataframe.part == "test"

    train = dataframe[train_index]
    valid = dataframe[valid_index]
    test = dataframe[test_index]

    print(f"Train size is {train.shape[0]}")
    print(f"validation samples is {valid.shape[0]}")
    print(f"Test samples is {valid.shape[0]}")

    target_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv((target_dir / "train.csv"), index=False)
    valid.to_csv((target_dir / "valid.csv"), index=False)
    test.to_csv((target_dir / "test.csv"), index=False)
