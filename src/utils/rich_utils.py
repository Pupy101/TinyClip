from pathlib import Path
from typing import Any, List, Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, is_rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = ("data", "model", "callbacks", "logger", "trainer", "paths", "extras"),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue: List[Any] = []
    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            log.warning("Field '%s' not found in config. Skipping '%s' config printing...", field, field)

    for cfg_field in cfg:
        if cfg_field not in queue:
            queue.append(cfg_field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    if not cfg.get("tags"):
        hydra_cfg = HydraConfig().cfg
        assert hydra_cfg is not None
        if "id" in hydra_cfg.hydra.job:  # type: ignore[attr-defined]
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags_lst = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags_lst

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)
