import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, is_rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as exc:
            log.exception("")
            raise exc

        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")
            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(  # pylint: disable=broad-exception-raised
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
