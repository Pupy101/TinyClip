from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=["experiment=mobilevit-xx-small-rubert-tiny"],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.callbacks = None
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 0.01
            cfg.trainer.limit_val_batches = 0.1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.sync_batchnorm = False
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.batch_size = 4
            cfg.model.batch_size = 4
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["experiment=mobilevit-xx-small-rubert-tiny", "ckpt_path=."],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.callbacks = None
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 0.1
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.sync_batchnorm = False
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.batch_size = 4
            cfg.model.batch_size = 4
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="function")
def cfg_train(  # type: ignore
    cfg_train_global: DictConfig, tmp_path: Path  # pylint: disable=redefined-outer-name
) -> DictConfig:  # pylint: disable=redefined-outer-name
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="function")
def cfg_eval(  # type: ignore
    cfg_eval_global: DictConfig, tmp_path: Path  # pylint: disable=redefined-outer-name
) -> DictConfig:
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()
