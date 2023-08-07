import os
import torch

from config import load_configs_from_run_dir, load_config_from_run_dir
from two_step_zoo.datasets.loaders import get_loaders_from_config
from two_step_zoo import get_two_step_module, get_single_module, Writer, get_trainer
from two_step_zoo.trainers.factory import get_single_trainer
from two_step_zoo.evaluators import Evaluator


def load_run(run_dir):
    if os.path.exists(os.path.join(run_dir, "config.json")):
        return load_single_module(run_dir)
    elif os.path.exists(os.path.join(run_dir, "shared_config.json")):
        return load_twostep_module(run_dir)
    else:
        raise FileNotFoundError(f"{run_dir} has neither `config.json` nor `shared_config.json`")


def get_writer(run_dir, cfg):
    return Writer(
        logdir=run_dir,
        make_subdir=False,
        tag_group=cfg["dataset"]
    )


def load_single_module(run_dir):
    cfg = load_config_from_run_dir(run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, valid_loader, test_loader = get_loaders_from_config(cfg, device)

    data_dim = 784 if cfg["dataset"] in ["mnist", "fashion-mnist"] else 3072
    data_shape = (1, 28, 28) if cfg["dataset"] in ["mnist", "fashion-mnist"] else (3, 32, 32)
    module = get_single_module(
        cfg,
        train_dataset_size=cfg["train_dataset_size"],
        data_dim=data_dim,
        data_shape=data_shape
    ).to(device)

    writer = get_writer(run_dir, cfg)

    trainer = get_single_trainer(
        module=module,
        writer=writer,
        cfg=cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        # TODO: Add in evaluators with evaluator factory
        evaluator=None
    )

    try:
        trainer.load_checkpoint("best_valid")
    except FileNotFoundError:
        trainer.load_checkpoint("latest")

    return {
        "module": module,
        "trainer": trainer,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
        # TODO: Add here as required
    }


def load_twostep_module(run_dir):
    gae_cfg, de_cfg, shared_cfg = load_configs_from_run_dir(run_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, valid_loader, test_loader = get_loaders_from_config(shared_cfg, device)

    two_step_module = get_two_step_module(gae_cfg, de_cfg, shared_cfg).to(device)

    writer = get_writer(run_dir, shared_cfg)

    trainer = get_trainer(
        two_step_module=two_step_module,
        writer=writer,
        gae_cfg=gae_cfg,
        de_cfg=de_cfg,
        shared_cfg=shared_cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        # TODO: Add in evaluators with evaluator factory
        gae_evaluator=None,
        de_evaluator=None,
        shared_evaluator=Evaluator(two_step_module, valid_loader=None, test_loader=None),
        load_best_valid_first=True,
        pretrained_gae_path="",
        freeze_pretrained_gae=None
    )

    # NOTE: Checkpoint loaded by default for two step module in this case
    return {
        "module": two_step_module,
        "trainer": trainer,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader
        # TODO: Add here as required
    }
