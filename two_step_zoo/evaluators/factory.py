import inspect

from .evaluator import Evaluator, metric_fn_dict
from ..datasets import get_loaders

import pdb


def get_evaluator(module, *, valid_loader, test_loader, valid_metrics, test_metrics, **kwargs):
    # Construct the metric_kwargs argument to the evaluator
    metric_kwargs = {}
    valid_metrics = set(valid_metrics)
    test_metrics = set(test_metrics)

    for metric in valid_metrics | test_metrics:
        metric_fn = metric_fn_dict[metric]
        metric_argspec = inspect.getfullargspec(metric_fn)

        metric_kwargs[metric] = {}

        # Allocate kwargs into metric_kwargs
        for i, arg in enumerate(metric_argspec.args + metric_argspec.kwonlyargs):
            if i < 2:
                # This is a default metric arg (module or dataloader)
                continue
            elif arg in kwargs:
                # Arg has been passed in to metric; add it to metric kwargs
                metric_kwargs[metric][arg] = kwargs[arg]
            elif i < len(metric_argspec.args) - len(metric_argspec.defaults):
                # Arg is required for metric but not passed through kwargs
                raise ValueError(f"Argument {arg} is required for metric {metric}")

    return Evaluator(
        module,
        valid_loader=valid_loader,
        test_loader=test_loader,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        metric_kwargs=metric_kwargs,
    )


def get_ood_evaluator(
        module,
        device,
        cfg,
        include_low_dim,
        valid_loader,
        test_loader,
        train_loader,
        savedir
):
    if "likelihood_ood_acc" not in cfg["test_metrics"]:
        cfg["test_metrics"].append("likelihood_ood_acc")
    if include_low_dim and "likelihood_ood_acc_low_dim" not in cfg["test_metrics"]:
        cfg["test_metrics"].append("likelihood_ood_acc_low_dim")

    ood_dataset_map = {
        "mnist": "fashion-mnist",
        "fashion-mnist": "mnist",
        "svhn": "cifar10",
        "cifar10": "svhn"
    }

    oos_train_loader, _, oos_test_loader = get_loaders(
        dataset=ood_dataset_map[cfg["dataset"]],
        device=device,
        data_root=cfg["data_root"],
        make_valid_loader=cfg["make_valid_loader"],
        train_batch_size=cfg["train_batch_size"],
        valid_batch_size=cfg.get("valid_batch_size", None),
        test_batch_size=cfg["test_batch_size"]
    )

    return get_evaluator(
        module,
        valid_loader=valid_loader, test_loader=test_loader, train_loader=train_loader,
        valid_metrics=cfg["valid_metrics"],
        test_metrics=cfg["test_metrics"],
        is_test_loader=test_loader, oos_test_loader=oos_test_loader,
        is_train_loader=train_loader, oos_train_loader=oos_train_loader,
        savedir=savedir,
        **cfg.get("metric_kwargs", {}),
    )