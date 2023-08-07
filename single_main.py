#!/usr/bin/env python3

import argparse
import pdb
import pprint
import torch

from config import get_single_config, load_config, get_single_cluster_config, process_cfg_args
from two_step_zoo import (
    get_single_module, get_single_trainer, get_loaders_from_config,
    get_writer, get_evaluator, get_ood_evaluator, get_clusterer, SupervisedDataset, get_loader
)


parser = argparse.ArgumentParser(
    description="Single Density Estimation or Generalized Autoencoder Training Module"
)

parser.add_argument("--gpu-id", type=int, default=-1,
    help="GPU to use")

parser.add_argument("--dataset", type=str,
    help="Dataset to train on. Required if load_dir not specified.")
parser.add_argument("--model", type=str,
    help="Model to train. Required if load_dir not specified.")
parser.add_argument("--is-gae", action="store_true",
    help="Indicates that we are training a generalized autoencoder.")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--max-epochs-loaded", type=int,
    help="New maximum shared epochs for loaded model.")
parser.add_argument("--load-best-valid-first", action="store_true",
    help="Load the best_valid checkpoint first")

parser.add_argument("--config", default=[], action="append",
    help="Override config entries. Specify as `key=value`.")

parser.add_argument("--only-test", action="store_true",
    help="Only perform a test, no training.")

parser.add_argument("--test-ood", action="store_true",
    help="Perform an OOD test.")

parser.add_argument("--run-name", type=str, default=None,
    help="Name of directory to store the run.")

parser.add_argument("--cluster-config", default=[], action="append",
    help="Override shared cluster config entries. Specify as `key=value`.")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available() and args.gpu_id != -1:
    device = f"cuda:{args.gpu_id}"
    
if args.load_dir:
    # NOTE: Not updating config values using cmd line arguments (besides max_epochs)
    #       when loading a run.
    cfg = load_config(
        args=args
    )
else:
    cfg = get_single_config(
        dataset=args.dataset,
        model=args.model,
        gae=args.is_gae,
        standalone=True
    )
    
    cfg = process_cfg_args(cfg, args.config, args)

    cluster_cfg = get_single_cluster_config(
        dataset=args.dataset
    )
    
    cluster_cfg = process_cfg_args(cluster_cfg, args.cluster_config, args)
    cfg = process_cfg_args(cfg, args.config, args)


pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "cfg" + 10*"-")
pp.pprint(cfg)

train_loader, valid_loader, test_loader = get_loaders_from_config(cfg, device)
writer = get_writer(args, cfg=cfg)

# Get labels for conditioning
if cfg["conditioning"] not in [None, "class"]:
    if cfg["conditioning"] == "agglomerative":
        cluster_cfg["cluster_method"] = "agglomerative"
        cluster_cfg["num_clusters"] = cfg["conditioning_dimension"]
        clusterer = get_clusterer(cluster_cfg, writer, device, transforms=cfg["transforms"])
        clusterer.initialize_clusters(train_loader, valid_loader, test_loader)

        new_train_labels = [0 for _ in range(train_loader.dataset.targets.shape[0])]
        new_valid_labels = [0 for _ in range(valid_loader.dataset.targets.shape[0])]
        new_test_labels = [0 for _ in range(test_loader.dataset.targets.shape[0])]

        for split,new_labels in zip(["train", "valid", "test"], [new_train_labels, new_valid_labels, new_test_labels]):
            for cluster_idx in range(len(clusterer.partitions)):
                for image_idx in clusterer.partitions[cluster_idx][split]:
                    new_labels[image_idx] = cluster_idx

        train_dataset = SupervisedDataset("train", "train", train_loader.dataset.inputs, torch.tensor(new_train_labels))
        valid_dataset = SupervisedDataset("valid", "valid", valid_loader.dataset.inputs, torch.tensor(new_valid_labels))
        test_dataset = SupervisedDataset("test", "test", test_loader.dataset.inputs, torch.tensor(new_test_labels))

        train_loader = get_loader(train_dataset, device, cfg["train_batch_size"], drop_last=True)
        valid_loader = get_loader(valid_dataset, device, cfg["valid_batch_size"], drop_last=False)
        test_loader = get_loader(test_dataset, device, cfg["test_batch_size"], drop_last=False)

        cfg["conditioning"] = "class"
    else:
        assert False, "Invalid conditioning method: " + cfg["conditioning"]

module = get_single_module(
    cfg,
    data_dim=cfg["data_dim"],
    data_shape=cfg["data_shape"],
    train_dataset_size=cfg["train_dataset_size"]
).to(device)


if args.test_ood or "likelihood_ood_acc" in cfg["test_metrics"]:
    evaluator = get_ood_evaluator(
        module,
        device,
        cfg=cfg,
        include_low_dim=False,
        valid_loader=valid_loader,
        test_loader=test_loader,
        train_loader=train_loader,
        savedir=writer.logdir
    )
else:
    evaluator = get_evaluator(
        module,
        train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cfg["valid_metrics"],
        test_metrics=cfg["test_metrics"],
        **cfg.get("metric_kwargs", {}),
    )


trainer = get_single_trainer(
    module=module,
    writer=writer,
    cfg=cfg,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    evaluator=evaluator,
    only_test=args.only_test
)

checkpoint_load_list = ["latest", "best_valid"]
if args.load_best_valid_first: checkpoint_load_list = checkpoint_load_list[::-1]

for ckpt in checkpoint_load_list:
    try:
        trainer.load_checkpoint(ckpt)
        break
    except FileNotFoundError:
        print(f"Did not find {ckpt} checkpoint")

trainer.train()