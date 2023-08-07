#!/usr/bin/env python3

import argparse
import pdb
import pprint
import torch

from config import get_single_config, load_config,get_single_cluster_config,process_cfg_args
from two_step_zoo import (
    get_writer,get_single_clustering_trainer,get_clusterer, get_single_clustering_module)
from two_step_zoo.datasets.loaders import get_loaders_from_config
from two_step_zoo.factory import get_single_module


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
parser.add_argument("--cluster-config", default=[], action="append",
    help="Override shared config entries. Specify as `key=value`.")

parser.add_argument("--only-test", action="store_true",
    help="Only perform a test, no training.")

parser.add_argument("--test-ood", action="store_true",
    help="Perform an OOD test.")

parser.add_argument("--run-name", type=str, default=None,
    help="Name of directory to store the run.")

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available() and args.gpu_id != -1:
    device = f"cuda:{args.gpu_id}"

if args.load_dir:
    # NOTE: Not updating config values using cmd line arguments (besides max_epochs)
    #       when loading a run.
    cfg,cluster_cfg = load_config(
        args=args,
        cfg_types=["_","cluster"]
    )
else:
    cfg = get_single_config(
        dataset=args.dataset,
        model=args.model,
        gae=args.is_gae,
        standalone=True
    )
    cluster_cfg = get_single_cluster_config(
        dataset=args.dataset,
    )

    cfg = process_cfg_args(cfg, args.config, args)
    cluster_cfg = process_cfg_args(cluster_cfg, args.cluster_config, args)


pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "cfg" + 10*"-")
pp.pprint(cfg)
pp.pprint(cluster_cfg)

# NOTE: Using more default arguments here because not using a shared_cfg.
#       (Compare with main.py)

train_loader, valid_loader, test_loader = get_loaders_from_config(cfg, device)
writer = get_writer(args, cfg=cfg)

writer.write_json(tag="config", data=cfg)
writer.write_json(tag="cluster_cfg", data=cluster_cfg)

clusterer = get_clusterer(cluster_cfg, writer, device, transforms=cfg["transforms"])
clusterer.initialize_clusters(train_loader, valid_loader, test_loader)

clustering_module = get_single_clustering_module(cfg, cluster_cfg, clusterer, get_single_module, args.run_name).to(device)
clustering_module.cluster_module_device = device

if cfg["early_stopping_metric"] == "fid" and "fid" not in cfg["valid_metrics"]:
    cfg["valid_metrics"].append("fid")

trainer = get_single_clustering_trainer(
    clustering_module=clustering_module,
    writer=writer,
    cfg=cfg,
    cluster_cfg=cluster_cfg,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    only_test=args.only_test
)


trainer.train()