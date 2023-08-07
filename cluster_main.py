#!/usr/bin/env python3

import argparse
import pprint
import torch

from config import get_cluster_configs, load_configs, process_cfg_args
from two_step_zoo import get_clustering_module, get_clustering_trainer, Writer, get_clusterer,get_id_estimator
from two_step_zoo.datasets.loaders import get_loaders_from_config

parser = argparse.ArgumentParser(description="Clustering Manifold Density Estimator")

parser.add_argument("--gpu-id", type=int, default=-1,
    help="GPU to use")

parser.add_argument("--dataset", type=str,
    help="Dataset to train on. Required if load-dir not specified.")
parser.add_argument("--gae-model", type=str,
    help="Model for generalized autoencoding. Required if load-dir not specified.")
parser.add_argument("--de-model", type=str,
    help="Model for density estimation. Required if load-dir not specified.")

parser.add_argument("--load-dir", type=str, default="",
    help="Directory to load from.")
parser.add_argument("--load-best-valid-first", action="store_true",
    help="Attempt to load the best_valid checkpoint first.")

parser.add_argument("--max-epochs-loaded", type=int,
    help="New maximum shared epochs for loaded model.")
parser.add_argument("--max-epochs-loaded-gae", type=int,
    help="New maximum epochs for loaded GAE model.")
parser.add_argument("--max-epochs-loaded-de", type=int,
    help="New maximum epochs for loaded DE model.")

parser.add_argument("--gae-config", default=[], action="append",
    help="Override gae config entries. Specify as `key=value`.")
parser.add_argument("--de-config", default=[], action="append",
    help="Override de config entries. Specify as `key=value`.")
parser.add_argument("--shared-config", default=[], action="append",
    help="Override shared config entries. Specify as `key=value`.")
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
    gae_cfg, de_cfg,shared_cfg,cluster_cfg = load_configs(
        args=args,
        density_estimator=args.de_model if args.de_model else None,
        cluster_model=True,
        cfg_types=["gae", "de", "shared", "cluster"]
    )
else:
    gae_cfg, de_cfg, shared_cfg, cluster_cfg = get_cluster_configs(
        dataset=args.dataset,
        generalized_autoencoder=args.gae_model,
        density_estimator=args.de_model
    )

    gae_cfg = process_cfg_args(gae_cfg, args.gae_config, args)
    de_cfg = process_cfg_args(de_cfg, args.de_config, args)
    shared_cfg = process_cfg_args(shared_cfg, args.shared_config, args)
    cluster_cfg = process_cfg_args(cluster_cfg, args.cluster_config, args)


pprint.sorted = lambda x, key=None: x
pp = pprint.PrettyPrinter(indent=4)
print(10*"-" + "-gae_cfg--" + 10*"-")
pp.pprint(gae_cfg)
print(10*"-" + "--de_cfg--" + 10*"-")
pp.pprint(de_cfg)
print(10*"-" + "shared_cfg" + 10*"-")
pp.pprint(shared_cfg)
print(10*"-" + "cluster_cfg" + 10*"-")
pp.pprint(cluster_cfg)

train_loader, valid_loader, test_loader = get_loaders_from_config(shared_cfg, device)

if args.load_dir:
    # NOTE: In this case, operate in the existing directory
    writer = Writer(
        logdir=args.load_dir,
        make_subdir=False,
        tag_group=args.dataset,
        run_name=args.run_name
    )
else:
    writer = Writer(
        logdir=shared_cfg["logdir_root"],
        make_subdir=True,
        tag_group=args.dataset,
        run_name=args.run_name
    )
writer.write_json(tag="gae_config", data=gae_cfg)
writer.write_json(tag="de_config", data=de_cfg)
writer.write_json(tag="shared_config", data=shared_cfg)
writer.write_json(tag="cluster_cfg", data=cluster_cfg)

clusterer = get_clusterer(cluster_cfg, writer, device, transforms=shared_cfg["transforms"])
clusterer.initialize_clusters(train_loader, valid_loader, test_loader)

if cluster_cfg["id_estimator"] is not None:
    id_estimator = get_id_estimator(cluster_cfg, writer)

    id_estimates = id_estimator.get_id_estimates(clusterer.cluster_dataloaders)
else:
    id_estimates = [gae_cfg["latent_dim"] for _ in range(cluster_cfg["num_clusters"])]

clustering_module = get_clustering_module(gae_cfg, de_cfg, shared_cfg, cluster_cfg, clusterer, id_estimates=id_estimates).to(device)

trainer = get_clustering_trainer(
    clustering_module=clustering_module,
    writer=writer,
    gae_cfg=gae_cfg,
    de_cfg=de_cfg,
    shared_cfg=shared_cfg,
    cluster_cfg=cluster_cfg,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    only_test=args.only_test
)
trainer.train()