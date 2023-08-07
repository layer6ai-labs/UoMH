import os
import json
import pdb
import ast

from .generalized_autoencoder import GAE_CFG_MAP
from .density_estimator import DE_CFG_MAP
from .shared_config import get_shared_config
from .cluster_config import get_cluster_config

# Add datasets below
_VALID_DATASETS = ["mnist", "fashion-mnist", "cifar10", "cifar100", "svhn", "celeba", "sphere"]

def parse_config_arg(key_value):
    assert "=" in key_value, "Must specify config items with format `key=value`"

    k, v = key_value.split("=", maxsplit=1)
  
    assert k, "Config item can't have empty key"
    assert v, "Config item can't have empty value"

    if "class" in str(v):
        v = str(v)
    else:
        try:
            v = ast.literal_eval(v)
        except ValueError:
            v = str(v)

    return k, v

def add_cluster_cfg_id_metric(cfg, args):
    if "id" in cfg["test_metrics"] or "clustered_id" in cfg["test_metrics"]:
        print("ID metric used, loading default cluster cfg")
        cluster_cfg = get_single_cluster_config(
            dataset=args.dataset
        )
        
        cluster_cfg = {**cluster_cfg, **dict(parse_config_arg(kv) for kv in args.cluster_config)}

        if "metric_kwargs" not in cfg:
            cfg["metric_kwargs"] = {}

        cfg["metric_kwargs"]["cluster_cfg"] = cluster_cfg
    
    return cfg

def process_cfg_args(cfg, cfg_args, args):
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in cfg_args)}
    cfg = add_cluster_cfg_id_metric(cfg, args)
    return cfg

def get_single_config(dataset, model, gae, standalone):
    assert dataset in _VALID_DATASETS, \
        f"Unknown dataset {dataset}"

    cfg_map = GAE_CFG_MAP if gae else DE_CFG_MAP
    base_config = cfg_map["base"](dataset, standalone)

    try:
        model_config_function = cfg_map[model]
    except KeyError:
        cfg_map.pop("base")
        raise ValueError(
            f"Invalid model {model} for {'GAE' if gae else 'DE'}. "
            + f"Valid choices are {cfg_map.keys()}."
        )
    
    cfg = {
        **base_config,

        "dataset": dataset,
        "model": model,

        "gae": gae,

        **model_config_function(dataset, standalone)
    }
    
    return cfg

def get_single_cluster_config(dataset):
    return get_cluster_config(dataset)

    
def get_cluster_configs(dataset, generalized_autoencoder, density_estimator):
    gae_cfg, de_cfg, shared_cfg = get_configs(dataset, generalized_autoencoder, density_estimator)

    cluster_cfg = get_cluster_config(dataset)

    return gae_cfg, de_cfg, shared_cfg, cluster_cfg

def get_configs(dataset, generalized_autoencoder, density_estimator):
    gae_cfg = get_single_config(dataset, generalized_autoencoder, True, False)
    de_cfg = get_single_config(dataset, density_estimator, False, False)

    shared_cfg = get_shared_config(dataset)

    shared_cfg["generalized_autoencoder"] = generalized_autoencoder
    shared_cfg["density_estimator"] = density_estimator

    de_cfg["data_dim"] = gae_cfg["latent_dim"]
    
    return gae_cfg, de_cfg, shared_cfg

def load_configs_from_run_dir(run_dir, cfg_types=["gae", "de", "shared", "cluster"]):
    cfgs = []

    for cfg_type in cfg_types:
        if cfg_type == "_":
            cfg_name = "config.json"
        elif "cluster" in cfg_type:
            cfg_name = "cluster_cfg.json"
        else:
            cfg_name = f"{cfg_type}_config.json"

        with open(os.path.join(run_dir, cfg_name), "r") as f:
            cfgs.append(json.load(f))

    return cfgs if len(cfgs) > 1 else cfgs[0]

def load_configs(args, density_estimator=None, cluster_model=False, cfg_types=["gae", "de", "shared"]):
    
    if not cluster_model and args.load_pretrained_gae: # NOTE: loading pretrained gae not supported for cluster models

        try:
            with open(os.path.join(args.load_dir, "gae_config.json"), "r") as f:
                gae_cfg = json.load(f)
        except FileNotFoundError:
            with open(os.path.join(args.load_dir, "config.json"), "r") as f:
                gae_cfg = json.load(f)

        _, de_cfg, shared_cfg = get_configs(gae_cfg["dataset"], gae_cfg["model"], density_estimator)

        de_cfg["data_dim"] = gae_cfg["latent_dim"]

        cfgs = [gae_cfg, de_cfg, shared_cfg]

        if "cluster" in cfg_types:
            cluster_cfg = get_cluster_config(gae_cfg["dataset"])
            cfgs.append(cluster_cfg)

    else:
        cfgs = load_configs_from_run_dir(args.load_dir, cfg_types=cfg_types)

    if args.max_epochs_loaded_gae:
        cfgs[0]["max_epochs"] = args.max_epochs_loaded_gae
    if args.max_epochs_loaded_de:
        cfgs[1]["max_epochs"] = args.max_epochs_loaded_de
    if args.max_epochs_loaded:
        cfgs[2]["max_epochs"] = args.max_epochs_loaded

    return cfgs


def load_config(args, cfg_types=["_"]):
    cfg = load_configs_from_run_dir(args.load_dir, cfg_types=cfg_types)

    if args.max_epochs_loaded:
        cfg["max_epochs"] = args.max_epochs_loaded

    return cfg