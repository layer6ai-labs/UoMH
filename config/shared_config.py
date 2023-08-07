import math
import torch 

def get_shared_config(dataset):
    return {
        "dataset": dataset,

        "sequential_training": True,
        "alternate_by_epoch": False,

        "max_epochs": 100,
        "early_stopping_metric": None,
        "max_bad_valid_epochs": None,
        "max_grad_norm": None,

        "make_valid_loader": True,
        "valid_fraction": 0.1,

        "epoch_sample_every": math.inf,

        "lr_scheduler": "cosine",
        "lr_scheduler_step": 100, # in epochs
        "lr_scheduler_gamma": 0.1, # used for step scheduler

        "data_root": "data/",
        "logdir_root": "runs/",

        "train_batch_size": 128,
        "valid_batch_size": 128,
        "test_batch_size": 128,

        "class_ind": -1,
        "transforms": None,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error"],

        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }