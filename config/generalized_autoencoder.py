import torch

def get_base_config(dataset, standalone):
    if standalone:
        standalone_info = {
            "train_batch_size": 128,
            "valid_batch_size": 128,
            "test_batch_size": 128,

            "make_valid_loader": True,

            "data_root": "data/",
            "logdir_root": "runs/",

             "class_ind": -1, 
            "transforms": None
        }
    else:
        standalone_info = {}

    return {
        "flatten": True,
        "denoising_sigma": None,
        "dequantize": False,
        "scale_data": True,
        "whitening_transform": False,

        "optimizer": "adam",
        "lr": 0.001,
        "use_lr_scheduler": False,
        "max_epochs": 100,
        "max_grad_norm": 10,

        "conditioning": None,
        "conditioning_dimension": 0,

        "early_stopping_metric": None,
        "max_bad_valid_epochs": None,
        "make_valid_loader": True,
        "valid_fraction": 0.1,

        "lr_scheduler": "cosine",
        "lr_scheduler_step": 100, # in epochs
        "lr_scheduler_gamma": 0.1, # used for step scheduler

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error"],

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        **standalone_info
    }



def get_ae_config(dataset, standalone):
    net = "mlp" if dataset in ["mnist", "fashion-mnist"] else "cnn"

    ae_base = {
        "latent_dim": 20,
    }

    if net == "mlp":
        net_configs = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [200, 200],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [200, 200],

            "flatten": True
        }

    elif net == "cnn":
        net_configs = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False
        }

    return {
        **ae_base,
        **net_configs,
    }


def get_avb_config(dataset):
    if dataset in ["mnist", "fashion-mnist"]:
        net = "mlp"
    elif "imagenet" in dataset:
        net = "residual"
    else:
        net = "cnn"

    avb_base = {
        "early_stopping_metric": None,
        "max_bad_valid_epochs": 10,

        "latent_dim": 20,

        "discriminator_hidden_dims": [256, 256],

        "single_sigma": True,

        "input_sigma": 3.,
        "prior_sigma": 1.,

        "lr": None,
        "disc_lr": 0.001,
        "nll_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_nll_lr_scheduler": True,
    }

    if net == "mlp":
        net_configs = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True,
            "max_epochs": 50,
            "noise_dim": 256,
        }

    elif net == "cnn":
        net_configs = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False,
            "max_epochs": 100,
            "noise_dim": 256,
        }

    return {
        **avb_base,
        **net_configs,
    }


def get_bigan_config(dataset, standalone):
    net = "mlp" if dataset in ["mnist", "fashion-mnist"] else "cnn"

    bigan_base = {
        "early_stopping_metric": "l2_reconstruction_error",
        "max_bad_valid_epochs": 50,

        "latent_dim": 20,

        "discriminator_hidden_dims": [256, 256],
        "num_discriminator_steps": 2,
        "wasserstein": True,
        "clamp": 0.01,
        "gradient_penalty": True,
        "lambda": 10.0,
        "recon_weight": 1.0,

        "optimizer": 'adam',
        "lr": None,
        "disc_lr": 0.0001,
        "ge_lr": 0.0001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_ge_lr_scheduler": True,

        "valid_metrics": ["l2_reconstruction_error"],
        "test_metrics": ["l2_reconstruction_error", "fid"],
    }

    if net == "mlp":
        net_configs = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True,
            "max_epochs": 200,
        }

    elif net == "cnn":
        net_configs = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False,
            "max_epochs": 200,
        }

    return {
        **bigan_base,
        **net_configs,
    }


def get_gan_config(dataset, standalone):
    return {}


def get_vae_config(dataset, standalone):

    if dataset in ["mnist", "fashion-mnist"]:
        net = "mlp"
    elif "imagenet" in dataset:
        net = "residual"
    else:
        net = "cnn"

    vae_base = {
        "latent_dim": 20,

        "single_sigma": True,

        "decoder_variance_lower_bound": 0,

        "base_distribution": "gaussian",
        "num_prior_components": 10,
        "distribution_mean_spacing": 1
    }

    if net == "mlp":
        net_configs = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [256],

            "flatten": True
        }

    elif net == "cnn":
        net_configs = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [32, 32, 16, 16],
            "encoder_kernel_size": [3, 3, 3, 3],
            "encoder_stride": [1, 1, 1, 1],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [16, 16, 32, 32],
            "decoder_kernel_size": [3, 3, 3, 3],
            "decoder_stride": [1, 1, 1, 1],

            "flatten": False
        }
    
    elif net == "residual":
        net_configs = {
            
            "encoder_net": "residual",
            "encoder_layer_channels": [16, 32, 64],
            "encoder_blocks_per_layer": [2, 2, 2],

            "decoder_net": "residual",
            "decoder_layer_channels": [64, 32], # Missing channel is for output channel dim
            "decoder_blocks_per_layer": [2, 2, 2],

            "flatten": False
        }

    return {
        **vae_base,
        **net_configs,
    }


def get_wae_config(dataset, standalone):
    net = "cnn"

    if "imagenet" in dataset:
        net = "residual"

    wae_base = {
        "latent_dim": 20,

        "max_epochs": 300,

        "discriminator_hidden_dims": [256, 256],

        "_lambda": 10.,
        "sigma": 1.,

        "lr": None,
        "disc_lr": 0.0005,
        "rec_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": False,
        "use_rec_lr_scheduler": False,

        "early_stopping_metric": None,
        "max_bad_valid_epochs": 30,

        "base_distribution": None,
        "num_mixture_components": 10,
        "distribution_mean_spacing": 1,

        "valid_metrics": ["l2_reconstruction_error"],
    }

    if net == "mlp":
        net_configs = {
            "encoder_net": "mlp",
            "encoder_hidden_dims": [256, 128],

            "decoder_net": "mlp",
            "decoder_hidden_dims": [128, 256],

            "flatten": True
        }

    elif net == "cnn":
        enc_hidden_channels = [64, 64, 32, 32]
        enc_kernel = [3, 3, 3, 3]
        enc_stride = [1, 1, 1, 1]

        net_configs = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": enc_hidden_channels,
            "encoder_kernel_size": enc_kernel,
            "encoder_stride": enc_stride,

            "decoder_net": "cnn",
            "decoder_hidden_channels": enc_hidden_channels[::-1],
            "decoder_kernel_size": enc_kernel[::-1],
            "decoder_stride": enc_stride[::-1],

            "flatten": False
        }

    elif net == "residual":
        net_configs = {
            
            "encoder_net": "residual",
            "encoder_layer_channels": [16, 32, 64, 128],
            "encoder_blocks_per_layer": [2, 2, 2, 2],

            "decoder_net": "residual",
            "decoder_layer_channels": [128, 64, 32], # Missing channel is for output channel dim
            "decoder_blocks_per_layer": [2, 2, 2, 2],

            "flatten": False
        }

    return {
        **wae_base,
        **net_configs,
    }


GAE_CFG_MAP = {
    "base": get_base_config,
    "ae": get_ae_config,
    "avb": get_avb_config,
    "bigan": get_bigan_config,
    "gan": get_gan_config,
    "vae": get_vae_config,
    "wae": get_wae_config
}