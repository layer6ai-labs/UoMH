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
            "transforms": None,

            "scheduler_args": {"betas": [0.9,0.999]},
        }
    else:
        standalone_info = {}

    return {
        "flatten": True,
        "denoising_sigma": None,
        "dequantize": False,
        "scale_data": False,
        "whitening_transform": True,
        "logit_eps": 1e-3,

        "conditioning": None,
        "conditioning_dimension": 0,

        "optimizer": "adam",
        "lr": 0.001,
        "use_lr_scheduler": False,
        "max_epochs": 100,
        "max_grad_norm": 10,
        "make_valid_loader": True,
        "valid_fraction": 0.1,

        "early_stopping_metric": None,
        "max_bad_valid_epochs": 30,

        "lr_scheduler": "cosine",
        "lr_scheduler_step": 100, # in epochs
        "lr_scheduler_gamma": 0.1, # used for step scheduler

        # NOTE: A validation metric should indicate better performance as it decreases.
        #       Thus, log_likelihood is not an appropriate validation metric.
        "valid_metrics": ["loss"],
        "test_metrics": ["log_likelihood"],

        "device": "cuda" if torch.cuda.is_available() else "cpu",

        **standalone_info
    }


def get_arm_config(dataset, standalone):
    arm_base = {
        "k_mixture": 10,

        "flatten": False,
        "early_stopping_metric": "loss",
        "max_bad_valid_epochs": 10,
        "use_lr_scheduler": True
    }

    if standalone:
        hidden_size = 256
        num_layers = 2
    else:
        hidden_size = 128
        num_layers = 1

    net_config = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
    }

    return{
        **arm_base,
        **net_config
    }


def get_avb_config(dataset, standalone):
    return {
        "whitening_transform": False,

        "max_epochs": 50,

        "noise_dim": 128,
        "latent_dim": 20,
        "encoder_net": "mlp",
        "decoder_net": "mlp",
        "encoder_hidden_dims": [256],
        "decoder_hidden_dims": [256],
        "discriminator_hidden_dims": [256, 256],

        "single_sigma": True,

        "input_sigma": 3.,
        "prior_sigma": 1.,

        "lr": None,
        "disc_lr": 0.001,
        "nll_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_nll_lr_scheduler": True
    }


def get_ebm_config(dataset):
    if dataset == "mnist" or dataset == "fashion-mnist":
        net = "mlp"
    else:
        net = "cnn"
        
    ebm_base = {
        "max_length_buffer": 8192,
        "x_lims": (-1, 1),
        "ld_steps": 60,
        "ld_step_size": 10,
        "ld_eps_new": 0.05,
        "ld_sigma": 0.005,
        "ld_grad_clamp": 0.03,
        "loss_alpha": 0.1,

        "scale_data": True,
        "whitening_transform": False,
        "spectral_norm": False,
    }

    if net == "mlp":
        net_config = {
            "net": "mlp",
            "energy_func_hidden_dims": [256,256]
        }

    elif net == "cnn":
        net_config = {
            "net": "cnn",
            "energy_func_hidden_channels": [64, 64, 32, 32],
            "energy_func_kernel_size": [3, 3, 3, 3],
            "energy_func_stride": [1, 1, 1, 1],

            "flatten": False
        }

    return {
        **ebm_base,
        **net_config
    }


def get_flow_config(dataset, standalone):
    return {
        "transform": "simple_nsf",
        "hidden_units": 64,
        "num_layers": 4,
        "num_blocks_per_layer": 3,

        "flatten": True,
        "scale_data": True,
        "whitening_transform": False,
        "dequantize": True,
        "do_batchnorm": False,

        "base_distribution": None,
        "distribution_mean_spacing": 1,
        "num_mixture_components": 10,
    }

def get_gan_config(dataset, standalone):
    net = "cnn"

    gan_base = {
        "early_stopping_metric": "loss",
        "max_bad_valid_epochs": 50,

        "latent_dim": 100,

        "num_discriminator_steps": 5,
        "wasserstein": True,
        "clamp": 0.01,
        "gradient_penalty": True,
        "lambda": 10.0,

        "optimizer": 'adam',
        "lr": None,
        "disc_lr": 0.001,
        "ge_lr": 0.001,

        "use_lr_scheduler": None,
        "use_disc_lr_scheduler": True,
        "use_ge_lr_scheduler": True,

        "spectral_norm": False,

        "base_distribution": "gaussian",
        "distribution_mean_spacing": 1,
        "num_mixture_components": 10,

        "valid_metrics": ["loss"],
        "test_metrics": ["loss", "fid"],

        "decoder_norm": "instance",
        "discriminator_norm": "instance",

        "scheduler_args": {"betas": [0.5,0.99]},

        "decoder_final_activation": "tanh"
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

        disc_configs = {
            "discriminator_hidden_dims": [256, 256],
            "disc_net": "mlp"
        }

    elif net == "cnn":
        net_configs = {
            "encoder_net": "cnn",
            "encoder_hidden_channels": [256, 128, 64],#, 32, 16, 16],
            "encoder_kernel_size": [4,4,4],
            "encoder_stride": [2,2,2],

            "decoder_net": "cnn",
            "decoder_hidden_channels": [256, 128, 64],
            "decoder_kernel_size": [4,4,4],
            "decoder_stride": [2,2,2],

            "flatten": False,
            "max_epochs": 200,
        }

        disc_configs = {
            "disc_hidden_channels": [64,128,256,10],
            "disc_kernel_size": [4,4,4,4],
            "disc_stride": [2,2,2,2],
            "disc_net": "cnn"
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

        disc_configs = {
            "disc_net": "residual",
            "disc_layer_channels": [16, 32, 64],
            "disc_blocks_per_layer": [2, 2, 2],
        }
        

    return {
        **gan_base,
        **net_configs,
        **disc_configs
    }


def get_vae_config(dataset, standalone):
    if dataset in ["mnist", "fashion-mnist"]:
        net = "mlp"
    elif "imagenet" in dataset:
        net = "cnn"
    else:
        net = "cnn"
        
    vae_base = {
        "whitening_transform": False,
        "latent_dim": 20,
        "use_lr_scheduler": True,

        "single_sigma": True,

        "decoder_variance_lower_bound": 0,

        "base_distribution": "gaussian",
        "num_prior_components": 10,
        "distribution_mean_spacing": 1,
        
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

    return {
        **vae_base,
        **net_configs,
    }


DE_CFG_MAP = {
    "base": get_base_config,
    "arm": get_arm_config,
    "avb": get_avb_config,
    "ebm": get_ebm_config,
    "flow": get_flow_config,
    "vae": get_vae_config,
    "gan": get_gan_config
}