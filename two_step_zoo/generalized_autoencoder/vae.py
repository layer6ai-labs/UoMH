import pdb
import numpy as np
import torch

from ..density_estimator import DensityEstimator
from . import GeneralizedAutoEncoder
from ..utils import batch_or_dataloader, get_distribution
from distributions import diagonal_gaussian_log_prob, diagonal_gaussian_entropy, diagonal_gaussian_sample

import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

import math

class GaussianVAE(GeneralizedAutoEncoder, DensityEstimator):
    def __init__(
            self,
            base_distribution,
            distribution_mean_spacing=1,
            num_prior_components=0,
            decoder_variance_lower_bound=0,
            device="cuda",
            **kwargs
            ):
        super().__init__(**kwargs)
        self.model_type = "vae"
        self.decoder_variance_lower_bound = decoder_variance_lower_bound
        self.base_distribution = base_distribution
        
        if "mixture_of_gaussians" in self.base_distribution:
            self.num_prior_components = num_prior_components
            self.latent_dim = kwargs["latent_dim"]

            self.mixture_weights,self.means,self.stds = get_distribution(self.base_distribution, self.num_prior_components, distribution_mean_spacing, self.latent_dim, device)
            self.reset_prior()
    
    def reset_prior(self):
        # Initialized for each forward pass because you can't backprop through distribution params
        if "mixture_of_gaussians" in self.base_distribution:
            mix = D.Categorical(F.softmax(self.mixture_weights, dim=0))
            comp = D.Independent(D.Normal(self.means, self.stds),1)
            self.prior = D.MixtureSameFamily(mix,comp)

    def sample(self, n_samples, true_sample=True):
        true_sample = False
     
        self.reset_prior()

        if self.base_distribution == "gaussian":
            z = torch.randn((n_samples, self.latent_dim)).to(self.device)
        elif "mixture_of_gaussians" in self.base_distribution:
            z = torch.stack([self.prior.sample() for _ in range(n_samples)]).to(self.device)

        mu, log_sigma = self.decode_to_transformed(z)
        sample = diagonal_gaussian_sample(mu, torch.exp(log_sigma)+self.decoder_variance_lower_bound) if true_sample else mu
        return self._inverse_data_transform(sample)
    
    def get_log_p_z(self, z):
        if self.base_distribution == "gaussian":
            return diagonal_gaussian_log_prob(z, torch.zeros_like(z), torch.zeros_like(z))
        elif "mixture_of_gaussians" in self.base_distribution:
            return self.prior.log_prob(z).unsqueeze(-1)

    @batch_or_dataloader()
    def decode_to_transformed(self, z, conditioning=None):

        if self.conditioning is not None and conditioning is None:
            conditioning = self.sample_conditioning(z.shape[0]).to(z.device)
     
        return self.decoder(z, conditioning=conditioning)

    @batch_or_dataloader(pass_idx=True, pass_label=True)
    def log_prob(self, x, conditioning=None, k=1, idx=None, encoder_params=None):
        self.reset_prior()

        if type(x) == tuple:
            label = x[1]
            x = x[0]
            if conditioning is None: conditioning = label

        # NOTE: With k=1, this gives the ELBO.
        batch_size = x.shape[0]

        # NOTE: Perform data transform _before_ repeat_interleave because we do not want
        #       to dequantize the same x point in several different ways.
        x = self._data_transform(x)
        
        x = x.repeat_interleave(k, dim=0)
        
        mu_z, log_sigma_z = self.encode_transformed(x, conditioning=conditioning)
        z = diagonal_gaussian_sample(mu_z, torch.exp(log_sigma_z))
        mu_x, log_sigma_x = self.decode_to_transformed(z, conditioning=conditioning)
        log_sigma_x =  torch.full(log_sigma_x.shape, math.log(1)).to(log_sigma_x.device) # HACK

        log_p_z = self.get_log_p_z(z)
        
        log_p_x_given_z = diagonal_gaussian_log_prob(
            x.flatten(start_dim=1),
            mu_x.flatten(start_dim=1),
            log_sigma_x.flatten(start_dim=1),
            sigma_eps=self.decoder_variance_lower_bound
        )
        
        if k == 1:
            h_z_given_x = diagonal_gaussian_entropy(log_sigma_z)
            loss = log_p_z + log_p_x_given_z + h_z_given_x
            return loss
        else:
            log_q_z_given_x = diagonal_gaussian_log_prob(z, mu_z, log_sigma_z)
            elbo = log_p_z + log_p_x_given_z - log_q_z_given_x
            return torch.logsumexp(elbo.reshape(batch_size, k, 1), dim=1) - np.log(k)