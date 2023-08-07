from itertools import chain
import pdb
import torch
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn as nn
import torch.nn.functional as F

from . import GeneralizedAutoEncoder
from distributions import diagonal_gaussian_sample

import torch.distributions as D
from ..utils import get_distribution

class WassersteinAutoEncoder(GeneralizedAutoEncoder):
    model_type = "wae"

    def __init__(
        self,
        latent_dim,
        encoder,
        decoder,
        discriminator,
        _lambda=10.0,
        sigma=1.0,
        base_distribution=None,
        num_mixture_components=10,
        distribution_mean_spacing=1,
        device='cuda',
        **kwargs
    ):
        super().__init__(
            latent_dim,
            encoder,
            decoder,
            **kwargs
        )
        self.discriminator = discriminator
        self._lambda = _lambda
        self.sigma = sigma
        self.base_distribution = base_distribution
        
        if base_distribution is not None:
            if "mixture_of_gaussians" in base_distribution:
                self.num_mixture_components = num_mixture_components
                
                self.mixture_weights,self.means,self.stds = get_distribution(base_distribution, self.num_mixture_components, distribution_mean_spacing, latent_dim, device)

                self.reset_prior()
            else:
                raise NotImplementedError(f"Base distribution {base_distribution} not implemented")

    def reset_prior(self):
        # Initialized for each forward pass because you can't backprop through distribution params
        if self.base_distribution is not None and "mixture_of_gaussians" in self.base_distribution:
            mix = D.Categorical(F.softmax(self.mixture_weights, dim=0))
            comp = D.Independent(D.Normal(self.means, self.stds),1)
            self.prior = D.MixtureSameFamily(mix,comp)
        
    def prior_sample(self, n_samples):
        self.reset_prior()

        if self.base_distribution is None:
            return diagonal_gaussian_sample(torch.zeros(n_samples, self.latent_dim), self.sigma).to(self.device)

        elif "mixture_of_gaussians" in self.base_distribution:
            z = torch.stack([self.prior.sample() for _ in range(n_samples)]).to(self.device)
            return z

    def train_batch(self, x, conditioning=None, **kwargs):
        self.reset_prior()
        # Train discriminator on batch with encoder and decoder fixed
        self.optimizer[0].zero_grad()
        discriminator_loss = self._discr_error_batch(x, conditioning=conditioning).mean()
        discriminator_loss.backward()
        self.optimizer[0].step()
        self.lr_scheduler[0].step()

        self.reset_prior()

        # Train encoder and decoder on batch with discriminator fixed
        self.optimizer[1].zero_grad()
        rec_loss = self._rec_error_batch(x, conditioning=conditioning).mean()
        rec_loss.backward()
        self.optimizer[1].step()
        self.lr_scheduler[1].step()

        return {
            "discriminator_loss": discriminator_loss,
            "reconstruction_loss": rec_loss
        }

    def _discr_error_batch(self, x, conditioning=None):
        
        x = self._data_transform(x)
        z_q = self.encode_transformed(x, conditioning=conditioning)

        mu = torch.zeros_like(z_q)
        z_p = self.prior_sample(z_q.shape[0]) # diagonal_gaussian_sample(mu, self.sigma)

        d_z_p = self.discriminator(z_p, conditioning=conditioning)
        d_z_q = self.discriminator(z_q, conditioning=conditioning)

        ones = torch.ones_like(d_z_q)
        zeros = torch.zeros_like(d_z_p)

        # NOTE: Train discriminator to be positive on encodings z_q
        d_z_p_loss = binary_cross_entropy_with_logits(d_z_p, zeros)
        d_z_q_loss = binary_cross_entropy_with_logits(d_z_q, ones)

        return self._lambda * (d_z_p_loss + d_z_q_loss)

    def _rec_error_batch(self, x, conditioning=None):
        # Reconstruction loss
        rec_loss, z_q = self.rec_error(x, conditioning=conditioning, return_z=True)

        # Discriminator loss
        d_z_q = self.discriminator(z_q)
        zeros = torch.zeros_like(d_z_q)
        d_loss = binary_cross_entropy_with_logits(d_z_q, zeros)

        return rec_loss + self._lambda * d_loss

    def sample(self, n_samples):
        mu = torch.zeros(n_samples, self.latent_dim).to(self.device)
        z_p = self.prior_sample(n_samples) # diagonal_gaussian_sample(mu, self.sigma)
        x = self.decode(z_p)
        return x

    def set_optimizer(self, cfg):
        disc_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            self.discriminator.parameters(), lr=cfg["disc_lr"]
        )
        rec_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=cfg["rec_lr"]
        )
        self.optimizer = [disc_optimizer, rec_optimizer]
        self.num_optimizers = 2

        disc_lr_scheduler = self._get_lr_scheduler(
            optim=disc_optimizer,
            use_scheduler=cfg.get("use_disc_lr_scheduler", False),
            cfg=cfg
        )
        rec_lr_scheduler = self._get_lr_scheduler(
            optim=rec_optimizer,
            use_scheduler=cfg.get("use_rec_lr_scheduler", False),
            cfg=cfg
        )
        self.lr_scheduler = [disc_lr_scheduler, rec_lr_scheduler]