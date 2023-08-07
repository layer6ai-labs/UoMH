import pdb
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits

from nflows.distributions import Distribution, StandardNormal
from itertools import chain

from ..utils import batch_or_dataloader, get_distribution
from ..two_step import TwoStepComponent

class LearnedDistribution(nn.Module):
    def __init__(self, base_distribution, latent_dim, num_mixture_components, distribution_mean_spacing, device):
        super().__init__()

        self.num_mixture_components = num_mixture_components
        self.base_distribution = base_distribution
        self.latent_dim = latent_dim
        self.distribution_mean_spacing = distribution_mean_spacing

        self.mixture_weights,self.means,self.stds = get_distribution(self.base_distribution, self.num_mixture_components, self.distribution_mean_spacing, self.latent_dim, device)

class GAN(TwoStepComponent):
    model_type = "gan"

    def __init__(
        self,
        latent_dim,

        decoder,
        discriminator,

        base_distribution="gaussian",
        num_mixture_components=1,
        distribution_mean_spacing=1,

        wasserstein=True,
        clamp=0.01,
        gradient_penalty=True,
        _lambda=10.0,
        num_discriminator_steps=2,

        device="cuda",
        
        **kwargs
    ):
        super().__init__(**kwargs)

        self.decoder = decoder
        self.discriminator = discriminator
        self.wasserstein = wasserstein
        self.clamp = clamp
        self.gradient_penalty = gradient_penalty
        self._lambda = _lambda
        self.num_discriminator_steps = num_discriminator_steps

        self.base_distribution = base_distribution
        self.latent_dim = latent_dim
        self.distribution_mean_spacing = distribution_mean_spacing
        self.num_mixture_components = num_mixture_components
        
        self.step_count = 0
        self.last_ge_loss = torch.tensor(0.0)

        if self.base_distribution == "gaussian":
            self.prior = StandardNormal([self.latent_dim])
        
        elif "mixture_of_gaussians" in self.base_distribution:
            self.learned_distribution = LearnedDistribution(self.base_distribution, self.latent_dim, self.num_mixture_components, self.distribution_mean_spacing, device)

    def reset_prior(self):
        # Initialized for each forward pass because you can't backprop through distribution params
        if "mixture_of_gaussians" in self.base_distribution:
            mix = D.Categorical(F.softmax(self.learned_distribution.mixture_weights, dim=0))
            comp = D.Independent(D.Normal(self.learned_distribution.means, self.learned_distribution.stds),1)
            self.prior = D.MixtureSameFamily(mix,comp)

    def train_batch(self, x, conditioning=None, **kwargs):

        self.optimizer[0].zero_grad()
        discriminator_loss = self._discr_error_batch(x, conditioning=conditioning).mean()
        discriminator_loss.backward()
        self.optimizer[0].step()
        if self.wasserstein and not self.gradient_penalty:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clamp, self.clamp)

        self.last_ge_loss = self.last_ge_loss.to(self.device)
        self.step_count += 1
        # NOTE: Take several steps for discriminator for each generator/encoder step
        if self.step_count >= self.num_discriminator_steps:
            self.step_count = 0

            self.optimizer[1].zero_grad()
            generator_encoder_loss = self._ge_error_batch(x, conditioning=conditioning).mean()
            generator_encoder_loss.backward()
            self.last_ge_loss = generator_encoder_loss
            self.optimizer[1].step()
            self.lr_scheduler[0].step() # update schedulers together to prevent ge having larger lr after many epochs
            self.lr_scheduler[1].step()

            self.reset_prior()

        return {
            "discriminator_loss": discriminator_loss,
            "generator_encoder_loss": self.last_ge_loss,
        }
    
    @batch_or_dataloader()
    def decode_to_transformed(self, z):
        return self.decoder(z)

    @batch_or_dataloader()
    def loss(self, x):
        x_flat = x.flatten(start_dim=1)
        d_gen = self.discriminator(x_flat)
        return d_gen.flatten()
     
    @batch_or_dataloader()
    def disc_loss(self, x):
        d_gen, d_real, x_true, x_fake = self._discriminator_outputs(x)

        if self.wasserstein:
            discriminator_loss = -torch.mean(d_real) + torch.mean(d_gen)

        return discriminator_loss.flatten()
    
    def sample_prior(self, n_samples):
        self.reset_prior()

        if self.base_distribution == "gaussian":
            z = torch.randn((n_samples, self.latent_dim)).to(self.device)
        elif "mixture_of_gaussians" in self.base_distribution:
            z = torch.stack([self.prior.sample() for _ in range(n_samples)]).to(self.device)

        return z

    def _discriminator_outputs(self, x, conditioning=None):
        
        x = self._data_transform(x)

        # sample from latent prior and decode (generate)
        z_p = self.sample_prior(n_samples=x.shape[0])
        x_g = self.decode_to_transformed(z_p, conditioning=conditioning)

        # NOTE: Discriminator always is MLP so flatten inputs
        x_flat = x.flatten(start_dim=1)
        x_g_flat = x_g.flatten(start_dim=1)
     
        d_gen = self.discriminator(x_g_flat)
        d_real = self.discriminator(x_flat)

        return d_gen, d_real, x, x_g

    def _discr_error_batch(self, x, conditioning=None):
        
        d_gen, d_real, x_true, x_fake = self._discriminator_outputs(x, conditioning=conditioning)

        if self.wasserstein and self.gradient_penalty:
            discriminator_loss = -torch.mean(d_real) + torch.mean(d_gen) + self._grad_penalty(x_true, x_fake)

        elif self.wasserstein:
            discriminator_loss = -torch.mean(d_real) + torch.mean(d_gen)

        else:
            zeros = torch.zeros_like(d_gen)
            ones = torch.ones_like(d_real)

            # NOTE: Train discriminator to be positive on real data + encodings
            d_z_g_correct = binary_cross_entropy_with_logits(d_gen, zeros)
            d_z_e_correct = binary_cross_entropy_with_logits(d_real, ones)
            discriminator_loss = d_z_g_correct + d_z_e_correct

        return discriminator_loss
    
    def _ge_error_batch(self, x, conditioning=None, idx=None):
        # Discriminator loss
        d_gen, d_real = self._discriminator_outputs(x, conditioning=conditioning)[0:2]
        
        if self.wasserstein:
            generator_encoder_loss = -torch.mean(d_gen) + torch.mean(d_real)

        else:
            zeros = torch.zeros_like(d_real)
            ones = torch.ones_like(d_gen)

            d_z_g_incorrect = binary_cross_entropy_with_logits(d_gen, ones)
            d_z_e_incorrect = binary_cross_entropy_with_logits(d_real, zeros)
            generator_encoder_loss = d_z_g_incorrect + d_z_e_incorrect

        return generator_encoder_loss
    
    def _grad_penalty(self, x_true, x_fake):
        # NOTE: sample uniformly for interpolation parameters
        eta = torch.rand(x_true.size(0)).to(self.device)
        for i in range(x_true.dim() - 1):
            eta = eta.unsqueeze(-1)

        interpolated_x = eta * x_true + ((1-eta) * x_fake)

        # NOTE: Discriminator always is MLP so flatten inputs
        interpolated_x_flat = interpolated_x.flatten(start_dim=1)
        d_x = self.discriminator(interpolated_x_flat)

        grads = torch.autograd.grad(d_x, interpolated_x_flat, grad_outputs=torch.ones_like(d_x), retain_graph=True, create_graph=True)[0]

        return ((grads.norm(2, dim=1) - 1)**2).mean() * self._lambda

    @batch_or_dataloader()
    def decode(self, z, conditioning=None):
        # NOTE: Assume decode *only* wants a single output
        x = self._decode_to_transformed_without_tuple(z, conditioning=conditioning)
        return self._inverse_data_transform(x)

    @batch_or_dataloader()
    def decode_to_transformed(self, z, conditioning=None):

        if self.conditioning is not None and conditioning is None:
            conditioning = self.sample_conditioning(z.shape[0]).to(z.device)

        return self.decoder(z, conditioning=conditioning)

    def _decode_to_transformed_without_tuple(self, z, conditioning=None):
        x = self.decode_to_transformed(z, conditioning=conditioning)
        return x[0] if type(x) == tuple else x

    def sample_conditioning(self, n_samples):
        return torch.multinomial(self.conditioning_counts, n_samples, replacement=True)
        
    def sample(self, n_samples, conditioning=None):
        z_p = self.sample_prior(n_samples=n_samples)
        x = self.decode(z_p, conditioning=conditioning)
        return x

    def set_optimizer(self, cfg):
        
        disc_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            self.discriminator.parameters(), lr=cfg["disc_lr"], **cfg.get("scheduler_args", {})
        )

        if "mixture" in self.base_distribution:
            ge_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
                chain(self.learned_distribution.parameters(), self.decoder.parameters()),
                lr=cfg["ge_lr"],
                **cfg.get("scheduler_args", {})
                
            )
        else:
            ge_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
                self.decoder.parameters(),
                lr=cfg["ge_lr"],
                **cfg.get("scheduler_args", {})
            )

        self.optimizer = [disc_optimizer, ge_optimizer]
        self.num_optimizers = 2

        disc_lr_scheduler = self._get_lr_scheduler(
            optim=disc_optimizer,
            use_scheduler=cfg.get("use_disc_lr_scheduler", False),
            cfg=cfg
        )
        ge_lr_scheduler = self._get_lr_scheduler(
            optim=ge_optimizer,
            use_scheduler=cfg.get("use_ge_lr_scheduler", False),
            cfg=cfg
        )
        self.lr_scheduler = [disc_lr_scheduler, ge_lr_scheduler]