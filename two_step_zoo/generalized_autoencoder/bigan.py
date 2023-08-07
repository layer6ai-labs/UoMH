from itertools import chain
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from . import GeneralizedAutoEncoder
from distributions import uniform_sample


class BiGAN(GeneralizedAutoEncoder):
    model_type = "bigan"

    def __init__(
        self,
        latent_dim,
        encoder,
        decoder,
        discriminator,

        wasserstein=True,
        clamp=0.01,
        gradient_penalty=True,
        _lambda=10.0,
        num_discriminator_steps=2,
        uniform_sample_range=[-1,1],
        recon_weight=1.0,
        
        **kwargs
    ):
        super().__init__(
            latent_dim,
            encoder,
            decoder,
            **kwargs
        )
        self.discriminator = discriminator
        self.wasserstein = wasserstein
        self.clamp = clamp
        self.gradient_penalty = gradient_penalty
        self._lambda = _lambda
        self.num_discriminator_steps = num_discriminator_steps
        self.uniform_sample_range = uniform_sample_range
        self.recon_weight = recon_weight
        
        self.step_count = 0
        self.last_ge_loss = torch.tensor(0.0)

    def train_batch(self, x, **kwargs):
        self.optimizer[0].zero_grad()
        discriminator_loss = self._discr_error_batch(x).mean()
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
            generator_encoder_loss = self._ge_error_batch(x).mean()
            generator_encoder_loss.backward()
            self.last_ge_loss = generator_encoder_loss
            self.optimizer[1].step()
            self.lr_scheduler[0].step() # update schedulers together to prevent ge having larger lr after many epochs
            self.lr_scheduler[1].step()

        return {
            "discriminator_loss": discriminator_loss,
            "generator_encoder_loss": self.last_ge_loss,
        }

    def _discriminator_outputs(self, x):
        x = self._data_transform(x)
        z_e = self.encode_transformed(x)

        # sample from latent prior and decode (generate)
        z_p = uniform_sample(template=z_e, range=self.uniform_sample_range)
        x_g = self.decode_to_transformed(z_p)

        # NOTE: Discriminator always is MLP so flatten inputs
        x_flat = x.flatten(start_dim=1)
        x_g_flat = x_g.flatten(start_dim=1)
        d_z_g = self.discriminator(torch.cat((x_g_flat, z_p), 1))
        d_z_e = self.discriminator(torch.cat((x_flat, z_e), 1))

        return d_z_g, d_z_e, x, x_g, z_e, z_p

    def _discr_error_batch(self, x):
        d_z_g, d_z_e, x_true, x_fake, z_true, z_fake = self._discriminator_outputs(x)

        if self.wasserstein and self.gradient_penalty:
            discriminator_loss = -torch.mean(d_z_e) + torch.mean(d_z_g) + self._grad_penalty(x_true, x_fake, z_true, z_fake)

        elif self.wasserstein:
            discriminator_loss = -torch.mean(d_z_e) + torch.mean(d_z_g)

        else:
            zeros = torch.zeros_like(d_z_g)
            ones = torch.ones_like(d_z_e)

            # NOTE: Train discriminator to be positive on real data + encodings
            d_z_g_correct = binary_cross_entropy_with_logits(d_z_g, zeros)
            d_z_e_correct = binary_cross_entropy_with_logits(d_z_e, ones)
            discriminator_loss = d_z_g_correct + d_z_e_correct

        return discriminator_loss
    
    def _ge_error_batch(self, x):
        # Reconstruction loss
        rec_loss, z_q = self.rec_error(x, return_z=True)
        
        # Discriminator loss
        d_z_g, d_z_e = self._discriminator_outputs(x)[0:2]
        
        if self.wasserstein:
            generator_encoder_loss = -torch.mean(d_z_g) + torch.mean(d_z_e)

        else:
            zeros = torch.zeros_like(d_z_e)
            ones = torch.ones_like(d_z_g)

            d_z_g_incorrect = binary_cross_entropy_with_logits(d_z_g, ones)
            d_z_e_incorrect = binary_cross_entropy_with_logits(d_z_e, zeros)
            generator_encoder_loss = d_z_g_incorrect + d_z_e_incorrect

        return generator_encoder_loss + self.recon_weight * rec_loss
    
    def _grad_penalty(self, x_true, x_fake, z_true, z_fake):
        # NOTE: sample uniformly for interpolation parameters
        eta = torch.rand(x_true.size(0)).to(self.device)
        for i in range(x_true.dim() - 1):
            eta = eta.unsqueeze(-1)

        interpolated_x = eta * x_true + ((1-eta) * x_fake)

        # NOTE: This is a W-BiGAN, so we also interpolate the latent vector
        eta = eta.view(-1, 1)
        interpolated_z = eta * z_true + (1-eta) * z_fake

        # NOTE: Discriminator always is MLP so flatten inputs
        interpolated_x_flat = interpolated_x.flatten(start_dim=1)
        d_input = torch.cat((interpolated_x_flat, interpolated_z), 1)
        d_x = self.discriminator(d_input)

        grads = torch.autograd.grad(d_x, d_input, grad_outputs=torch.ones_like(d_x), retain_graph=True, create_graph=True)[0]

        return ((grads.norm(2, dim=1) - 1)**2).mean() * self._lambda

    def sample(self, n_samples):
        z_p = uniform_sample(shape=(n_samples, self.latent_dim), device=self.device, range=self.uniform_sample_range)
        x = self.decode(z_p)
        return x

    def set_optimizer(self, cfg):
        disc_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            self.discriminator.parameters(), lr=cfg["disc_lr"]
        )
        ge_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=cfg["ge_lr"]
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