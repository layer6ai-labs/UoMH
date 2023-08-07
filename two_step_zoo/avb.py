from itertools import chain
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from .generalized_autoencoder import GeneralizedAutoEncoder
from .density_estimator import DensityEstimator
from distributions import diagonal_gaussian_sample, diagonal_gaussian_log_prob
from .utils import batch_or_dataloader


class AdversarialVariationalBayes(GeneralizedAutoEncoder, DensityEstimator):
    model_type = "avb"

    def __init__(
        self,
        latent_dim,
        noise_dim,
        encoder,
        decoder,
        discriminator,
        input_sigma=1.0,
        prior_sigma=1.0,
        cnn=False,
        **kwargs
    ):
        super().__init__(
            latent_dim,
            encoder,
            decoder,
            **kwargs
        )
        self.noise_dim = noise_dim
        self.discriminator = discriminator
        self.input_sigma = input_sigma
        self.prior_sigma = prior_sigma
        self.cnn = cnn

    @batch_or_dataloader()
    def encode_transformed(self, x):
        # NOTE: Overridden to generate noise as input to encoder
        mu = torch.zeros((x.shape[0], self.noise_dim)).to(self.device)
        eps = diagonal_gaussian_sample(mu, self.input_sigma)
        return self.encoder(x, eps) if self.cnn else self.encoder(torch.cat((x, eps), 1))

    def train_batch(self, x, **kwargs):
        # Train discriminator on batch with encoder and decoder fixed
        self.optimizer[0].zero_grad()
        discriminator_loss = self._discr_error_batch(x).mean()
        discriminator_loss.backward()
        self.optimizer[0].step()
        self.lr_scheduler[0].step()

        # Train encoder and decoder on batch with discriminator fixed
        self.optimizer[1].zero_grad()
        nll_loss = self.loss(x).mean()
        nll_loss.backward()
        self.optimizer[1].step()
        self.lr_scheduler[1].step()

        return {
            "discriminator_loss": discriminator_loss,
            "negative_log_likelihood_loss": nll_loss
        }

    def _discr_error_batch(self, x):
        x = self._data_transform(x)
        z_q = self.encode_transformed(x)

        mu_p = torch.zeros((x.shape[0], self.latent_dim)).to(self.device)
        z_p = diagonal_gaussian_sample(mu_p, self.prior_sigma)

        # NOTE: Discriminator always is MLP so flatten inputs
        x_flat = x.flatten(start_dim=1)
        d_z_q = self.discriminator(torch.cat((x_flat, z_q), 1))
        d_z_p = self.discriminator(torch.cat((x_flat, z_p), 1))

        ones = torch.ones_like(d_z_q)
        zeros = torch.zeros_like(d_z_p)
        # NOTE: Train discriminator to be positive on encodings z_q
        d_z_loss = binary_cross_entropy_with_logits(d_z_p, zeros)
        d_z_q_loss = binary_cross_entropy_with_logits(d_z_q, ones)

        return d_z_loss + d_z_q_loss

    @batch_or_dataloader()
    def log_prob(self, x):
        # NOTE: Log prob calculated as ELBO using Discriminator net for log q(x|z)
        x = self._data_transform(x)
        z = self.encode_transformed(x)

        mu_x, log_sigma_x = self.decode_to_transformed(z)
        log_p_x_given_z = diagonal_gaussian_log_prob(
            x.flatten(start_dim=1),
            mu_x.flatten(start_dim=1),
            log_sigma_x.flatten(start_dim=1))

        # NOTE: Discriminator always is MLP so flatten inputs
        x_flat = x.flatten(start_dim=1)
        d_z = self.discriminator(torch.cat((x_flat, z), 1))

        return log_p_x_given_z - d_z

    def sample(self, n_samples, true_sample=True):
        # NOTE: Same as GaussianVAE
        z = torch.randn((n_samples, self.latent_dim)).to(self.device)
        mu, log_sigma = self.decode_to_transformed(z)
        sample = diagonal_gaussian_sample(mu, torch.exp(log_sigma)) if true_sample else mu
        return self._inverse_data_transform(sample)

    def set_optimizer(self, cfg):
        disc_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            self.discriminator.parameters(), lr=cfg["disc_lr"]
        )
        nll_optimizer = self._OPTIMIZER_MAP[cfg["optimizer"]](
            chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=cfg["nll_lr"]
        )
        self.optimizer = [disc_optimizer, nll_optimizer]
        self.num_optimizers = 2

        disc_lr_scheduler = self._get_lr_scheduler(
            optim=disc_optimizer,
            use_scheduler=cfg.get("use_disc_lr_scheduler", False),
            cfg=cfg
        )
        nll_lr_scheduler = self._get_lr_scheduler(
            optim=nll_optimizer,
            use_scheduler=cfg.get("use_nll_lr_scheduler", False),
            cfg=cfg
        )
        self.lr_scheduler = [disc_lr_scheduler, nll_lr_scheduler]