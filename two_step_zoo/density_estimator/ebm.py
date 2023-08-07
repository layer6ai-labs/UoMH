import random

import torch
import numpy as np

from . import DensityEstimator
from ..utils import batch_or_dataloader


class EnergyBasedModel(DensityEstimator):

    model_type = 'ebm'

    def __init__(
            self,
            energy_func,
            x_shape,  # shape of transformed data
            max_length_buffer=8192,
            x_lims=(-1, 1),  # boundaries for transformed data
            ld_steps=60,
            ld_step_size=10,
            ld_eps_new=0.05,
            ld_sigma=0.005,
            ld_grad_clamp=0.03,
            loss_alpha=0.1,

            **kwargs
    ):
        super().__init__(**kwargs)
        self.energy_func = energy_func
        self.max_length_buffer = max_length_buffer
        self.x_lims = x_lims
        self.diff = x_lims[1] - x_lims[0]
        self.ld_steps = ld_steps
        self.ld_step_size = ld_step_size
        self.ld_eps_new = ld_eps_new
        self.ld_sigma = ld_sigma
        self.ld_grad_clamp = ld_grad_clamp
        self.loss_alpha = loss_alpha
        self.x_shape = x_shape
        self.buffer = torch.rand((max_length_buffer,) + x_shape) * self.diff + x_lims[0]
        self.register_buffer('sample_buffer', self.buffer)

    def _langevin_dynamics(self, x, steps, step_size, sigma, grad_clamp):
        is_training = self.energy_func.training
        self.energy_func.eval()
        for p in self.energy_func.parameters():
            p.requires_grad = False
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        for _ in range(steps):
            x.requires_grad = True  # torch.no_grad() disables this, so it has to be inside the loop, not outside
            out = self.energy_func(x)
            out.sum().backward()
            grad = x.grad.detach().clamp_(-grad_clamp, grad_clamp)
            with torch.no_grad():
                x = x + torch.randn_like(x) * sigma
                x = x - step_size * grad
                x.clamp_(self.x_lims[0], self.x_lims[1])
        for p in self.energy_func.parameters():
            p.requires_grad = True
        self.energy_func.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)
        return x

    def sample(self, n_samples, steps=60, step_size=10, eps_new=0.0, sigma=0.005, grad_clamp=0.03,
               for_loss=False, update_buffer=False):
        # Initialize langevin dynamics from random noise and buffer
        n_new = np.random.binomial(n_samples, eps_new)
        rand_x = torch.rand((n_new,) + self.x_shape) * self.diff + self.x_lims[0]
        if n_new < n_samples:
            old_x = torch.stack(random.choices(self.buffer, k=n_samples-n_new))
            x = torch.cat([rand_x, old_x], dim=0).detach().to(self.device)
        else:
            x = rand_x.detach().to(self.device)

        x = self._langevin_dynamics(x, steps, step_size, sigma, grad_clamp)

        if update_buffer:
            self.buffer = torch.cat((x.cpu(), self.buffer))
            self.buffer = self.buffer[:self.max_length_buffer]

        if for_loss:
            return x
        else:
            return self._inverse_data_transform(x)

    @batch_or_dataloader()
    def log_prob(self, x):
        # NOTE: this function returns the log_prob up to an additive constant
        x = self._data_transform(x)
        return -self.energy_func(x)

    @batch_or_dataloader(agg_func=lambda x: torch.mean(torch.Tensor(x)))
    def loss(self, x):
        batch_size = x.shape[0]
        x = self._data_transform(x)
        if self.x_shape is None:
            self.x_shape = x.shape[1:]

        pos = self.energy_func(x)
        neg = self.energy_func(self.sample(n_samples=batch_size,
                                           steps=self.ld_steps,
                                           step_size=self.ld_step_size,
                                           eps_new=self.ld_eps_new,
                                           sigma=self.ld_sigma,
                                           grad_clamp=self.ld_grad_clamp,
                                           for_loss=True,
                                           update_buffer=True)
                               )
        cd_loss = (pos - neg).mean()
        reg_loss = (torch.square(pos) + torch.square(neg)).mean()
        return cd_loss + self.loss_alpha * reg_loss