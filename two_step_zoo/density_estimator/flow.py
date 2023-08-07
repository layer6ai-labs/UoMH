from nflows.distributions import Distribution, StandardNormal
import torch.distributions as D
import torch
import torch.nn as nn
import torch.nn.functional as F
from nflows.flows.base import Flow

from . import DensityEstimator
from ..utils import batch_or_dataloader, get_distribution
import pdb

class ContextStandardNormal(StandardNormal):
    def _sample(self, num_samples, context):
        return torch.randn(num_samples, *self._shape, device=self._log_z.device)

class AdaptedMixtureSameFamily():
    def __init__(self, distribution):
        self.distribution = distribution

    def log_prob(self, inputs, context=None):
        return self.distribution.log_prob(inputs)
    
    def sample(self, num_samples, context=None): # Context to play nice with nflows
        return torch.stack([self.distribution.sample() for _ in range(num_samples)])

class EmbeddingNet(nn.Module):
    def __init__(self, conditioning_dimension):
        super().__init__()
        self.conditioning_dimension = conditioning_dimension

    def forward(self, idxs):
        conditioning_vector = torch.zeros(idxs.shape[0], self.conditioning_dimension).to(idxs.device)
        conditioning_vector[torch.arange(idxs.shape[0]), idxs] = 1
        return conditioning_vector 

class NormalizingFlow(DensityEstimator):

    model_type = "nf"

    def __init__(self, 
        dim, 
        transform, 
        base_distribution_type=None, 
        num_mixture_components=0, 
        distribution_mean_spacing=1, 
        conditioning=None,
        conditioning_dimension=0,
        device="cuda",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.transform = transform
        self.dim = dim
        self.base_distribution_type = base_distribution_type

        self.conditioning = conditioning
        self.embedding_net = None
        if conditioning is not None:
            self.embedding_net = EmbeddingNet(conditioning_dimension)

        if self.base_distribution_type is None:
            self.base_distribution = StandardNormal([dim])
        
        elif "mixture_of_gaussians" in self.base_distribution_type:
            self.num_mixture_components = num_mixture_components

            self.mixture_weights,self.means,self.stds = get_distribution(self.base_distribution_type, self.num_mixture_components, distribution_mean_spacing, self.dim, device)
            self.reset_prior()
            
        else:
            raise NotImplementedError(f"Base distribution {self.base_distribution_type} not implemented")

        self._nflow = Flow(
            transform=self.transform,
            distribution=self.base_distribution,
            embedding_net=self.embedding_net
        )

        self._nflow._context_used_in_base = True

    def reset_prior(self):
        # Initialized for each forward pass because you can't backprop through distribution params
        if self.base_distribution_type is not None and "mixture_of_gaussians" in self.base_distribution_type:
            mix = D.Categorical(F.softmax(self.mixture_weights, dim=0))
            comp = D.Independent(D.Normal(self.means, self.stds),1)
            self.base_distribution = AdaptedMixtureSameFamily(D.MixtureSameFamily(mix,comp))

            self._nflow = Flow(
                transform=self.transform,
                distribution=self.base_distribution
            )

    def sample_conditioning(self, n_samples):
        return torch.multinomial(self.conditioning_counts, n_samples, replacement=True)

    def sample(self, n_samples):
        self.reset_prior()

        if self.conditioning is not None:
            conditioning = self.sample_conditioning(1).to(self.device)
       
            samples = self._nflow.sample(n_samples, context=conditioning)
        else:
            samples = self._nflow.sample(n_samples)

        return self._inverse_data_transform(samples)

    @batch_or_dataloader(pass_label=True)
    def log_prob(self, x, conditioning=None):
        self.reset_prior()

        if type(x) == tuple:
            label = x[1]
            x = x[0]
            if conditioning is None and self.conditioning is not None: conditioning = label

        x = self._data_transform(x)

        log_prob = self._nflow.log_prob(x, context=conditioning)
        
        if len(log_prob.shape) == 1:
            log_prob = log_prob.unsqueeze(1)

        return log_prob