import numpy as np
import torch
import torch.distributions as D


# NOTE: log_prob and entropy sum along the second dimension since the distribution
#       has diagonal covariance.
def diagonal_gaussian_log_prob(x, mu, log_sigma, sigma_eps=0):
    unreduced_log_prob = -np.log(2*np.pi)/2 - log_sigma - ((x - mu)/(torch.exp(log_sigma) + sigma_eps))**2/2
    return unreduced_log_prob.sum(dim=1, keepdim=True)


def diagonal_gaussian_entropy(log_sigma):
    unreduced_entropy = np.log(2*np.pi)/2 + 1./2 + log_sigma
    return unreduced_entropy.sum(dim=1, keepdim=True)


def diagonal_gaussian_sample(mu, sigma, shape=None):
    if shape is None:
        eps = torch.randn_like(mu)
    else:
        eps = torch.randn(shape).to(mu.device)
    return mu + sigma*eps


def uniform_sample(template=None, shape=None, device=None, range=[0,1]):
    if template is not None:
        x = torch.rand_like(template)
    elif shape is not None and device is not None:
        x = torch.rand(shape).to(device)
    else:
        raise ValueError("Must provide template or shape and device to uniform_sample")
    x = x * range[1] + (1-x) * range[0]
    
    return x


def get_gaussian_mixture(weights, mus, sigmas):
    mix = D.Categorical(logits=weights)
    comp = D.Independent(D.Normal(mus, sigmas), 1)
    return D.MixtureSameFamily(mix, comp)