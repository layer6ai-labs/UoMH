import torch

from distributions import get_gaussian_mixture
from . import DensityEstimator
from ..utils import batch_or_dataloader


class AutoRegressiveModel(DensityEstimator):

    model_type = 'arm'

    def __init__(self, ar_network, **kwargs):
        super().__init__(**kwargs)
        self.ar_network = ar_network


class GaussianMixtureLSTMModel(AutoRegressiveModel):

    def __init__(self, ar_network, image_height, input_length, **kwargs):
        super().__init__(ar_network, **kwargs)
        self.image_height = image_height
        if image_height is None:
            assert len(self.data_shape) == 1
        self.input_length = input_length

    @batch_or_dataloader()
    def log_prob(self, x):
        x = self._data_transform(x)
        if self.image_height is None:
            x = torch.unsqueeze(x, 1)
        weights, mus, sigmas = self.ar_network.forward(x)
        gmm = get_gaussian_mixture(weights, mus, sigmas)
        out = gmm.log_prob(torch.permute(x.flatten(start_dim=2), (0, 2, 1)))
        return out.sum(dim=1, keepdim=True)

    def sample(self, n_samples):
        mix_params = self.ar_network.linear(torch.zeros((n_samples, 1, self.ar_network.hidden_size)).to(self.device))
        weights, mus, sigmas = self.ar_network.split_transform_and_reshape(mix_params)
        new_coordinate = get_gaussian_mixture(weights, mus, sigmas).sample()
        samples = new_coordinate
        h_c = None
        for _ in range(self.input_length - 1):
            weights, mus, sigmas, h_c = self.ar_network.forward(x=new_coordinate,
                                                                return_h_c=True,
                                                                h_c=h_c,
                                                                not_sampling=False)
            new_coordinate = get_gaussian_mixture(weights, mus, sigmas).sample()
            samples = torch.cat((samples, new_coordinate), dim=1)
        if self.image_height is None:
            samples = torch.squeeze(samples, 2)
        else:
            samples = torch.permute(samples, (0, 2, 1))
            samples = torch.reshape(samples, (samples.shape[0], samples.shape[1], self.image_height, -1))
        return self._inverse_data_transform(samples)