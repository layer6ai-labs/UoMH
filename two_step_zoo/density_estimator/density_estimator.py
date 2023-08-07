import pdb

from ..two_step import TwoStepComponent


class DensityEstimator(TwoStepComponent):

    def sample(self, n_samples):
        raise NotImplementedError("sample not implemented")

    def log_prob(self, x, **kwargs):
        raise NotImplementedError("log_prob not implemented")

    def loss(self, x, **kwargs):
        return -self.log_prob(x, **kwargs).mean()