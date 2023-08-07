import torch
import torch.nn as nn
from torch.autograd.functional import jvp
import pdb

from ..two_step import TwoStepComponent
from ..utils import batch_or_dataloader


class GeneralizedAutoEncoder(TwoStepComponent):
    """
    GeneralizedAutoEncoder Parent Class
    """
    model_type = None

    def __init__(
            self,

            latent_dim,

            encoder,
            decoder,

            **kwargs
    ):
        super().__init__(**kwargs)

        self.latent_dim = latent_dim

        self.encoder = encoder
        self.decoder = decoder

    @batch_or_dataloader()
    def encode(self, x):
        # NOTE: Assume encode *only* wants a single output
        x = self._data_transform(x)
        return self._encode_transformed_without_tuple(x)

    @batch_or_dataloader()
    def encode_transformed(self, x, conditioning=None):
        return self.encoder(x, conditioning=conditioning)

    def _encode_transformed_without_tuple(self, x, conditioning=None):
        z = self.encode_transformed(x, conditioning=conditioning)
        return z[0] if type(z) == tuple else z

    @batch_or_dataloader()
    def decode(self, z, conditioning=None):
        # NOTE: Assume decode *only* wants a single output
        x = self._decode_to_transformed_without_tuple(z, conditioning=conditioning)
        return self._inverse_data_transform(x)

    def sample_conditioning(self, n_samples):
        return torch.multinomial(self.conditioning_counts, n_samples, replacement=True)

    @batch_or_dataloader()
    def decode_to_transformed(self, z, conditioning=None):

        if self.conditioning is not None and conditioning is None:
            conditioning = self.sample_conditioning(z.shape[0]).to(z.device)
     
        return self.decoder(z, conditioning=conditioning)

    def _decode_to_transformed_without_tuple(self, z, conditioning=None):
        x = self.decode_to_transformed(z, conditioning=conditioning)
        return x[0] if type(x) == tuple else x

    @batch_or_dataloader(pass_label=True)
    def rec_error(self, x, conditioning=None, return_z=False):
        # pdb.set_trace()
        if type(x) != torch.Tensor:
            label = x[1]
            x = x[0]
            
            if self.conditioning is not None:
                conditioning = label

        x = self._data_transform(x)

        z = self._encode_transformed_without_tuple(x, conditioning=conditioning)

        rec_x = self._decode_to_transformed_without_tuple(z, conditioning=conditioning)

        if return_z:
            return torch.sum(torch.square(x - rec_x).flatten(start_dim=1), dim=1, keepdim=True), z
        else:
            return torch.sum(torch.square(x - rec_x).flatten(start_dim=1), dim=1, keepdim=True)

    def _decoder_jacobian(self, z):
        '''Compute flattened Jacobian of decoder at input `z`'''
        jac = []
        for i in range(self.latent_dim):
            v = torch.zeros_like(z)
            v[:,i] = 1

            decode_fn = lambda z: self.decode(z)

            # TODO: forward-mode AD
            _, jac_vec_prod = jvp(decode_fn, z, v, create_graph=True)
            jac.append(jac_vec_prod.flatten(start_dim=1))

        return torch.stack(jac, dim=2)

    @batch_or_dataloader()
    def log_det_jtj(self, x):
        z = self.encode(x)
        jac = self._decoder_jacobian(z)
        jtj = torch.bmm(jac.transpose(1, 2), jac)

        cholesky_factor = torch.linalg.cholesky(jtj)
        cholesky_diagonal = torch.diagonal(cholesky_factor, dim1=1, dim2=2)
        log_det_jtj = 2 * torch.sum(torch.log(cholesky_diagonal), dim=1, keepdim=True)

        return log_det_jtj