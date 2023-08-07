'''Sample transforms for designing normalizing flows'''

import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn

from nflows.nn import nets as nets
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import (
    AdditiveCouplingTransform,
    AffineCouplingTransform,
    PiecewiseRationalQuadraticCouplingTransform
)
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation, Permutation
from nflows.transforms.conv import OneByOneConvolution
from nflows.transforms.normalization import BatchNorm


class SimpleFlowTransform(CompositeTransform):
    '''Simple flow transform designed to act on flat data'''

    def __init__(
        self,
        features,
        hidden_features,
        num_layers,
        num_blocks_per_layer,
        include_linear=True,
        num_bins=8,
        tail_bound=1.0,
        activation=F.relu,
        dropout_probability=0.0,
        batch_norm_within_layers=False,
        coupling_constructor=PiecewiseRationalQuadraticCouplingTransform,
        net="mlp",
        data_shape=None,
        do_batchnorm=False,
        conditioning=None,
        conditioning_dimension=None
    ):
        mask = torch.ones(features)
        mask[::2] = -1

        self.model_type = net

        if conditioning_dimension is None:
            conditioning_dimension = 0

        assert not (do_batchnorm and net == "cnn"), "Batchnorm only implemented for 1D inputs"

        if net == "cnn":
            mask = mask[:data_shape[0]]
       
        def create_resnet(in_features, out_features):
            if net == "cnn":
                assert data_shape is not None

                return nets.ConvResidualNet(
                    in_features,
                    out_features,
                    hidden_channels=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                    context_features=conditioning_dimension
                )
            else:
                return nets.ResidualNet(
                    in_features,
                    out_features,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                    context_features=conditioning_dimension
                )

        layers = []
        for _ in range(num_layers):
            coupling_transform = coupling_constructor(
                mask=mask,
                transform_net_create_fn=create_resnet,
                tails="linear",
                num_bins=num_bins,
                tail_bound=tail_bound,
            )
            layers.append(coupling_transform)
            mask *= -1

            if include_linear:

                if self.model_type == "cnn":
                    linear_transform = CompositeTransform([
                        OneByOneConvolution(data_shape[0], identity_init=True)
                    ])
                else:
                    if do_batchnorm:
                        layers.append(BatchNorm(features))
                    linear_transform = CompositeTransform([
                        RandomPermutation(features=features),
                        LULinear(features, identity_init=True)])
                layers.append(linear_transform)

        super().__init__(layers)


class SimpleNSFTransform(SimpleFlowTransform):

    def __init__(self, **kwargs):
        super().__init__(
            coupling_constructor=PiecewiseRationalQuadraticCouplingTransform,
            **kwargs,
        )