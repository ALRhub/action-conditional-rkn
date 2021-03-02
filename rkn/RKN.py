from typing import Tuple

import numpy as np
import torch

from rkn.Decoder import SplitDiagGaussianDecoder
from rkn.Encoder import Encoder
from rkn.RKNLayer import RKNLayer
from util.ConfigDict import ConfigDict
from util.TimeDistributed import TimeDistributed

optim = torch.optim
nn = torch.nn


def elup1_inv(x: torch.Tensor) -> torch.Tensor:
    """
    inverse of elu+1, numpy only, for initialization
    :param x: input
    :return:
    """
    return np.log(x) if x < 1.0 else (x - 1.0)


def elup1(x: torch.Tensor) -> torch.Tensor:
    """
    elu + 1 activation faction to ensure positive covariances
    :param x: input
    :return: exp(x) if x < 0 else x + 1
    """
    return torch.exp(x).where(x < 0.0, x + 1.0)


class RKN:

    def __init__(self, target_dim: int, lod: int, cell_config: ConfigDict, use_cuda_if_available: bool = True):

        """
        :param target_dim:
        :param lod:
        :param cell_config:
        :param use_cuda_if_available:
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        self._lod = lod
        self._lsd = 2 * self._lod
        self.c = cell_config

        # parameters
        self._enc_out_normalization = self.c.enc_out_norm
        self._learning_rate = self.c.learning_rate
        # main model

        # Its not ugly, its pythonic :)
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(lod, output_normalization=self._enc_out_normalization)
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        self._rkn_layer = RKNLayer(latent_obs_dim=lod, cell_config=cell_config).to(self._device)

        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(lod, out_dim=target_dim), num_outputs=2).to(self._device)

        # build (default) initial state
        if self.c.learn_initial_state_covar:
            init_state_covar = elup1_inv(self.c.initial_state_covar)
            self._init_state_covar_ul = \
                nn.Parameter(nn.init.constant_(torch.empty(1, self._lsd), init_state_covar))
        else:
            self._init_state_covar_ul = self.c.initial_state_covar * torch.ones(1, self._lsd)

        self._initial_mean = torch.zeros(1, self._lsd).to(self._device)
        self._icu = torch.nn.Parameter(self._init_state_covar_ul[:, :self._lod].to(self._device))
        self._icl = torch.nn.Parameter(self._init_state_covar_ul[:, self._lod:].to(self._device))
        self._ics = torch.zeros(1, self._lod).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def _build_enc_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_dec_hidden_layers_mean(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for mean decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def _build_dec_hidden_layers_var(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for variance decoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, obs_batch: torch.Tensor, obs_valid_batch: torch.Tensor):
        """
        Note that as in RKN ICML 2020 Paper we decode the posterior here. However for comparsions in acRKN we
        decode the prior to ensure a fairer comparison.

        :param obs_batch: batch of observation sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :return: mean and variance
        """

        w, w_var = self._enc(obs_batch)
        post_mean, post_cov = self._rkn_layer(w, w_var, self._initial_mean, [self._icu, self._icl, self._ics],
                                              obs_valid_batch)
        out_mean, out_var = self._dec(post_mean, torch.cat(post_cov, dim=-1))

        return out_mean, out_var
