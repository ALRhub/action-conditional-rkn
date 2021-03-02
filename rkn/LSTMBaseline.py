import numpy as np
import torch

from rkn.Decoder import SplitDiagGaussianDecoder
from rkn.Encoder import Encoder
from rkn.RKN import RKN
from util.ConfigDict import ConfigDict
from util.TimeDistributed import TimeDistributed

optim = torch.optim
nn = torch.nn


class LSTMBaseline(RKN):

    def __init__(self, target_dim: int, lod: int, cell_config: ConfigDict, use_cuda_if_available: bool = True):
        """
        TODO: Gradient Clipping?
        :param target_dim:
        :param lod:
        :param cell_config:
        :param use_cuda_if_available:
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")

        self._lod = lod
        self._lsd = 2 * self._lod

        # parameters TODO: Make configurable
        self._enc_out_normalization = "pre"
        self._initial_state_variance = 10.0
        self._learning_rate = 1e-3

        # main model
        # Its not ugly, its pythonic :)
        Encoder._build_hidden_layers = self._build_enc_hidden_layers
        enc = Encoder(lod, output_normalization=self._enc_out_normalization)
        self._enc = TimeDistributed(enc, num_outputs=2).to(self._device)

        self._lstm_layer = nn.LSTM(input_size=2 * lod, hidden_size=5 * lod, batch_first=True).to(self._device)

        SplitDiagGaussianDecoder._build_hidden_layers_mean = self._build_dec_hidden_layers_mean
        SplitDiagGaussianDecoder._build_hidden_layers_var = self._build_dec_hidden_layers_var
        self._dec = TimeDistributed(SplitDiagGaussianDecoder(lod, out_dim=target_dim), num_outputs=2).to(self._device)

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def forward(self, obs_batch: torch.Tensor, obs_valid_batch: torch.Tensor, target_batch: torch.Tensor):
        """
        :param obs_batch: batch of observation sequences
        :param obs_valid_batch: batch of observation valid flag sequences
        :param target_batch: batch of sequences of ground truth for loss
        :return: out_mean, out_var
        """
        if not self._never_invalid:
            #here masked values are set to zero. You can also put an unrealistic value like a negative number.
            obs_masked_batch = obs_batch * obs_valid_batch

        w, w_var = self._enc(obs_masked_batch)
        out, _ = self._lstm_layer(torch.cat([w, w_var], dim=-1))
        out_mean, out_var = self._dec(out[..., : 2 * self._lod].contiguous(), out[..., 2 * self._lod:].contiguous())

        return out_mean, out_var
