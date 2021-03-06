import time as t
from typing import Tuple

import numpy as np
import torch
from rkn.acrkn.AcRKN import AcRKN
from torch.utils.data import TensorDataset, DataLoader

optim = torch.optim
nn = torch.nn


class Infer:

    def __init__(self, model: AcRKN, use_cuda_if_available: bool = True):

        """
        :param model: nn module for acrkn
        :param use_cuda_if_available:  if to use gpu
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._model = model
        self._learning_rate = self._model.c.learning_rate

        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def predict(self, obs: torch.Tensor, act: torch.Tensor, obs_valid: torch.Tensor, batch_size: int = -1) -> Tuple[
        float, float]:
        """
        Predict using the model
        :param obs: observations to evaluate on
        :param act: actions to evalauate on
        :param obs_valid: observation valid flag
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model = self._model.eval()
        out_mean_list = []
        out_var_list = []
        dataset = TensorDataset(obs, act, obs_valid)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        for batch_idx, (obs_batch, act_batch, obs_valid_batch) in enumerate(loader):
            with torch.no_grad():
                # Assign data tensors to devices
                torch_obs = obs_batch.to(self._device)
                torch_act = act_batch.to(self._device)
                torch_obs_valid = obs_valid_batch.to(self._device)

                # Forward Pass
                out_mean, out_var = self._model(torch_obs, torch_act, torch_obs_valid)

                out_mean_list.append(out_mean.cpu())
                out_var_list.append(out_var.cpu())

        return torch.cat(out_mean_list), torch.cat(out_var_list)
