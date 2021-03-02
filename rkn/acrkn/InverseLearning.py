import os
import time as t
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from rkn.acrkn.AcRknInv import AcRknInv
from util.Losses import mse

writer = SummaryWriter('/home/vshaj/CLAS/Logs/ALRhub/INV_jointspace-lam.1')

optim = torch.optim
nn = torch.nn


class Learn:

    def __init__(self, model: AcRknInv, lam: float, feature_space: bool, save_path=None,
                 use_cuda_if_available: bool = True):

        """
        :param model: nn module for acrkn
        :param lam: the regularization weight
        :param feature_space: if to take losses for forward model in the latent space or not
        :param use_cuda_if_available: if to use gpu
        """

        self._device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._model = model
        self._lam = lam
        self._feature_space = feature_space
        if save_path is None:
            self._save_path = os.getcwd() + '/experiments/Franka/saved_models/model.torch'
        else:
            self._save_path = save_path
        self._learning_rate = self._model.c.learning_rate\

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def train_step(self, train_obs: np.ndarray, train_act: np.ndarray, train_obs_valid: np.ndarray,
                   train_targets: np.ndarray, train_act_targets: np.ndarray, batch_size: int) \
            -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions to be fed to the ac_predict stage
        :param train_obs_valid: training valid flag
        :param train_targets: training targets
        :param train_act_targets: training targets for actions
        :param batch_size: batch_size
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        self._model.train()
        dataset = TensorDataset(train_obs, train_act, train_obs_valid, train_targets, train_act_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        fwd_mse = inv_mse = 0
        t0 = t.time()
        b = list(loader)[0]

        for batch_idx, (obs, act, obs_valid, targets, act_targets) in enumerate(loader):
            obs_batch = (obs).to(self._device)
            act_batch = act.to(self._device)
            obs_valid_batch = obs_valid.to(self._device)
            target_batch = (targets).to(self._device)
            act_target_batch = (act_targets).to(self._device)

            # Set Optimizer to Zero
            self._optimizer.zero_grad()

            # Forward Pass
            act_mean, out_mean, out_var, latent_target = self._model(obs_batch, act_batch, obs_valid_batch,
                                                                     target_batch)

            loss_inv = mse(act_target_batch, act_mean)

            if self._feature_space:
                loss_forward = mse(latent_target, out_mean)
            else:
                loss_forward = mse(target_batch, out_mean)

            fwd_mse += loss_forward.detach().cpu().numpy()
            inv_mse += loss_inv.detach().cpu().numpy()

            loss = loss_inv + self._lam * loss_forward

            loss.backward()

            if self._model.c.clip_gradients:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        avg_fwd_rmse = np.sqrt(fwd_mse / len(list(loader)))
        avg_inv_rmse = np.sqrt(inv_mse / len(list(loader)))

        return avg_fwd_rmse, avg_inv_rmse, t.time() - t0

    def eval(self, obs: np.ndarray, act: np.ndarray, obs_valid: np.ndarray, targets: np.ndarray,
             act_targets: np.ndarray, batch_size: int = -1) -> Tuple[float, float]:
        """
        :param obs: observations
        :param act: actions to feed the ac_predict stage
        :param obs_valid: observation valid flag
        :param targets: next observation targets
        :param act_targets: ground truth actions
        :param batch_size: batch size
        :return:
        """
        self._model.eval()
        dataset = TensorDataset(obs, act, obs_valid, targets, act_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        fwd_mse = inv_mse = 0
        num_batches = len(list(loader))
        for batch_idx, (obs, act, obs_valid, targets, act_targets) in enumerate(loader):
            with torch.no_grad():
                obs_batch = (obs).to(self._device)
                act_batch = act.to(self._device)
                obs_valid_batch = obs_valid.to(self._device)
                target_batch = (targets).to(self._device)
                act_target_batch = (act_targets).to(self._device)

                # Forward Pass
                act_mean, out_mean, out_var, latent_target = self._model(obs_batch, act_batch, obs_valid_batch,
                                                                         target_batch)

                loss_inv = mse(act_target_batch, act_mean)

                if self._feature_space:
                    loss_forward = mse(latent_target, out_mean)
                else:
                    loss_forward = mse(target_batch, out_mean)

                fwd_mse += loss_forward.detach().cpu().numpy()
                inv_mse += loss_inv.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        avg_fwd_rmse = np.sqrt(fwd_mse / num_batches)
        avg_inv_rmse = np.sqrt(inv_mse / num_batches)

        return avg_fwd_rmse, avg_inv_rmse

    def train(self, train_obs: torch.Tensor, train_act: torch.Tensor, train_obs_valid: torch.Tensor,
              train_targets: torch.Tensor, train_act_targets: torch.Tensor, epochs: int, batch_size: int,
              val_obs: torch.Tensor = None, val_act: torch.Tensor = None, val_obs_valid: torch.Tensor = None,
              val_targets: torch.Tensor = None, val_act_targets: torch.Tensor = None, val_interval: int = 1,
              val_batch_size: int = -1) -> None:
        """
        Train function
        :param train_obs: observations for training
        :param train_targets: targets for training
        :param train_act_targets: action targets for training
        :param epochs: epochs to train for
        :param batch_size: batch size for training
        :param val_obs: observations for validation
        :param val_targets: targets for validation
        :param val_act_targets: action targets for validation
        :param val_interval: validate every <this> iterations
        :param val_batch_size: batch size for validation, to save memory
        """

        """ Train Loop"""
        if val_batch_size == -1:
            val_batch_size = 4 * batch_size

        best_rmse = np.inf

        for i in range(epochs):
            train_fwd_rmse, train_inv_rmse, time = self.train_step(train_obs, train_act, train_obs_valid, train_targets,
                                                                   train_act_targets, batch_size)
            print("Training Iteration {:04d}: Forward RMSE: {:.5f}, Inverse RMSE: {:.5f}, Took {:4f} seconds".format(
                i + 1, train_fwd_rmse, train_inv_rmse, time))
            writer.add_scalar("Loss/train_forward", train_fwd_rmse, i)
            writer.add_scalar("Loss/train_inverse", train_inv_rmse, i)
            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                fwd_rmse, inv_rmse = self.eval(val_obs, val_act, val_obs_valid, val_targets, val_act_targets,
                                               batch_size=val_batch_size)
                print("Validation: Forward RMSE: {:.5f}, Inverse RMSE: {:.5f}".format(fwd_rmse, inv_rmse))
                if inv_rmse<best_rmse:
                    torch.save(self._model.state_dict(), self._save_path)
            writer.add_scalar("Loss/test_forward", fwd_rmse, i)
            writer.add_scalar("Loss/test_inverse", inv_rmse, i)
            writer.flush()
        writer.close()
