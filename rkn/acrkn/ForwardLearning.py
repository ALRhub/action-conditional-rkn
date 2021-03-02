import os
import time as t
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from rkn.acrkn.AcRKN import AcRKN
from util.Losses import mse, gaussian_nll

writer = SummaryWriter('/home/vshaj/CLAS/Logs/ALRhub/fwd_noinit5')

optim = torch.optim
nn = torch.nn


class Learn:

    def __init__(self, model: AcRKN, loss: str, metric: str, save_path=None, use_cuda_if_available: bool = True):
        """
        :param model: nn module for rkn
        :param loss: type of loss to train on 'nll' or 'mse'
        :param metric: type of metric to print during training 'nll' or 'mse'
        :param use_cuda_if_available: if gpu training set to True
        """

        self._device = torch.device("cuda" if torch.cuda.is_available() and use_cuda_if_available else "cpu")
        self._loss = loss
        self._metric = metric
        self._model = model
        self._learning_rate = self._model.c.learning_rate
        if save_path is None:
            self._save_path = os.getcwd() + '/experiments/Franka/saved_models/model.torch'
        else:
            self._save_path = save_path

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)
        self._shuffle_rng = np.random.RandomState(42)  # rng for shuffling batches

    def train_step(self, train_obs: np.ndarray, train_act: np.ndarray, train_obs_valid: np.ndarray,
                   train_targets: np.ndarray, batch_size: int) \
            -> Tuple[float, float, float]:
        """
        Train once on the entire dataset
        :param train_obs: training observations
        :param train_act: training actions
        :param train_obs_valid: training valid flag
        :param train_targets: training targets
        :param batch_size:
        :return: average loss (nll) and  average metric (rmse), execution time
        """
        self._model.train()
        dataset = TensorDataset(train_obs, train_act, train_obs_valid, train_targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        avg_loss = avg_metric = 0
        t0 = t.time()
        b = list(loader)[0]

        for batch_idx, (obs, act, obs_valid, targets) in enumerate(loader):
            # Assign tensors to device
            obs_batch = (obs).to(self._device)
            act_batch = act.to(self._device)
            obs_valid_batch = obs_valid.to(self._device)
            target_batch = (targets).to(self._device)
            # Set Optimizer to Zero
            self._optimizer.zero_grad()

            # Forward Pass
            out_mean, out_var = self._model(obs_batch, act_batch, obs_valid_batch)

            ## Calculate Loss
            if self._loss == 'nll':
                loss = gaussian_nll(target_batch, out_mean, out_var)
            else:
                loss = mse(target_batch, out_mean)

            # Backward Pass
            loss.backward()

            # Clip Gradients
            if self._model.c.clip_gradients:
                torch.nn.utils.clip_grad_norm(self._model.parameters(), 5.0)

            # Backward Pass Via Optimizer
            self._optimizer.step()

            with torch.no_grad():
                if self._metric == 'nll':
                    metric = gaussian_nll(target_batch, out_mean, out_var)
                else:
                    metric = mse(target_batch, out_mean)
            avg_loss += loss.detach().cpu().numpy()
            avg_metric += metric.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))
        if self._metric == 'nll':
            avg_metric = avg_metric / len(list(loader))
        else:
            avg_metric = np.sqrt(avg_metric / len(list(loader)))

        return avg_loss, avg_metric, t.time() - t0

    def eval(self, obs: np.ndarray, act: np.ndarray, obs_valid: np.ndarray, targets: np.ndarray,
             batch_size: int = -1) -> Tuple[float, float]:
        """
        Evaluate model
        :param obs: observations to evaluate on
        :param act: actions to evalauate on
        :param obs_valid: observation valid flag
        :param targets: targets to evaluate on
        :batch_size: batch_size for evaluation, this does not change the results and is only to allow evaluating on more
         data than you can fit in memory at once. Default: -1, .i.e. batch_size = number of sequences.
        """
        # rescale only batches so the data can be kept in unit8 to lower memory consumptions
        self._model.eval()
        dataset = TensorDataset(obs, act, obs_valid, targets)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        avg_loss = 0.0
        avg_metric = 0.0

        for batch_idx, (obs_batch, act_batch, obs_valid_batch, targets_batch) in enumerate(loader):
            with torch.no_grad():
                # Assign tensors to devices
                obs_batch = (obs_batch).to(self._device)
                act_batch = act_batch.to(self._device)
                obs_valid_batch = obs_valid_batch.to(self._device)
                targets_batch = (targets_batch).to(self._device)

                # Forward Pass
                out_mean, out_var = self._model(obs_batch, act_batch, obs_valid_batch)

                ## Calculate Loss
                if self._loss == 'nll':
                    loss = gaussian_nll(targets_batch, out_mean, out_var)
                else:
                    loss = mse(targets_batch, out_mean)

                if self._metric == 'nll':
                    metric = gaussian_nll(targets_batch, out_mean, out_var)
                else:
                    metric = mse(targets_batch, out_mean)

                avg_loss += loss.detach().cpu().numpy()
                avg_metric += metric.detach().cpu().numpy()

        # taking sqrt of final avg_mse gives us rmse across an apoch without being sensitive to batch size
        if self._loss == 'nll':
            avg_loss = avg_loss / len(list(loader))
        else:
            avg_loss = np.sqrt(avg_loss / len(list(loader)))

        if self._metric == 'nll':
            avg_metric = avg_metric / len(list(loader))
        else:
            avg_metric = np.sqrt(avg_metric / len(list(loader)))

        return avg_loss, avg_metric

    def train(self, train_obs: torch.Tensor, train_act: torch.Tensor, train_obs_valid: torch.Tensor,
              train_targets: torch.Tensor, epochs: int, batch_size: int,
              val_obs: torch.Tensor = None, val_act: torch.Tensor = None, val_obs_valid: torch.Tensor = None,
              val_targets: torch.Tensor = None, val_interval: int = 1,
              val_batch_size: int = -1) -> None:
        """
        Train function
        :param train_obs: observations for training
        :param train_targets: targets for training
        :param epochs: number of epochs to train for
        :param batch_size: batch size for training
        :param val_obs: observations for validation
        :param val_targets: targets for validation
        :param val_interval: validate every <this> iterations
        :param val_batch_size: batch size for validation, to save memory
        """

        """ Train Loop"""
        if val_batch_size == -1:
            val_batch_size = 4 * batch_size
        best_loss = np.inf

        for i in range(epochs):
            train_loss, train_metric, time = self.train_step(train_obs, train_act, train_obs_valid, train_targets,
                                                             batch_size)
            print("Training Iteration {:04d}: {}:{:.5f}, {}:{:.5f}, Took {:4f} seconds".format(
                i + 1, self._loss, train_loss, self._metric, train_metric, time))
            writer.add_scalar("Loss/train_forward", train_loss, i)
            if val_obs is not None and val_targets is not None and i % val_interval == 0:
                val_loss, val_metric = self.eval(val_obs, val_act, val_obs_valid, val_targets,
                                                 batch_size=val_batch_size)
                if val_loss<best_loss:
                    #TODO : Currently this condition is defined only wrt rmse loss in mind.
                    torch.save(self._model.state_dict(), self._save_path)
                print("Validation: {}: {:.5f}, {}: {:.5f}".format(self._loss, val_loss, self._metric, val_metric))
                writer.add_scalar("Loss/test_forward", val_loss, i)
                writer.flush()
            writer.close()
