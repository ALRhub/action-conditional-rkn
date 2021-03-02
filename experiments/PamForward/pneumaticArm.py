import sys
sys.path.append('.')
import argparse
import os
from collections import OrderedDict

import numpy as np
import torch

from data.muscleData_NormSeq import mpgRobotSeq
from rkn.acrkn.AcRKN import AcRKN
from rkn.acrkn.ForwardLearning import Learn
from rkn.acrkn.ForwardInference import Infer
from rkn_cell.acrkn_cell import AcRKNCell
from util.metrics import naive_baseline
from util.metrics import root_mean_squared
from util.multistepRecurrent import longHorizon_Seq
from util.ConfigDict import ConfigDict


nn = torch.nn


def generate_pam_data_set(data, percentage_imputation):
    train_targets = data.train_targets
    test_targets = data.test_targets

    train_obs = data.train_obs
    test_obs = data.test_obs

    rs = np.random.RandomState(seed=42)
    train_obs_valid = rs.rand(train_targets.shape[0], train_targets.shape[1], 1) < 1 - percentage_imputation
    train_obs_valid[:, :5] = True
    print("Fraction of Valid Train Observations:",
          np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
    rs = np.random.RandomState(seed=23541)
    test_obs_valid = rs.rand(test_targets.shape[0], test_targets.shape[1], 1) < 1 - percentage_imputation
    test_obs_valid[:, :5] = True
    print("Fraction of Valid Train Observations:", np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))

    train_act = data.train_acts
    test_act = data.test_acts

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(
        train_obs_valid).bool(), torch.from_numpy(train_targets).float(), torch.from_numpy(
        test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(
        test_obs_valid).bool(), torch.from_numpy(test_targets).float()

"""Data"""
tar_type = 'delta'  #'delta' - if to train on differences to current states
                    #'next_state' - if to trian directly on the  next states
data = mpgRobotSeq(targets=tar_type, standardize=True)
impu = 0.75
train_obs, train_act, train_obs_valid, train_targets, test_obs, test_act, test_obs_valid, test_targets = generate_pam_data_set(
    data, impu)
act_dim = train_act.shape[-1]

"""Naive Baseline"""
naive_baseline(test_obs[:, :-1, :], test_obs[:, 1:, :], steps=[1, 3, 5, 7, 10], data=data, denorma=True)

"""Model Parameters"""
save_path = os.getcwd() + '/experiments/PamForward/saved_models/model.torch'

def experiment(encoder_dense, decoder_dense, batch_size, num_basis, control_basis, latent_obs_dim, lr, epochs, load,
               expName,
               gpu):
    '''
    joints : Give a list of joints (0-6) on which you want to train on eg: [1,4]
    lr: Learning Rate
    num_basis : Number of transition matrices to be learnt
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    class pamRKN(AcRKN):

        def __init__(self, target_dim: int, lod: int, lad: int, cell_config: ConfigDict,
                     layer_norm: bool,
                     use_cuda_if_available: bool = True):
            self._layer_norm = layer_norm
            super(pamRKN, self).__init__(target_dim, lod, lad, cell_config, use_cuda_if_available)

        def _build_enc_hidden_layers(self):
            layers = []
            # hidden layer 1
            layers.append(nn.Linear(in_features=4, out_features=encoder_dense))
            layers.append(nn.ReLU())
            nn.Dropout(0.5)
            return nn.ModuleList(layers), encoder_dense

        def _build_dec_hidden_layers_mean(self):
            return nn.ModuleList([
                nn.Linear(in_features=2 * self._lod, out_features=decoder_dense),
                nn.ReLU(),
                nn.Dropout(0.5)

            ]), decoder_dense

        def _build_dec_hidden_layers_var(self):
            return nn.ModuleList([
                nn.Linear(in_features=3 * self._lod, out_features=decoder_dense),
                nn.ReLU()
            ]), decoder_dense

    ##### Modify Config Parameters
    cell_conf = AcRKNCell.get_default_config()
    cell_conf._c_dict['num_basis'] = num_basis
    cell_conf._c_dict['control_net_hidden_units'] = [control_basis, control_basis, control_basis]
    cell_conf._c_dict['learning_rate'] = lr
    cell_conf._c_dict['never_invalid'] = False

    ##### Define Model, Train and Inference Modules

    acrkn_model = pamRKN(4, latent_obs_dim, act_dim, cell_config=cell_conf, layer_norm=True)
    acrkn_learn = Learn(acrkn_model, loss='rmse', metric='rmse', save_path=save_path)
    acrkn_infer = Infer(acrkn_model)

    if load == False:
        #### Train the Model
        acrkn_learn.train(train_obs, train_act, train_obs_valid, train_targets, int(epochs), batch_size, test_obs, test_act,
                          test_obs_valid, test_targets)

    ##### Load best model
    acrkn_model.load_state_dict(torch.load(save_path))


    # CALCULATING MULTISTEP PREDICTION METRICS

    def multiStepRMSE(steps=None, horizon=5, type='next_state'):
        for stepName, step in steps.items():
            lh = longHorizon_Seq(test_obs.cpu().detach(), test_act.cpu().detach(), test_targets.cpu().detach(),
                                 acrkn_infer, 4, data, steps=step, horizon=horizon,
                                 type=type, standardize=True)
            p, t = lh.multistep()
            print(root_mean_squared(p, t, data, denorma=True))

    multiStepRMSE(steps=OrderedDict({'1step': 1, '3step': 3, '5step': 5, '7step': 7, '10step': 10}),type=tar_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default=120, help='The network eg. googlenet')
    parser.add_argument('--decoder', default=120, help='The attack method eg. fgsm')
    parser.add_argument('--batch_size', default=500, help='Batch Size')
    parser.add_argument('--num_basis', default=15, help='Number Of Basis Matrices In Locally Linear Transition Model')
    parser.add_argument('--control_basis', default=120, help='Hidden Layer Size For Action Conditioning NN')
    parser.add_argument('--latent_dim', default=60, help='Latent observation Dimension Of RKN Cell')
    parser.add_argument('--lr', default=3e-3, help='Learning Rate')
    parser.add_argument('--epochs', default=250, help='Number Of Epochs To Train')
    parser.add_argument('--load', default=False, help='Whether to Load saved model or train from scratch')
    parser.add_argument('--exp', default='pam_forward', help='Name Of Experiment')
    parser.add_argument('--gpu', default='0', help='GPU number')
    args = parser.parse_args()
    experiment(int(args.decoder), int(args.encoder), int(args.batch_size), int(args.num_basis), int(args.control_basis),
               int(args.latent_dim), float(args.lr), int(args.epochs), args.load, args.exp, args.gpu)


if __name__ == '__main__':
    main()
