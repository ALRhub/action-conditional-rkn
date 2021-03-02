import os
import sys
sys.path.append('.')

import argparse
import torch
import numpy as np


from data.frankaData_Seq_Inv import frankaArmSeq
from rkn.acrkn.AcRknInv import AcRknInv
from rkn_cell.acrkn_cell import AcRKNCell
from rkn.acrkn.InverseLearning import Learn
from rkn.acrkn.InverseInference import Infer
from util.ConfigDict import ConfigDict
from util.metrics import naive_baseline, root_mean_squared
from util.dataProcess import diffToAct

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

nn = torch.nn


################ Generate Data ##################
def generate_franka_data_set(data, percentage_imputation):
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
    print("Fraction of Valid Test Observations:", np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))

    train_act = data.train_current_acts
    test_act = data.test_current_acts

    train_act_targets = data.train_act_targets
    test_act_targets = data.test_act_targets

    return torch.from_numpy(train_obs).float(), torch.from_numpy(train_act).float(), torch.from_numpy(
        train_obs_valid).float(), torch.from_numpy(train_targets).float(), torch.from_numpy(
        train_act_targets).float(), torch.from_numpy(
        test_obs).float(), torch.from_numpy(test_act).float(), torch.from_numpy(
        test_obs_valid).float(), torch.from_numpy(test_targets).float(), torch.from_numpy(test_act_targets).float(),


"""Data"""
dim = 14
tar_type = 'delta'  #'delta' - if to train on differences to previous actions/ current states
                    #'next_state' - if to trian directly the current ground truth actions / next states
data = frankaArmSeq(standardize=True, targets=tar_type)
impu = 0.00
train_obs, train_act, train_obs_valid, train_targets, train_act_targets, test_obs, test_act, test_obs_valid, test_targets, test_act_targets = generate_franka_data_set(
    data, impu)

"""Naive Baseline - Predicting Previous Actions"""
naive_baseline(train_act[:, :-1, :], train_act[:, 1:, :], data, 'actions', steps=[1, 3, 5, 10, 20], denorma=True)
naive_baseline(test_act[:, :-1, :], test_act[:, 1:, :], data, 'actions', steps=[1, 3, 5, 10, 20], denorma=True)

"""Model Parameters"""
latent_obs_dim = 15
act_dim = 7

batch_size = 1000
epochs = 250
save_path = os.getcwd() + '/experiments/Franka/saved_models/model.torch'


def experiment(encoder_dense, decoder_dense, act_decoder_dense, batch_size, num_basis, control_basis, latent_obs_dim, lr, epochs, load,
               expName,
               gpu):
    '''
    joints : Give a list of joints (0-6) on which you want to train on eg: [1,4]
    lr: Learning Rate
    num_basis : Number of transition matrices to be learnt
    '''


    class FrankaInvRKN(AcRknInv):

        def __init__(self, target_dim: int, lod: int, lad: int, cell_config: ConfigDict,
                     layer_norm: bool,
                     use_cuda_if_available: bool = True):
            self._layer_norm = layer_norm
            super(FrankaInvRKN, self).__init__(target_dim, lod, lad, cell_config, use_cuda_if_available)

        def _build_enc_hidden_layers(self):
            '''
            return: list of hidden layers and last hidden layer dimension
            '''
            layers = []
            # hidden layer 1
            layers.append(nn.Linear(in_features=dim, out_features=encoder_dense))
            layers.append(nn.ReLU())
            return nn.ModuleList(layers), encoder_dense

        def _build_target_enc_hidden_layers(self):
            '''
            return: list of hidden layers and last hidden layer dimension
            '''
            layers = []
            # hidden layer 1
            layers.append(nn.Linear(in_features=dim, out_features=encoder_dense))
            layers.append(nn.ReLU())
            return nn.ModuleList(layers), encoder_dense

        def _build_dec_hidden_layers_mean(self):
            '''
            return: list of hidden layers and last hidden layer dimension
            '''
            return nn.ModuleList([
                nn.Linear(in_features=2 * self._lod, out_features=decoder_dense),
                nn.ReLU(),

            ]), decoder_dense

        def _build_dec_hidden_layers_var(self):
            '''
            This is used only when you train with 'nll' loss. We use mse loss in most experiments.

            return: list of hidden layers and last hidden layer dimension
            '''
            return nn.ModuleList([
                nn.Linear(in_features=3 * self._lod, out_features=decoder_dense),
                nn.ReLU()
            ]), decoder_dense

        def _build_act_dec_hidden_layers_mean(self):
            '''
            return: list of hidden layers and last hidden layer dimension
            '''
            return nn.ModuleList([
                nn.Linear(in_features=3 * self._lod, out_features=act_decoder_dense),
                nn.ReLU(),

            ]), act_decoder_dense

    ##### Modify Config Parameters
    cell_conf = AcRKNCell.get_default_config()
    cell_conf._c_dict['num_basis'] = num_basis
    cell_conf._c_dict['control_net_hidden_units'] = [control_basis, control_basis, control_basis]
    cell_conf._c_dict['learning_rate'] = lr

    ##### Define Model and Train It

    acrkn_model = FrankaInvRKN(dim, latent_obs_dim, act_dim, cell_config=None, layer_norm=True)
    acrkn_learn = Learn(acrkn_model, lam = 0.1, feature_space=False, save_path = save_path)
    acrkn_infer = Infer(acrkn_model)

    ##### Train the model
    if load == True:
        acrkn_learn.train(train_obs, train_act, train_obs_valid, train_targets, train_act_targets, epochs, batch_size, test_obs,
                    test_act, test_obs_valid, test_targets, test_act_targets, val_batch_size=batch_size)


    ##### Load best model
    acrkn_model.load_state_dict(torch.load(save_path))

    ##### Test RMSE
    pred = acrkn_infer.predict(test_obs, test_act, test_obs_valid, test_targets, batch_size=500)


    if tar_type=='delta':
        pred,_ = diffToAct(pred.cpu().detach().numpy(),data.test_prev_acts,data,standardize=True)
    else:
        pred = pred.cpu().detach().numpy()

    rmse = root_mean_squared(pred, test_act.cpu().detach().cpu(), data, tar='actions', denorma=True,
                             plot=[1, 2, 3])
    print('Inverse RMSE Final', rmse)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', default=120, help='Observation Encoder Hidden Layer Size')
    parser.add_argument('--decoder', default=240, help='Observation Decoder Hidden Layer Size')
    parser.add_argument('--act_decoder', default=512, help='Action Decoder Hidden Layer Size')
    parser.add_argument('--batch_size', default=250, help='Batch Size')
    parser.add_argument('--num_basis', default=15, help='Number Of Basis Matrices In Locally Linear Transition Model')
    parser.add_argument('--control_basis', default=120, help='Hidden Layer Size For Action Conditioning NN')
    parser.add_argument('--latent_dim', default=60, help='Latent State Dimension Of RKN Cell')
    parser.add_argument('--lr', default=7e-3, help='Learning Rate')
    parser.add_argument('--epochs', default=750, help='Number Of Epochs To Train')
    parser.add_argument('--load', default=False, help='If to Load saved model or train from scratch')
    parser.add_argument('--exp', default='franka_inverse', help='Name Of Experiment')
    parser.add_argument('--gpu', default='0', help='GPU number')
    args = parser.parse_args()
    experiment(int(args.encoder), int(args.decoder), int(args.act_decoder), int(args.batch_size), int(args.num_basis),
               int(args.control_basis), int(args.latent_dim), float(args.lr),int(args.epochs), args.load, args.exp,
               args.gpu)


if __name__ == '__main__':
    main()