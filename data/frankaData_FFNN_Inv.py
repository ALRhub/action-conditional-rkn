import os

import numpy as np


class boschRobot():
    """
    The data relates to franka robot arm from Bosch.
    Normalization Statistics saved as 'observations','actions','state_diff' and
    'act_diff'
    """

    name = 'boschrobot'

    def __init__(self, standardize=True, targets='current_state', dim=14):
        self.datapath = os.getcwd() + '/data/FrankaData/rubber_acce/without/'
        self.dim = dim
        self.downsample = 1
        self.standardize = standardize
        self.tar_type = targets
        self.normalization = None
        self._load_data()

    def normalize(self, data, mean, std):
        return (data - mean) / (std + 1e-10)

    def denormalize(self, data, mean, std):
        return data * (std + 1e-10) + mean

    def get_statistics(self, data, dim, difference=False):
        if difference:
            data = (data[:, 1:, :dim] - data[:, :-1, :dim])
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
        data = reshape(data);
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    def _loop_data(self):
        firstFlag = True
        for f in os.listdir(self.datapath):
            data_in, data_out = self._load_file(f)
            if firstFlag:
                full_data_in = data_in
                full_data_out = data_out
                firstFlag = False
            else:
                full_data_in = np.concatenate((full_data_in, data_in))
                full_data_out = np.concatenate((full_data_out, data_out))
        return np.array(full_data_in), np.array(full_data_out)

    def _load_data(self):
        '''
        go through each files and get the necessary training data (state, actions, targets,
        normalization statistics) etc
        '''
        # train_file='experiment_result_0_slow.npz' ;test_file='experiment_result_1_fast.npz' ;
        # data_train_in, data_train_out = self._load_file(train_file)
        # data_test_in, data_test_out = self._load_file(test_file)

        data_in, data_out = self._loop_data()
        # randomize
        arr = np.arange(data_in.shape[0])
        np.random.seed(1234)  ###usually 42
        np.random.shuffle(arr)
        data_in = data_in[arr, :, :]
        data_out = data_out[arr, :, :]

        # train_test_split
        numData = data_in.shape[0]
        self.num_Train = int(0.8 * numData)
        self.num_Test = numData - self.num_Train
        data_train_in = data_in[:self.num_Train, :, :]
        data_train_out = data_out[:self.num_Train, :, :]
        data_test_in = data_in[self.num_Train:, :, :]
        data_test_out = data_out[self.num_Train:, :, :]

        # train_test_split
        self.num_Train = data_train_in.shape[0]
        self.num_Test = data_test_in.shape[0]

        # create (state,action,next_state) tuple

        train_obs = data_train_in[:, 1:-1, :self.dim];
        test_obs = data_test_in[:, 1:-1, :self.dim]
        train_prev_act = data_train_out[:, :-2, :];
        test_prev_act = data_test_out[:, :-2, :]
        train_current_act = data_train_out[:, 1:-1, :];
        test_current_act = data_test_out[:, 1:-1, :]

        train_next_state = data_train_in[:, 2:, :self.dim];
        test_next_state = data_test_in[:, 2:, :self.dim]

        # get different statistics for state, actions, delta_state, delta_action and residuals which will be used for standardization
        mean_state_diff, std_state_diff = self.get_statistics(train_obs, self.dim, difference=True)
        mean_act_diff, std_act_diff = self.get_statistics(train_prev_act, 7, difference=True)
        mean_obs, std_obs = self.get_statistics(train_obs, self.dim)
        mean_act, std_act = self.get_statistics(train_prev_act, 7)

        self.normalization = dict()
        self.normalization['observations'] = [mean_obs, std_obs]
        self.normalization['actions'] = [mean_act, std_act]
        self.normalization['state_diff'] = [mean_state_diff, std_state_diff]
        self.normalization['act_diff'] = [mean_act_diff, std_act_diff]

        # setting obs_valid_flag

        rs = np.random.RandomState(seed=42)
        percentage_imputation = 0.5
        train_obs_valid = rs.rand(train_next_state.shape[0], train_next_state.shape[1], 1) < 1 - percentage_imputation
        train_obs_valid[:, :5] = True
        print("Fraction of Valid Train Observations:",
              np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
        rs = np.random.RandomState(seed=23541)
        test_obs_valid = rs.rand(test_next_state.shape[0], test_next_state.shape[1], 1) < 1 - percentage_imputation
        test_obs_valid[:, :5] = True
        print("Fraction of Valid Test Observations:", np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))

        # reshape
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
        train_next_state = reshape(train_next_state);
        train_obs = reshape(train_obs);
        train_prev_act = reshape(train_prev_act);
        train_current_act = reshape(train_current_act);
        self.train_obs_valid = reshape(train_obs_valid);
        test_next_state = reshape(test_next_state);
        test_obs = reshape(test_obs);
        test_prev_act = reshape(test_prev_act)
        test_current_act = reshape(test_current_act)
        self.test_obs_valid = reshape(test_obs_valid);

        # mean_res_diff, std_res_diff = self.get_residual_statistics(train_obs)

        # compute delta
        if self.tar_type == 'delta':
            train_targets = (train_next_state - train_obs)
            test_targets = (test_next_state - test_obs)
            train_act_targets = (train_current_act - train_prev_act)
            test_act_targets = (test_current_act - test_prev_act)
        else:
            train_targets = train_next_state
            test_targets = test_next_state
            train_act_targets = train_current_act
            test_act_targets = test_current_act

        # Standardize
        if self.standardize:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Standardizing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

            self.train_obs = self.normalize(train_obs, self.normalization["observations"][0],
                                            self.normalization["observations"][1])
            self.train_current_acts = self.normalize(train_current_act, self.normalization["actions"][0],
                                                     self.normalization["actions"][1])

            self.train_prev_acts = self.normalize(train_prev_act, self.normalization["actions"][0],
                                                  self.normalization["actions"][1])
            self.test_obs = self.normalize(test_obs, self.normalization["observations"][0],
                                           self.normalization["observations"][1])
            self.test_current_acts = self.normalize(test_current_act, self.normalization["actions"][0],
                                                    self.normalization["actions"][1])

            self.test_prev_acts = self.normalize(test_prev_act, self.normalization["actions"][0],
                                                 self.normalization["actions"][1])
            if self.tar_type == 'delta':
                self.train_targets = self.normalize(train_targets, self.normalization["state_diff"][0],
                                                    self.normalization["state_diff"][1])
                self.test_targets = self.normalize(test_targets, self.normalization["state_diff"][0],
                                                   self.normalization["state_diff"][1])
                self.train_act_targets = self.normalize(train_act_targets, self.normalization["act_diff"][0],
                                                        self.normalization["act_diff"][1])
                self.test_act_targets = self.normalize(test_act_targets, self.normalization["act_diff"][0],
                                                       self.normalization["act_diff"][1])
            else:
                self.train_targets = self.normalize(train_targets, self.normalization["observations"][0],
                                                    self.normalization["observations"][1])
                self.test_targets = self.normalize(test_targets, self.normalization["observations"][0],
                                                   self.normalization["observations"][1])
                self.train_act_targets = self.normalize(train_current_act, self.normalization["actions"][0],
                                                        self.normalization["actions"][1])
                self.test_act_targets = self.normalize(test_current_act, self.normalization["actions"][0],
                                                       self.normalization["actions"][1])
        else:
            self.train_obs = train_obs
            self.train_prev_acts = train_prev_act
            self.train_current_acts = train_current_act
            self.train_act_targets = train_act_targets
            self.train_targets = train_targets
            self.test_obs = test_obs
            self.test_prev_acts = test_prev_act
            self.test_current_acts = test_current_act
            self.test_act_targets = test_act_targets
            self.test_targets = test_targets
        return True

    def _load_file(self, f):
        # Load each file. Specify Episode Length.
        # Episodize
        data = np.load(self.datapath + f)
        data_in = np.concatenate((data['q'], data['qd'], data['qdd']), -1)
        data_out = data['tau']
        self.episode_length = 100
        H = data_in.shape[0]

        data_ep_in = [data_in[ind:ind + self.episode_length, :] for ind in range(0, H, self.episode_length) if
                      (ind + self.episode_length < H - 1)]
        data_ep_out = [data_out[ind:ind + self.episode_length, :] for ind in range(0, H, self.episode_length) if
                       (ind + self.episode_length < H - 1)]

        return np.array(data_ep_in), np.array(data_ep_out)
