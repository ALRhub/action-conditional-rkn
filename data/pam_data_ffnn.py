import json
import os

import numpy as np


class mpgRobot():
    """
    The data relates to musculo-skeltal robot arm from Max Plank Tubingen.
    """

    name = 'mpgrobot'

    def __init__(self,standardize=True,targets='next_state'):

        self.datapath = os.getcwd() + '/data/PAMData/Data2/'
        self.dim = 4
        self.downsample = 1
        self.appendValue = -3
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

        data_in, data_out =  self._loop_data()
        #print(data_in.shape,data_out.shape)

        #randomize
        arr = np.arange(data_in.shape[0])
        np.random.seed(seed=np.random.randint(0,100)) #122
        np.random.shuffle(arr)
        data_in = data_in[arr, :, :]
        data_out = data_out[arr, :,:]

        #train_test_split
        numData = data_in.shape[0]
        self.num_Train = int(0.8 * numData)
        self.num_Test = numData - self.num_Train
        data_train_in = data_in[:self.num_Train,:,:]
        data_train_out = data_out[:self.num_Train,:,:]
        data_test_in = data_in[self.num_Train:,:,:]
        data_test_out = data_out[self.num_Train:,:,:]

        #create (state,action,next_state)
        train_obs = data_train_in[:, :-1, :]; test_obs = data_test_in[:, :-1, :]
        train_act = data_train_out[:, :-1, :]; test_act = data_test_out[:, :-1, :]
        train_targets = data_train_in[:, 1:, :];test_targets = data_test_in[:, 1:, :]

        # get different statistics for state, actions, delta_state, delta_action and residuals which will be used for standardization
        mean_state_diff, std_state_diff = self.get_statistics(train_obs, self.dim, difference=True)
        mean_obs, std_obs = self.get_statistics(train_obs, self.dim)
        mean_act, std_act = self.get_statistics(train_act, 2*self.dim)
        self.normalization = dict()
        self.normalization['observations'] = [mean_obs, std_obs]
        self.normalization['actions'] = [mean_act, std_act]
        self.normalization['diff'] = [mean_state_diff, std_state_diff]

        # Observation Valid Setting Flag
        rs = np.random.RandomState(seed=42)
        percentage_imputation = 0.5
        train_obs_valid = rs.rand(train_targets.shape[0], train_targets.shape[1], 1) < 1 - percentage_imputation
        train_obs_valid[:, :5] = True
        print("Fraction of Valid Train Observations:",
              np.count_nonzero(train_obs_valid) / np.prod(train_obs_valid.shape))
        rs = np.random.RandomState(seed=23541)
        test_obs_valid = rs.rand(test_targets.shape[0], test_targets.shape[1], 1) < 1 - percentage_imputation
        test_obs_valid[:, :5] = True
        print("Fraction of Valid Test Observations:", np.count_nonzero(test_obs_valid) / np.prod(test_obs_valid.shape))


        # reshape
        reshape = lambda x: np.reshape(x, (x.shape[0] * x.shape[1], -1))
        train_targets = reshape(train_targets);
        train_obs = reshape(train_obs);
        train_act = reshape(train_act);
        self.train_obs_valid = reshape(train_obs_valid);
        test_targets = reshape(test_targets);
        test_obs = reshape(test_obs);
        test_act = reshape(test_act)
        self.test_obs_valid = reshape(test_obs_valid);



        # compute delta
        if self.tar_type == 'delta':
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On Differences <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            train_targets = (train_targets - train_obs)
            test_targets = (test_targets - test_obs)
            self.normalization['targets'] = [mean_state_diff, std_state_diff]
        else:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>> Training On Next States(not differences) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            self.normalization['diff'] = [mean_obs, std_obs]


        # Standardize
        if self.standardize:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>Standardizing The Data<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")



            self.train_obs = self.normalize(train_obs, self.normalization["observations"][0],
                                            self.normalization["observations"][1])
            self.train_acts = self.normalize(train_act, self.normalization["actions"][0],
                                             self.normalization["actions"][1])

            self.test_obs = self.normalize(test_obs, self.normalization["observations"][0],
                                           self.normalization["observations"][1])
            self.test_acts = self.normalize(test_act, self.normalization["actions"][0],
                                            self.normalization["actions"][1])

            if self.tar_type == 'delta':
                self.train_targets = self.normalize(train_targets, self.normalization["diff"][0],
                                                    self.normalization["diff"][1])
                self.test_targets = self.normalize(test_targets, self.normalization["diff"][0],
                                                   self.normalization["diff"][1])
            else:
                self.train_targets = self.normalize(train_targets, self.normalization["next_state"][0],
                                                    self.normalization["next_state"][1])
                self.test_targets = self.normalize(test_targets, self.normalization["next_state"][0],
                                                   self.normalization["next_state"][1])


        else:
            self.train_obs = train_obs
            self.train_acts = train_act
            self.train_targets = train_targets
            self.test_obs = test_obs
            self.test_acts = test_act
            self.test_targets = test_targets

        return True




    def _load_file(self, f):
        # Load each file. Each episode/file is between length approx 80 and 150.
        # Hence make every episode to be of equal length 160 by padding with a high negative value(-3).
        with open(self.datapath + f) as json_file:
            data = json.load(json_file)
        #data_in = np.array(data['ob'])[:, [0,1,2,3,8,9,10,11,12,13,14,15]]
        data_in = np.array(data['ob'])[:, [0, 1, 2, 3]]
        #print(np.max(np.max(np.max(data_in))),np.min(np.min(np.min(data_in))))
        data_out = np.cumsum(np.array(data['action'])[:, :8], 1)
        #print(data_out.shape)
        self.episode_length = 40
        H = data_in.shape[0]

        data_ep_in = [data_in[ind:ind + self.episode_length, :] for ind in range(0, H, self.episode_length) if
                      (ind + self.episode_length < H - 1)]
        data_ep_out = [data_out[ind:ind + self.episode_length, :] for ind in range(0, H, self.episode_length) if
                       (ind + self.episode_length < H - 1)]

        return np.array(data_ep_in), np.array(data_ep_out)
