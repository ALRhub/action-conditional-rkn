import numpy as np
from data.frankaData_FFNN_Inv import boschRobot

class frankaArmSeq(boschRobot):
    """
    The data relates to the one collected from Franka Arms pushing 6 sets of
    different weights at the University Of Lincoln.

    """

    name = 'boscharm'

    def __init__(self,standardize=True,targets='next_state',dim=14):
        print(targets)
        super(frankaArmSeq, self).__init__(standardize=standardize,targets=targets,dim=dim)
        self._getSequenceData()

    def _getSequenceData(self):
        '''This function reshapes the non sequential data of shape [numEp*epLen,DataDim] into
        sequential data of dim [numEp,epLen,DataDim]
        '''
        # reshape
        #numEp=3888; epLen=99
        numEp=self.num_Train; epLen=self.episode_length-2
        reshape = lambda x: np.reshape(x, (numEp,epLen, -1))
        print(self.train_targets.shape)
        self.train_targets = reshape(self.train_targets);
        self.train_obs = reshape(self.train_obs);
        self.train_prev_acts = reshape(self.train_prev_acts);
        self.train_current_acts = reshape(self.train_current_acts)
        self.train_act_targets = reshape(self.train_act_targets)
        self.train_obs_valid = reshape(self.train_obs_valid);


        #numEp = 972;epLen = 99
        numEp = self.num_Test;epLen = self.episode_length-2
        reshape = lambda x: np.reshape(x, (numEp, epLen, -1))
        self.test_targets = reshape(self.test_targets);
        self.test_obs = reshape(self.test_obs);
        self.test_prev_acts = reshape(self.test_prev_acts);
        self.test_current_acts = reshape(self.test_current_acts)
        self.test_act_targets = reshape(self.test_act_targets)
        self.test_obs_valid = reshape(self.test_obs_valid);





