import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from data.pam_data_ffnn import mpgRobot

class mpgRobotSeq(mpgRobot):
    """
    The data relates to an inverse dynamics problem for a seven
    degrees-of-freedom SARCOS anthropomorphic robot arm. The task is
    to map from a 21-dimensional input space (7 joint positions,
    7 joint velocities, 7 joint accelerations) to the corresponding
    7 joint torques.

    Training data: sarcos_inv, 44484x28, double array
    Test data: sarcos_inv_test, 4449x28, double array

    Data order:
    7 joint pos, 7 joint vel, 7 joint acc, 7 joint torques

    Previously used to learn the mapping from (pos, vel, acc) --> torques

    These data had previously been used in the papers
     - LWPR: An O(n) Algorithm for Incremental Real Time Learning
       in High Dimensional Space, S. Vijayakumar and S. Schaal,
       Proc ICML 2000, 1079-1086 (2000).
     - Statistical Learning for Humanoid Robots, S. Vijayakumar,
       A. D'Souza, T. Shibata, J. Conradt, S. Schaal, Autonomous
       Robot, 12(1) 55-69 (2002)
     - Incremental Online Learning in High Dimensions S. Vijayakumar,
       A. D'Souza, S. Schaal, Neural Computation 17(12) 2602-2634 (2005)
    """

    name = 'mpgrobot'

    def __init__(self,standardize=True,targets='next_state'):
        super(mpgRobotSeq, self).__init__(standardize=standardize,targets=targets)
        self._getSequenceData()

    def _getSequenceData(self):
        '''This function reshapes the non sequential data of shape [numEp*epLen,DataDim] into
        sequential data of dim [numEp,epLen,DataDim]
        '''
        # reshape
        #numEp=3888; epLen=99
        numEp=self.num_Train; epLen=self.episode_length-1
        reshape = lambda x: np.reshape(x, (numEp,epLen, -1))
        self.train_targets = reshape(self.train_targets);
        self.train_obs = reshape(self.train_obs);
        self.train_acts = reshape(self.train_acts);
        self.train_obs_valid = reshape(self.train_obs_valid);
        #numEp = 972;epLen = 99
        numEp = self.num_Test;epLen = self.episode_length-1
        reshape = lambda x: np.reshape(x, (numEp, epLen, -1))
        self.test_targets = reshape(self.test_targets);
        self.test_obs = reshape(self.test_obs);
        self.test_acts = reshape(self.test_acts)
        self.test_obs_valid = reshape(self.test_obs_valid);





