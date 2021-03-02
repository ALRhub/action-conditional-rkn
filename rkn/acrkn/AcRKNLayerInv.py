import torch
from rkn_cell.kalman_ops.update import Update
from util.ConfigDict import ConfigDict
from rkn_cell.kalman_ops.ac_predict import AcPredict
nn = torch.nn


class AcRKNLayerInv(nn.Module):
    @staticmethod
    def get_default_config() -> ConfigDict:
        config = ConfigDict(
            num_basis=15,
            bandwidth=3,
            trans_net_hidden_units=[],
            control_net_hidden_units=[60],
            trans_net_hidden_activation="Tanh",
            control_net_hidden_activation='ReLU',
            learn_trans_covar=True,
            trans_covar=1,
            learn_initial_state_covar=True,
            initial_state_covar=1,
            learning_rate=7e-3,
            enc_out_norm='post',
            clip_gradients=True,
            never_invalid=True
        )
        config.finalize_adding()
        return config

    def __init__(self, latent_obs_dim: int, act_dim: int, cell_config: ConfigDict = None, dtype=torch.float32):
        super().__init__()
        if cell_config == None:
            self.c = self.get_default_config()
        else:
            self.c = cell_config
        self._lad = act_dim
        self._lod = latent_obs_dim
        self._lsd = 2 * latent_obs_dim
        self._update = Update(self._lod, self.c)
        self._predict = AcPredict(self._lod, self._lad, self.c)

    def forward(self, latent_obs, obs_vars, action, initial_mean, initial_cov, obs_valid=None):
        """
        This currently only returns the posteriors. If you also need the priors uncomment the corresponding parts

        :param latent_obs: latent observations
        :param obs_vars: uncertainty estimate in latent observations
        :param action: control signals
        :param initial_mean: mean of initial belief
        :param initial_cov: covariance of initial belief (as 3 vectors)
        :param obs_valid: flags indicating which observations are valid, which are not
        """

        # tif you need a version that also returns the prior uncomment the respective parts below
        # prepare list for return

        prior_mean_list = []
        prior_cov_list = [[], [], []]

        post_mean_list = []
        post_cov_list = [[], [], []]


        # initialize prior
        prior_mean, prior_cov = initial_mean, initial_cov

        # actual computation
        for i in range(latent_obs.shape[1]):

            cur_obs_valid = obs_valid[:, i] if obs_valid is not None else None

            # update belief with updateLayer
            post_mean, post_cov = \
                self._update(prior_mean, prior_cov, latent_obs[:, i], obs_vars[:, i], cur_obs_valid)

            post_mean_list.append(post_mean)
            [post_cov_list[i].append(post_cov[i]) for i in range(3)]
            import numpy as np

            # predict next belief state ahead in time
            next_prior_mean, next_prior_cov = self._predict(post_mean, post_cov, action[:, i])

            prior_mean_list.append(next_prior_mean)
            [prior_cov_list[i].append(next_prior_cov[i]) for i in range(3)]

            prior_mean = next_prior_mean
            prior_cov = next_prior_cov

        # stack results
        prior_means = torch.stack(prior_mean_list, 1)
        prior_covs = [torch.stack(x, 1) for x in prior_cov_list]

        post_means = torch.stack(post_mean_list, 1)
        post_covs = [torch.stack(x, 1) for x in post_cov_list]

        return post_means, post_covs, prior_means, prior_covs
