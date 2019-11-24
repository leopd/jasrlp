import copy
from collections import deque
import gym
import numpy as np
import random
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import warnings

from timebudget import timebudget
timebudget.set_quiet()

from typing import List, Union, NamedTuple, Tuple

from rldqn import DQN, FCNet, RandomLearner


def box_scale(space:gym.spaces.box.Box) -> float:
    """Returns the scaling factor for an action space.  If the action space is [-2,2] Box, this outputs 2.
    Lots of assertions assuming all dimensions are the same.
    """
    lo = min(space.low)
    assert lo == max(space.low), "Action space is anisotropic"
    hi = min(space.high)
    assert hi == max(space.high), "Action space is anisotropic"
    assert lo == (-hi), "Action space is assymetric"
    return hi


class DDPG(DQN):

    def __init__(self, env, eps:float=0.5, gamma:float=0.99, net_args:dict={}):
        super().__init__(env, eps, gamma, net_args)
        del self.opt
        self.opt_q = torch.optim.Adam(params=self.qnet.parameters())
        self.opt_mu = torch.optim.Adam(params=self.munet.parameters())

    def init_env(self, env):
        self.env = env
        self.obs_space = env.observation_space
        assert self.obs_space.__class__.__name__ == "Box", "Only Box observation spaces supported"
        self.act_space = env.action_space
        assert self.act_space.__class__.__name__ == "Box", "Only Box action spaces supported"
        self.obs_dim = np.prod(self.obs_space.shape)
        self.act_dim = np.prod(self.act_space.shape)
        self.tau = 0.001  # HYPERPARAMETER

    def build_nets(self, env, net_args):
        if 'hidden_dims' not in net_args:
            net_args['hidden_dims'] = [64,64]
        net_args['activation'] = nn.Tanh  # gotta be for correct output scaling.
        out_scale = box_scale(env.action_space)

        # Actor network: mu
        in_dim = self.obs_dim
        out_dim = self.act_dim
        self.munet = FCNet(in_dim, out_dim, final_activation=True, output_scaling=out_scale, **net_args)
        self.target_munet = copy.deepcopy(self.munet)
        print(f"Actor (mu): {self.munet}")

        # Critic network: q
        in_dim = self.obs_dim + self.act_dim
        out_dim = 1
        self.qnet = FCNet(in_dim, out_dim, final_activation=False, **net_args)
        self.target_qnet = copy.deepcopy(self.qnet)
        print(f"Critic (Q): {self.qnet}")


    def target_nets_elastic_follow(self):
        """Update the two target networks with self.tau * the online network
        """
        self._target_update(self.target_qnet, self.qnet, self.tau)
        self._target_update(self.target_munet, self.munet, self.tau)

    def _target_update(self, target:nn.Module, online:nn.Module, tau:float):
        assert target.state_dict().keys() == online.state_dict().keys()
        update = target.state_dict()
        for key in target.state_dict().keys():
            old = target.state_dict()[key]
            nu = online.state_dict()[key]
            update[key] = old * (1.0 - tau) + tau * nu
        target.load_state_dict(update)
        target.eval()

    def get_greedy_action(self, obs):
        with torch.no_grad():
            self.qnet.eval()
            action_batch_of_1 = self.munet.calc_qval_batch([obs])
            action_vec = action_batch_of_1[0,:]
            return action_vec.cpu().numpy()

    @timebudget
    def do_learning(self):
        if len(self._replay) < self.minimum_transitions_in_replay:
            return
        minibatch_size = self.minibatch_size
        batch = self._replay.sample(minibatch_size)
        assert batch is not None

        # Implement DDPG learning algorithm.
        s, a, s1, r, f = batch

        makevec = lambda t: t.view(-1)

        # First update online Q network
        self.opt_q.zero_grad()
        self.qnet.train()
        sa = torch.cat([s, a], dim=1)
        q_online = self.qnet.calc_qval_batch(sa)
        assert q_online.numel() == minibatch_size
        q_online = makevec(q_online)
        with timebudget('q_target'):
            a1 = self.target_munet.calc_qval_batch(s1)
            s1a1 = torch.cat([s1, a1], dim=1)
            q_s1a1 = self.target_qnet.calc_qval_batch(s1a1)
            future_r = (1-f) * self.gamma * makevec(q_s1a1)
            q_target = r + future_r

        with timebudget('q_optimizer'):
            assert q_online.shape == q_target.shape  # Subtracting column vectors from row vectors leads to badness.
            loss = self.loss_func(q_online, q_target)
            if self.iter_cnt % self.show_loss_every == 0:
                print(f"Loss = {loss:.5f}")
            loss.backward()
            self.opt_q.step()

        # Update actor network
        warnings.warn("actor network update not implemented")

        # Move target networks
        self.target_nets_elastic_follow()
