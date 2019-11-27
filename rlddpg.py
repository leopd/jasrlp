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


class DampedRandomWalk():
    """Also known as Ornstein-Uhlenbeck process, which is super un-helpful.
    """

    def __init__(self, dims:int, damping:float, sigma:float):
        self.dims = dims
        assert damping >= 0
        self.damping = damping
        self.sigma = sigma
        self.x = self._rand()

    def _rand(self) -> np.ndarray:
        return np.random.randn(self.dims) * self.sigma

    def next(self) -> np.ndarray:
        dx = (-self.x) * self.damping + self._rand()
        self.x += dx
        return np.copy(self.x)



class DDPG(DQN):

    def __init__(self, env, eps:float=0.5, gamma:float=0.99, net_args:dict={}, lr=1e-4, buffer_size:int=100000):
        super().__init__(env, eps, gamma, net_args, buffer_size)
        del self.opt
        self.opt_q = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.opt_mu = torch.optim.Adam(params=self.munet.parameters(), lr=lr)
        self.init_noise()

    def init_env(self, env):
        self.env = env
        self.obs_space = env.observation_space
        assert self.obs_space.__class__.__name__ == "Box", "Only Box observation spaces supported"
        self.act_space = env.action_space
        assert self.act_space.__class__.__name__ == "Box", "Only Box action spaces supported"
        self.obs_dim = np.prod(self.obs_space.shape)
        self.act_dim = np.prod(self.act_space.shape)
        self.tau = 0.001  # HYPERPARAMETER
        self.noise_damp = 0.15  # HYPERPARAMETER

    def init_noise(self):
        out_scale = box_scale(self.env.action_space)
        self.noise = DampedRandomWalk(self.act_dim, self.noise_damp, out_scale)

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


    def get_action(self, obs):
        if (obs is None) or (self.eps>=1):
            return self.get_random_action()

        a = self.get_greedy_action(obs)
        noise = self.noise.next()
        noisy_action = a + self.eps * noise
        # Note I'm not bothering to clip the action to the space - I'm trusting the environment will do this
        return noisy_action


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
        self.iter_cnt += 1
        minibatch_size = self.minibatch_size
        batch = self._replay.sample(minibatch_size)
        assert batch is not None

        # Implement DDPG learning algorithm.
        s, a, s1, r, f = batch

        makevec = lambda t: t.view(-1)

        # First update online Q network
        with timebudget('critic_update'):
            self.opt_q.zero_grad()
            self.qnet.train()
            q_online = self.qnet.forward_cat(s,a)
            assert q_online.numel() == minibatch_size
            q_online = makevec(q_online)
            a1 = self.target_munet(s1)
            q_s1a1 = self.target_qnet.forward_cat(s1,a1)
            future_r = (1-f) * self.gamma * makevec(q_s1a1)
            q_target = r + future_r
            assert q_online.shape == q_target.shape  # Subtracting column vectors from row vectors leads to badness.
            critic_loss = self.loss_func(q_online, q_target)
            critic_loss.backward()
            self.opt_q.step()

        # Update actor network
        with timebudget('actor_update'):
            self.opt_mu.zero_grad()
            self.munet.train()
            # Calculate expected return over the sampled transitions for the online actor & critic
            J = self.qnet.forward_cat(s, self.munet(s))
            mu_loss = (-J).mean()
            mu_loss.backward()
            self.opt_mu.step()

        if self.iter_cnt % self.show_loss_every == 0:
            print(f"Critic Loss = {critic_loss:.5f}.  Mu loss = {mu_loss:.5f}")

        # Move target networks
        with timebudget('move_targets'):
            self.target_nets_elastic_follow()

