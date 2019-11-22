from collections import deque
import numpy as np
import random
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Union, NamedTuple

class Transition(NamedTuple):
    state: list
    action: int
    state1: list
    reward: float


class ReplayBuffer():

    def __init__(self, size:int=1000000):
        self._buffer = deque(maxlen=size)

    def append(self, transition:Transition):
        self._buffer.append(transition)

    def sample(self, num:int) -> List[Transition]:
        return random.sample(self._buffer, num)


class RandomLearner():

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self._replay = ReplayBuffer()

    def get_random_action(self):
        return self.action_space.sample()

    def get_action(self, obs):
        return self.get_random_action()

    def record_transition(self, observation, action, observation_next, reward):
        t = Transition(observation, action, observation_next, reward)
        self._replay.append(t)

    def rollout(self, max_iter:int=1000, render:bool=False):
        self.env.reset()
        obs_last = None
        start_time = time.time()
        for cnt in range(max_iter):
            if render:
                self.env.render()
            action = self.get_action(obs_last)
            obs, reward, is_done, info = self.env.step(action)
            self.record_transition(obs_last, action, obs, reward)
            obs_last = obs
            if is_done:
                break
        elapsed = time.time() - start_time
        fps = cnt / elapsed
        print(f"Completed {cnt} frames at {fps:.1f}fps")
        self.env.close()


def one_hot(which:int, dims:int) -> Tensor:
    out = torch.zeros(dims)
    out[which] = 1
    return out

class QNetBase():
    pass



class FCNet(QNetBase):

    def __init__(self, input_dim:int, output_classes:int, hidden_dims:List[int]):
        layer_dims = [input_dim] + hidden_dims + [output_classes]
        layers = []
        for i in range(len(layer_dims)-1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers = layers[:-1]  # remove last ReLU
        self.layers = nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        act = self.layers(x)
        return act



class DQN(RandomLearner):
    """We expect whatever code is using this thing to manually set the eps-greedy schedule explicitly.
    """

    def __init__(self, env, eps:float=0.5):
        super().__init__(env)
        self.qnet = self.make_qnet_for_env(env)
        self.eps = eps

    def make_qnet_for_env(self, env):
        obs_space = env.observation_space
        assert obs_space.__class__.__name__ == "Box", "Only Box observation spaces supported"
        act_space = env.action_space
        assert act_space.__class__.__name__ == "Discrete", "Only Discrete action spaces supported"
        self.obs_dim = np.prod(obs_space.shape)
        self.act_dim = act_space.n
        in_dim = self.obs_dim + self.act_dim  # Q network takes s,a as input
        out_dim = 1  # Q network is regression to a scalar
        print(f"Creating FCNet with {in_dim}->{out_dim} dims for {self.obs_dim} observations and {self.act_dim} actions")
        qnet = FCNet(in_dim, out_dim, [16,16])
        return qnet
        

    def calc_qval(self, observation, action):
        o_tensor = Tensor(observation)
        a_tensor = one_hot(action, self.act_dim)
        qin = torch.cat([o_tensor, a_tensor])
        qin = qin.unsqueeze(0)  # add minibatch dimension
        out = self.qnet.forward(qin)
        return out

    def get_action(self, obs):
        if (obs is None) or (torch.rand(1).item() < self.eps):
            a = self.get_random_action()
            print(f"random: {a}")
            return a
        else:
            a = self.get_greedy_action(obs)
            print(f"greedy: {a}")
            return a

    def all_actions(self):
        return range(self.env.action_space.n)

    def get_greedy_action(self, obs):
        action_scores = [self.calc_qval(obs,a) for a in self.all_actions()]
        action = np.argmax(action_scores)
        return action

