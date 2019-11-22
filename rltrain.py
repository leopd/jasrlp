from collections import deque
import random
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

    def get_action(self):
        return self.get_random_action()

    def record_transition(self, observation, action, observation_next, reward):
        t = Transition(observation, action, observation_next, reward)
        self._replay.append(t)

    def rollout(self, max_iter:int=1000, render:bool=False):
        self.env.reset()
        obs_last = None
        for _ in range(max_iter):
            if render:
                self.env.render()
            action = self.get_action()
            obs, reward, is_done, info = self.env.step(action)
            self.record_transition(obs_last, action, obs, reward)
            obs_last = obs
            if is_done:
                break
        self.env.close()


class QNetBase():


    def qval(self, observation, action):
        o_tensor = Tensor(observation)
        a_tensor = Tensor(action)
        qin = torch.cat([o_tensor, a_tensor])
        qval = self.forward(qin)
        return qval


class FCNet(QNetBase):

    def __init__(self, input_dim:int, output_classes:int, hidden_dims:List[int]):
        layer_dims = [input_dim] + hidden_dims + [output_classes]
        layers = []
        for i in range(len(layer_dims)-1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU)

        layers = layers[:-1]  # remove last ReLU
        self.layers = nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        act = self.layers(x)
        out = F.log_softmaxs(a)
        return out


def make_qnet_for_env(env):
    obs_space = env.observation_space
    assert obs_space.__class__.__name__ == "Box", "Only Box observation spaces supported"
    act_space = env.observation_space
    assert act_space.__class__.__name__ == "Discrete", "Only Discrete action spaces supported"
    obs_dim = np.prod(obs_space.shape)
    act_dim = act_space.n
    in_dim = obs_dim + act_dim  # Q network takes s,a as input
    out_dim = 1  # Q network is regression to a scalar
    qnet = FCNet(in_dim, out_dim, [16,16])
    return qnet
        

class DQN(RandomLearner):
    """We expect whatever code is using this thing to manually set the eps-greedy schedule explicitly.
    """

    def __init__(self, env, eps:float=0.5):
        super().__init__(env)
        self.qnet = make_qnet_for_env(env)
        self.eps = eps

    def get_action(self, obs):
        if torch.rand(1).item < self.eps:
            return self.get_random_action()
        else:
            return self.get_greedy_action(obs)

    def get_greedy_action(self, obs):
        action_scores = [self.qnet.qval(obs,a) for a in self.all_actions()]
        action = np.argmax(action_scores)
        return action

