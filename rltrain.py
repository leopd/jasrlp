from collections import deque
import numpy as np
import random
import time
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import warnings

from typing import List, Union, NamedTuple, Tuple

class Transition(NamedTuple):
    state: list
    action: int
    state1: list
    reward: float


class ReplayBuffer():
    """Super stupid-simple Replay buffer.  
    Would be much better to store dense tensors, and use tensor indexing to sample.
    Further optimizaitons would be to store that in GPU if it fits, and to encode it for downstream tasks like Q-network.
    """

    def __init__(self, size:int=1000000):
        self._buffer = deque(maxlen=size)

    def append(self, transition:Transition):
        for val in transition:
            if val is None:
                # Something is None - skip
                return
        self._buffer.append(transition)

    def sample(self, num:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns a tuple of (s, a, s1, r) where each is a minibatch-sized tensor.
        """
        if num > len(self._buffer):
            return None
        transition_list = random.sample(self._buffer, num)
        # Convert these to tensors.
        out = []
        example = transition_list[0]
        for i in range(len(example)):
            tensor = torch.Tensor([t[i] for t in transition_list])
            #if len(tensor.shape) == 1:
                #tensor = tensor.unsqueeze(0)
            out.append(tensor)
        return out


class RandomLearner():

    def __init__(self, env):
        self.init_env(env)
        self.env = env
        self.action_space = env.action_space
        self._replay = ReplayBuffer()

    def init_env(self, env):
        self.env = env
        self.obs_space = env.observation_space
        assert self.obs_space.__class__.__name__ == "Box", "Only Box observation spaces supported"
        self.act_space = env.action_space
        assert self.act_space.__class__.__name__ == "Discrete", "Only Discrete action spaces supported"
        self.obs_dim = np.prod(self.obs_space.shape)
        self.act_dim = self.act_space.n

    def get_random_action(self):
        return self.action_space.sample()

    def get_action(self, obs):
        return self.get_random_action()

    def record_transition(self, observation, action, observation_next, reward):
        t = Transition(observation, action, observation_next, reward)
        self._replay.append(t)

    def rollout(self, max_iter:int=1000, render:bool=False) -> Tuple[int, float]:
        """Returns number of iterations and total reward
        """
        self.env.reset()
        obs_last = None
        total_reward = 0
        start_time = time.time()
        for cnt in range(max_iter):
            if render:
                self.env.render()
            action = self.get_action(obs_last)
            obs, reward, is_done, info = self.env.step(action)
            self.record_transition(obs_last, action, obs, reward)
            total_reward += reward
            obs_last = obs
            if is_done:
                break
            self.do_learning()
        elapsed = time.time() - start_time
        fps = cnt / elapsed
        print(f"Episode reward: {total_reward}. {cnt} frames at {fps:.1f}fps")
        self.env.close()
        return cnt, total_reward

    def do_learning(self):
        # Random learner has nothing to learn
        pass


def one_hot(which:int, dims:int) -> Tensor:
    out = torch.zeros(dims)
    if isinstance(which, Tensor):
        out[which.long()] = 1
    else:
        out[which] = 1
    return out


class QNetBase(nn.Module):
    pass


class FCNet(QNetBase):

    def __init__(self, input_dim:int, output_classes:int, hidden_dims:List[int], act_dim:int):
        super().__init__()
        self.act_dim = act_dim
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
        # No final activation since we're regressing
        return act

    def calc_qval(self, observation, action):
        """Calculates a single q-value
        """
        o_tensor = Tensor(observation)
        a_tensor = one_hot(action, self.act_dim)
        qin = torch.cat([o_tensor, a_tensor])
        qin = qin.unsqueeze(0)  # add minibatch dimension
        out = self.forward(qin)
        return out

    def calc_qval_batch(self, observations, actions):
        """Calculates a single q-value
        """
        o_tensor = Tensor(observations)
        a_tensor = torch.stack([one_hot(a, self.act_dim) for a in actions])
        assert o_tensor.shape[0] == a_tensor.shape[0]
        qin = torch.cat([o_tensor, a_tensor], dim=1)
        out = self.forward(qin)
        return out

    @classmethod
    def for_env(cls, env) -> "FCNet":
        obs_space = env.observation_space
        assert obs_space.__class__.__name__ == "Box", "Only Box observation spaces supported"
        act_space = env.action_space
        assert act_space.__class__.__name__ == "Discrete", "Only Discrete action spaces supported"
        obs_dim = np.prod(obs_space.shape)
        act_dim = act_space.n
        in_dim = obs_dim + act_dim  # Q network takes s,a as input
        out_dim = 1  # Q network is regression to a scalar
        print(f"Creating FCNet with {in_dim}->{out_dim} dims for {obs_dim} observations and {act_dim} actions")
        qnet = FCNet(in_dim, out_dim, [16,16], act_dim)
        return qnet


class DQN(RandomLearner):
    """We expect whatever code is using this thing to manually set the eps-greedy schedule explicitly.
    """

    def __init__(self, env, eps:float=0.5, gamma:float=0.99):
        super().__init__(env)
        self.qnet = FCNet.for_env(env)
        self.eps = eps
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters())
        self.iter_cnt = 0

    def get_action(self, obs):
        if (obs is None) or (torch.rand(1).item() < self.eps):
            a = self.get_random_action()
            #print(f"random: {a}")
            return a
        else:
            a = self.get_greedy_action(obs)
            #print(f"greedy: {a}")
            return a

    def all_actions(self):
        return range(self.env.action_space.n)

    def get_greedy_action(self, obs, use_target_network:bool=False):
        if use_target_network:
            warnings.warn("TODO: target network")
        action_scores = [self.qnet.calc_qval(obs,a) for a in self.all_actions()]
        action = np.argmax(action_scores)
        return action

    def do_learning(self):
        self.iter_cnt += 1
        minibatch_size = 16  # HYPERPARAMETER
        batch = self._replay.sample(minibatch_size)
        if batch is None:
            # not enough data yet.
            return

        # we have a minibatch.  Let's do this.
        self.opt.zero_grad()
        s, a, s1, r = batch
        q_online = self.qnet.calc_qval_batch(s, a)
        assert len(q_online) == minibatch_size
        q_s1_amax = []
        for n in range(minibatch_size): #TODO: vectorize this...
            a1max = self.get_greedy_action(s1[n], use_target_network=True)
            q = self.qnet.calc_qval(s1[n,:], a1max)
            q_s1_amax.append(q)
        q_s1_amax = torch.Tensor(q_s1_amax)
        q_target = r + self.gamma * q_s1_amax

        loss = ((q_online - q_target)**2).mean()
        if self.iter_cnt % 20 == 0:
            print(f"Loss = {loss:.5f}")
        loss.backward()
        self.opt.step()

