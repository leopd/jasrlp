import copy
from collections import deque
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

class Transition(NamedTuple):
    state: list
    action: int
    state1: list
    reward: float
    is_final: int  # 1 for True

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

    @timebudget
    def sample(self, num:int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Returns a tuple of (s, a, s1, r, f) where each is a minibatch-sized tensor.
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

    def record_transition(self, observation, action, observation_next, reward, is_final):
        is_final = 1 if is_final else 0
        t = Transition(observation, action, observation_next, reward, is_final)
        self._replay.append(t)

    @timebudget
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
            self.record_transition(obs_last, action, obs, reward, is_done)
            total_reward += reward
            obs_last = obs
            if is_done:
                break
            self.do_learning()
        elapsed = time.time() - start_time
        fps = cnt / elapsed
        #print(f"Episode reward: {total_reward}. {cnt} frames at {fps:.1f}fps")
        self.env.close()
        return cnt, total_reward

    def do_learning(self):
        # Random learner has nothing to learn
        pass


class QNetBase(nn.Module):
    pass


#TODO: refactor all this so that the net outputs a Q value for each class simultaneously.
# More efficient, and easier for the net to distinguish appropriate actions.
class FCNet(QNetBase):

    def __init__(self, input_dim:int, output_classes:int, hidden_dims:List[int]):
        super().__init__()
        layer_dims = [input_dim] + hidden_dims + [output_classes]
        layers = []
        for i in range(len(layer_dims)-1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            #layers.append(nn.ReLU())
            layers.append(nn.Tanh())
        layers = layers[:-1]  # remove last ReLU
        self.layers = nn.Sequential(*layers)

    def forward(self, x:Tensor) -> Tensor:
        act = self.layers(x)
        # No final activation since we're regressing
        return act

    def calc_qval_batch(self, observations):
        """Calculates a minibatch of q-values
        """
        o_tensor = Tensor(observations)
        qin = o_tensor
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
        print(f"Creating FCNet with {obs_dim}->{act_dim} dims for {obs_dim} observations and {act_dim} actions")
        qnet = FCNet(obs_dim, act_dim, [128])
        return qnet


class DQN(RandomLearner):
    """We expect whatever code is using this thing to manually set the eps-greedy schedule explicitly.
    """

    def __init__(self, env, eps:float=0.5, gamma:float=0.99):
        super().__init__(env)
        self.qnet = FCNet.for_env(env)
        self.copy_to_target()
        #self.loss_func = nn.SmoothL1Loss()
        self.loss_func = lambda x,y: ((x-y)**2).mean()
        self.eps = eps
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters())
        #self.opt = torch.optim.SGD(params=self.qnet.parameters(), lr=0.01)
        self.iter_cnt = 0
        self.minibatch_size = 32  # HYPERPARAMETER

    def copy_to_target(self):
        """Copies the online Q network to the target network
        """
        self.target_qnet = copy.deepcopy(self.qnet)

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
            net = self.target_qnet
        else:
            net = self.qnet
        action_scores = self.qnet.calc_qval_batch([obs])
        action = torch.argmax(action_scores[0,:]).item()
        return action

    @timebudget
    def do_learning(self):
        minibatch_size = self.minibatch_size
        batch = self._replay.sample(minibatch_size)
        if batch is None:
            # not enough data yet.
            return

        # we have a minibatch.  Let's do this.
        self.opt.zero_grad()
        s, a, s1, r, f = batch
        q_online_all_a = self.qnet.calc_qval_batch(s)
        q_online = q_online_all_a.gather(1, a.long().view(-1,1))  # Magic: https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497
        assert q_online.numel() == minibatch_size
        #q_online = q_online.view(-1)  #... for some reason, this causes everything to explode with huber loss, but makes warning go away
        with timebudget('q_target'):
            q_s1 = self.target_qnet.calc_qval_batch(s1)
            q_s1_amax = q_s1.max(dim=1)[0]
            future_r = (1-f) * self.gamma * q_s1_amax
            q_target = r + future_r

        with timebudget('optimizer'):
            loss = self.loss_func(q_online, q_target)
            if self.iter_cnt % 200 == 0:
                print(f"Loss = {loss:.5f}")
            loss.backward()
            self.opt.step()

        self.iter_cnt += 1
        if self.iter_cnt % 1000 == 0:
            self.copy_to_target()

