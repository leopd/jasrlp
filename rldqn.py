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

    def __init__(self, size:int=(1000 * 100)):
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

    def __len__(self):
        return len(self._buffer)


class RandomLearner():

    def __init__(self, env, buffer_size:int=100000):
        self.init_env(env)
        self.env = env
        self.action_space = env.action_space
        self._replay = ReplayBuffer(size=buffer_size)
        self.reward_history = []
        self.rollout_max_iter = 1000  # HYPERPARAMETER

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
    def rollout(self, render:bool=False) -> Tuple[int, float]:
        """Runs an episode from reset to done. Returns number of iterations and total reward
        """
        obs_last = self.env.reset()
        total_reward = 0
        start_time = time.time()
        for cnt in range(self.rollout_max_iter):
            if render:
                self.env.render()
            with timebudget('rollout-forwards'):
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
        self.reward_history.append(total_reward)
        return cnt, total_reward

    def do_learning(self):
        # Random learner has nothing to learn
        pass


class QNetBase(nn.Module):
    pass


class FCNet(QNetBase):

    def __init__(self, input_dim:int, output_classes:int, hidden_dims:List[int], activation=nn.ReLU, final_activation:bool=False, output_scaling:float=1.0):
        super().__init__()
        layer_dims = [input_dim] + hidden_dims + [output_classes]
        layers = []
        for i in range(len(layer_dims)-1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation())
        if not final_activation:
            layers = layers[:-1]  # remove last activation
        self.layers = nn.Sequential(*layers)
        self.output_scaling = output_scaling

    def forward(self, x:Tensor) -> Tensor:
        act = self.layers(x) * self.output_scaling
        # No final activation since we're regressing
        return act

    def calc_qval_batch(self, observations):
        """Calculates a minibatch of q-values
        """
        warnings.warn("calc_qval_batch can usually be replaced with just forward()")
        o_tensor = Tensor(observations)
        qin = o_tensor
        out = self.forward(qin)
        return out

    def forward_cat(self, state:Tensor, action:Tensor):
        """When the network takes both state and actions as inputs, this fuses them together first
        """
        sa = torch.cat([state, action], dim=1)
        return self.forward(sa)

    @classmethod
    def for_discrete_action(cls, env, hidden_dims=[64,64], activation=nn.Tanh) -> "FCNet":
        obs_space = env.observation_space
        assert obs_space.__class__.__name__ == "Box", "Only Box observation spaces supported"
        act_space = env.action_space
        assert act_space.__class__.__name__ == "Discrete", "Only Discrete action spaces supported"
        obs_dim = np.prod(obs_space.shape)
        act_dim = act_space.n
        print(f"Creating FCNet with {obs_dim}->{act_dim} dims for {obs_dim} observations and {act_dim} actions")
        qnet = FCNet(obs_dim, act_dim, hidden_dims=hidden_dims, activation=activation)
        return qnet


class DQN(RandomLearner):
    """We expect whatever code is using this thing to manually set the eps-greedy schedule explicitly.
    """

    def __init__(self, env, eps:float=0.5, gamma:float=0.99, net_args:dict={}, lr:float=1e-4, buffer_size:int=100000):
        super().__init__(env, buffer_size)
        self.build_nets(env, net_args)
        self.loss_func = nn.SmoothL1Loss()  # huber loss
        #self.loss_func = lambda x,y: ((x-y)**2).mean()  # MSE
        self.eps = eps
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.iter_cnt = 0
        self.minibatch_size = 32  # HYPERPARAMETER
        self.show_loss_every = 1000 # HYPERPARAMETER
        self.minimum_transitions_in_replay = 10000 # HYPERPARAMETER
        self.copy_to_target_every = 1000 # HYPERPARAMETER

    def build_nets(self, env, net_args):
        self.qnet = FCNet.for_discrete_action(env, **net_args)
        self.copy_to_target()

    def copy_to_target(self):
        """Copies the online Q network to the target network
        """
        self.target_qnet = copy.deepcopy(self.qnet)
        self.target_qnet.eval()

    def get_action(self, obs):
        if (obs is None) or (torch.rand(1).item() < self.eps):
            a = self.get_random_action()
            #print(f"random: {a}")
            return a
        else:
            a = self.get_greedy_action(obs)
            #print(f"greedy: {a}")
            return a

    def get_greedy_action(self, obs):
        with torch.no_grad():
            self.qnet.eval()
            action_scores = self.qnet.calc_qval_batch([obs])
            action = torch.argmax(action_scores[0,:]).item()
            return action

    @timebudget
    def do_learning(self):
        if len(self._replay) < self.minimum_transitions_in_replay:
            return
        minibatch_size = self.minibatch_size
        batch = self._replay.sample(minibatch_size)
        assert batch is not None

        # DQN training algorithm
        self.opt.zero_grad()
        self.qnet.train()
        s, a, s1, r, f = batch
        q_online_all_a = self.qnet.calc_qval_batch(s)
        # Pick the appropriate "a" column from all the actions. 
        q_online = q_online_all_a.gather(1, a.long().view(-1,1))  # Magic: https://discuss.pytorch.org/t/select-specific-columns-of-each-row-in-a-torch-tensor/497
        assert q_online.numel() == minibatch_size
        q_online = q_online.view(-1)  # Must make this a single dim vector.
        with timebudget('q_target'):
            q_s1 = self.target_qnet.calc_qval_batch(s1)
            q_s1_amax = q_s1.max(dim=1)[0]
            future_r = (1-f) * self.gamma * q_s1_amax
            q_target = r + future_r

        with timebudget('optimizer'):
            assert q_online.shape == q_target.shape  # Subtracting column vectors from row vectors leads to badness.
            loss = self.loss_func(q_online, q_target)
            if self.iter_cnt % self.show_loss_every == 0:
                print(f"Loss = {loss:.5f}")
            loss.backward()
            self.opt.step()

        self.iter_cnt += 1
        if self.iter_cnt % self.copy_to_target_every == 0:
            self.copy_to_target()


