from collections import deque
import random
import torch
import torch.nn as nn

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
        



