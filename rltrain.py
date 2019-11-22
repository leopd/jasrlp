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



