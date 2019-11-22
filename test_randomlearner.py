import gym
import pytest

import rltrain

def test_cartpole_random():
    env = gym.make('CartPole-v0')
    learner = rltrain.RandomLearner(env)
    learner.rollout(render=False)

