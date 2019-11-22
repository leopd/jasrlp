import gym
import pytest

import rltrain

def test_cartpole_random():
    env = gym.make('CartPole-v0')
    learner = rltrain.DQN(env)
    learner.rollout(render=False)

def test_mcar_random():
    env = gym.make('MountainCar-v0')
    learner = rltrain.DQN(env)
    learner.rollout(render=False)

