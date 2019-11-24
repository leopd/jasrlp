import gym
import pytest

import rltrain

def test_pendulum():
    env = gym.make('Pendulum-v0')
    learner = rltrain.DDPG(env)
    learner.rollout(render=False)

def test_lander():
    env = gym.make('LunarLanderContinuous-v2')
    learner = rltrain.DDPG(env)
    learner.rollout(render=False)

