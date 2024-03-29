import gym
import numpy as np
import pytest

import rltrain
import helpers

def test_cartpole_random():
    env = gym.make('CartPole-v0')
    learner = rltrain.DQN(env)
    learner.rollout(render=False)

def test_mcar_random():
    env = gym.make('MountainCar-v0')
    learner = rltrain.DQN(env)
    learner.rollout(render=False)

def test_qval_batch():
    env = gym.make('MountainCar-v0')
    qnet = rltrain.FCNet.for_discrete_action(env)
    N = 50
    obs = []
    # Generate some valid observations any which way
    env.reset()
    for _ in range(N):
        a = env.action_space.sample()
        s1, r, _, _ = env.step(a)
        obs.append(s1)

    qvals = qnet.calc_qval_batch(obs)
    assert len(qvals.shape) == 2
    assert qvals.shape[0] == N
    assert qvals.shape[1] == env.action_space.n

