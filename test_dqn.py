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


def test_cartpole_trains():
    env = gym.make('CartPole-v0')
    learner = rltrain.DQN(env, gamma=0.8, lr=0.01, net_args={'hidden_dims': [16]})
    learner.minimum_transitions_in_replay = 500

    # random warmup
    learner.eps = 1
    for i in range(100):
        learner.rollout()
    # train on epsilon schedule
    for d in range(5):
        eps = 1.0 - d/4.5
        _ = helpers.rollout_score_dist(learner, eps, n=20, hist=False)
    rewards, _ = helpers.rollout_score_dist(learner, 0, n=20, hist=False)
    mean_reward = np.mean(rewards)
    assert mean_reward > 50
