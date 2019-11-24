import gym
import numpy as np
import pytest

import rltrain
import helpers

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


def test_pendulum_trains():
    env = gym.make('Pendulum-v0')
    learner = rltrain.DDPG(env, gamma=0.95, lr=0.01, net_args={'hidden_dims': [16]})
    learner.minimum_transitions_in_replay = 1000
    learner.minibatch_size = 256

    # random warmup
    learner.eps = 1
    for i in range(30):
        learner.rollout()
    # train on epsilon schedule
    for d in range(5):
        eps = 0.95 - d/4.5
        _ = helpers.rollout_score_dist(learner, eps, n=10, hist=False)
    rewards, _ = helpers.rollout_score_dist(learner, 0, n=20, hist=False)
    mean_reward = np.mean(rewards)
    assert mean_reward > -500

