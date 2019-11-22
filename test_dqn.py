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


def test_qval_batch():
    env = gym.make('MountainCar-v0')
    qnet = rltrain.FCNet.for_env(env)
    N = 50
    obs = []
    acts = []
    # Generate some valid observations any which way
    env.reset()
    for _ in range(N):
        a = env.action_space.sample()
        s1, r, _, _ = env.step(a)
        obs.append(s1)
        acts.append(a)

    qvals = qnet.calc_qval_batch(obs, acts)
    assert len(qvals) == N

