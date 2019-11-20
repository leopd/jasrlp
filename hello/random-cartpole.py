#!/usr/bin/env python

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, r, is_done, info = env.step(action)
    print(obs)
    if is_done:
        break
env.close()

