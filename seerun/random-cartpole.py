#!/usr/bin/env python

import gym
import time

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    obs, r, is_done, info = env.step(action)
    print(obs)
    time.sleep(0.15)
    if is_done:
        break
env.close()

