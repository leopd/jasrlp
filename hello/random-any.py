#!/usr/bin/env python

import argparse
import gym
import time


def parser():
    parser = parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=str, default='list')
    return parser
    return parser.parse_args

def list_environments():
    reg = gym.envs.registration.registry
    names = reg.env_specs.keys()
    for n in names:
        print(n)

def main(args):
    if args.env == 'list':
        list_environments()
        return
        
    env = gym.make(args.env)
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, r, is_done, info = env.step(action)
        print(obs)
        if is_done:
            print("Done")
            time.sleep(2)
            break
    env.close()


if __name__ == "__main__":
    args = parser().parse_args()
    main(args)
