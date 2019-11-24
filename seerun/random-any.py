#!/usr/bin/env python

import argparse
from collections import defaultdict
import gym
import time


def parser():
    parser = parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=str, default='list')
    return parser
    return parser.parse_args

def list_environments():
    reg = gym.envs.registration.registry
    names = sorted(list(reg.env_specs.keys()))
    by_category = defaultdict(list)
    for name in names:
        entry_point = reg.env_specs[name].entry_point
        category, _ = entry_point.split(':',1)
        by_category[category].append(name)
    for category, names in by_category.items():
        print(f"## {category}")
        for name in names:
            print(f"  {name}")

def main(args):
    if args.env == 'list':
        list_environments()
        return
        
    env = gym.make(args.env)
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, r, is_done, info = env.step(action)
        print(f"R:{r} O:{obs}")
        if is_done:
            print("Done")
            time.sleep(2)
            break
    env.close()


if __name__ == "__main__":
    args = parser().parse_args()
    main(args)
