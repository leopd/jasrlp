import matplotlib.pyplot as plt
import numpy as np
import torch

fix = lambda t: t.cpu().detach().numpy()

def rollout_score_dist(learner, eps:float, n:int=100, hist:bool=True) -> [list, list]:
    learner.eps = eps
    frames = []
    rewards = []
    for i in range(100):
        f, r = learner.rollout(render=False)
        frames.append(f)
        rewards.append(r)
    if hist:
        plt.hist(rewards, bins=20)
        plt.show()
    print(f"Reward mean,std = {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")
    return rewards, frames

def sample_qvals(learner, N:int=3000):
    ob = (torch.rand(N,4) - 0.5) * 3.14
    ob[:,0:2] *= 0
    qb = learner.qnet.calc_qval_batch(ob)
    ob = fix(ob)
    qb = fix(qb)
    return ob, qb

def plot_a_general(ob,c,title):
    plt.scatter(x=ob[:,2],y=ob[:,3], c=c, alpha=0.5)
    plt.xlabel("angle"); plt.ylabel("angle speed")
    plt.title(title)
    plt.colorbar()

def plot_a0a1a10(ob, qb):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plot_a_general(ob, qb[:,0], "Action 0")
    plt.subplot(1,3,2)
    plot_a_general(ob, qb[:,1], "Action 1")
    plt.subplot(1,3,3)
    plot_a_general(ob, qb[:,1] - qb[:,0], "A1-A0")


def plot_qval_act_diff(learner):
    ob,qb = sample_qvals()
    plot_a_general(ob, qb[:,1] - qb[:,0], "A1-A0")
    plt.show()

def plot_q(learner):
    ob,qb = sample_qvals(learner)
    plot_a0a1a10(ob, qb)
    plt.show()


