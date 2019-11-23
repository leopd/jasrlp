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


class CartPoleViz():

    def __init__(self, learner):
        self.learner = learner
        self.obs_dim = learner.obs_dim
        self.obs_scale = 3.14 / 2

    def sample_qvals(self, N:int=3000):
        ob = (torch.rand(N, self.obs_dim) - 0.5) * self.obs_scale * 2
        ob[:,0:2] *= 0
        qb = self.learner.qnet.calc_qval_batch(ob)
        ob = fix(ob)
        qb = fix(qb)
        return ob, qb

    def plot_a_general(self, ob, c, title):
        plt.scatter(x=ob[:,2],y=ob[:,3], c=c, alpha=0.5)
        plt.xlabel("angle"); plt.ylabel("angle speed")
        plt.title(title)
        plt.colorbar()

    def plot_a0a1a10(self, ob, qb):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        self.plot_a_general(ob, qb[:,0], "Action 0")
        plt.subplot(1,3,2)
        self.plot_a_general(ob, qb[:,1], "Action 1")
        plt.subplot(1,3,3)
        self.plot_a_general(ob, qb[:,1] - qb[:,0], "A1-A0")


    def plot_qval_act_diff(self):
        ob,qb = sample_qvals()
        self.plot_a_general(ob, qb[:,1] - qb[:,0], "A1-A0")
        plt.show()

    def plot_q(self):
        ob,qb = self.sample_qvals()
        self.plot_a0a1a10(ob, qb)
        plt.show()

def MountainCarViz(CartPoleViz):

    def __init__(self, learner):
        super().__init__(learneR)
