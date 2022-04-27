"""
Comment: experiment codes for submission of Environment Design for Biased Decision Makers
----------
Parameters
----------
Returns
-------
"""

import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import time
import math
import sys
from MDPenvironment import *
from Agent import *
from Principal import *
path = 'results/'

def baseline_experiment():
    """
    Comment: baseline experiment
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    Ntestcase = 1000
    data = np.load("seeds.npz", allow_pickle=True)
    Ntestseed = data['s'][:Ntestcase]
    mdp = MDPbasic()
    mdp.simtestcase1()
    outp = 'baseline1000'
    discount_gamma_list = np.arange(0.9,0.05,-0.1)
    bounded_horizon_list = np.arange(1,11)
    hyperbolic_k_list = np.array([0.1,np.sqrt(10)/10,1.0,np.sqrt(10),10])
    ans = np.zeros([Ntestcase, 4, 11])
    # opt, mypoic; discounted; bounded; hyperbolic
    tstart = time.time()
    for iter in range(Ntestcase):
        mdp.simtestcase1_rsa(seed=Ntestseed[iter])
        mdp.simtestcase1(seed=Ntestseed[iter])
        optai = optimal_ai_discounted(mdp, gamma = 0.99)
        ans[iter,0,0] = opt_evaluator(mdp, optai[0], tau=-1)
        myopic = myopic_human(mdp)
        # tau here should not be -1 and smaller than T
        ans[iter, 0, 1] = opt_evaluator(mdp, myopic[0], tau= 5 )
        for i in range(np.size(discount_gamma_list)):
            bounded = optimal_ai_discounted(mdp, gamma=discount_gamma_list[i])
            ans[iter, 1, i] = opt_evaluator(mdp, bounded[0], tau=5)
        for i in range(np.size(bounded_horizon_list)):
            bounded = bounded_rational_human_discounted(mdp, tau=bounded_horizon_list[i], gamma=0.99)
            ans[iter, 2, i] = opt_evaluator(mdp, bounded[0], tau=bounded_horizon_list[i])
        for i in range(np.size(hyperbolic_k_list)):
            hyper = hyperbolic_human(mdp, k=hyperbolic_k_list[i])
            ans[iter, 3, i] = opt_evaluator(mdp, hyper[0], tau=5)
        if np.mod(iter,10) == 9:
            print("iter=",iter, " ,time=", time.time()-tstart)
            # np.savez(outp+'.npz', ans=ans, paras=Ntestseed)
    print("finish all", iter, " ,time=", time.time() - tstart)
    np.savez(path+outp+'.npz', ans=ans, paras=Ntestseed)
    return -1


def action_nudge_experiment():
    """
    Comment: action nudge experiment
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    Ntestcase = 1000
    data = np.load("seeds.npz", allow_pickle=True)
    Ntestseed = data['s'][:Ntestcase]
    mdp = MDPbasic()
    mdp.simtestcase1()
    outp = 'action_nudge1000'

    # select special bias para
    discount_gamma = 0.4
    bounded_horizon = 3
    hyperbolic_k = 1.0
    base_reward = 50 # expected optimal average reward for 1000 test cases
    nudge_budget = np.array([0.01,0.02,0.05,0.10,0.20,0.50])
    # nudge_budget = np.array([0.01])
    nudge_budget = nudge_budget * base_reward
    def nudge_cost(Q, T, tau = -1):
        cost = np.zeros([T, Q.shape[1], Q.shape[2]])
        for t in range(T):
            for s in range (Q.shape[1]):
                cost[t,s,:] = np.max(Q[t,s,:]) - Q[t,s,:]
        return cost
    def sub_nudge_model(Q):
        r = np.zeros(np.size(nudge_budget))
        c = np.zeros(np.size(nudge_budget))
        for k in range(np.size(nudge_budget)):
            # nudge = nudge_ai_budget(mdp, Q, cost_budget=nudge_budget[k])
            nudge = nudge_ai_budget_rsa(mdp, nudge_cost(Q,20), cost_budget=nudge_budget[k])
            a = opt_evaluator(mdp, nudge[0], tau=-1, cost=nudge[1])
            r[k] = a[0]
            c[k] = a[1]
        return np.hstack((r,c))
    ans = np.zeros([Ntestcase, 4, np.size(nudge_budget)*2+1]) # bias + rewrad + cost
    # ans_cost = np.zeros([Ntestcase, 4, np.size(nudge_budget)+1]) # actually cost
    tstart = time.time()
    for iter in range(Ntestcase):
        mdp.simtestcase1_rsa(seed=Ntestseed[iter])
        mdp.simtestcase1(seed=Ntestseed[iter])
        myopic = myopic_human(mdp)
        ans[iter, 0, 0] = opt_evaluator(mdp, myopic[0], tau=5)
        ans[iter, 0,1:] = sub_nudge_model(myopic[1])
        discount = optimal_ai_discounted(mdp, gamma=discount_gamma)
        ans[iter, 1, 0] = opt_evaluator(mdp, discount[0], tau=5)
        ans[iter, 1, 1:] = sub_nudge_model(discount[1])
        bounded = bounded_rational_human_discounted(mdp, tau=bounded_horizon, gamma=0.99)
        ans[iter, 2, 0] = opt_evaluator(mdp, bounded[0], tau=bounded_horizon)
        ans[iter, 2, 1:] = sub_nudge_model(bounded[1])
        hyper = hyperbolic_human(mdp, k=hyperbolic_k)
        ans[iter, 3, 0] = opt_evaluator(mdp, hyper[0], tau=5)
        ans[iter, 3, 1:] = sub_nudge_model(hyper[1])
        if np.mod(iter,1) == 0:
            print("iter=",iter, " ,time=", time.time()-tstart)
            # np.savez(outp+'.npz', ans=ans, paras=Ntestseed)
    print("finish all", iter, " ,time=", time.time() - tstart)
    np.savez(path + outp+'.npz', ans=ans, paras=Ntestseed)
    return -1

def reward_function_modificaiton():
    """
    Comment: reward function modification
    ----------
    Parameters
    ----------
    Returns
    -------
    """ 
    Ntestcase = 1000
    data = np.load("seeds.npz", allow_pickle=True)
    Ntestseed = data['s'][:Ntestcase]
    mdp = MDPbasic()
    mdp.simtestcase1()
    outp = 'reward_modify1000'
    # select special bias para
    discount_gamma = 0.4
    bounded_horizon = 3
    hyperbolic_k = 1.0
    base_reward = 120 # expected optimal average reward for 1000 test cases
    poison_budget = np.array([0.05, 0.10, 0.20, 0.50,1.0,2.0])
    ans = np.zeros([Ntestcase, 4, np.size(poison_budget) * 2 + 1])  # bias + rewrad + cost

    def sub_poison_model(tp, g, tau=1):
        # poison = poison_ai_budget_gradient(mdp, human_t=tau, lr=0.01, gamma=g, Niter=10000, beta=1.0)
        poison = poison_ai_to_policy(mdp, g[:tau], tp[:tau, :], consistent=False, testmode=False)
        poisoned_mdp = MDPbasic()
        info = mdp.print_all_info()
        R = info[3]
        P = info[4]
        poisoned_mdp.read_info([R, P])
        poisoned_mdp.poison_reward(poison)
        return poisoned_mdp, np.sum(poison)
    tstart = time.time()
    for iter in range(Ntestcase):
        mdp.simtestcase1_rsa(seed=Ntestseed[iter])
        mdp.simtestcase1(seed=Ntestseed[iter])
        optai = optimal_ai_discounted(mdp, gamma = 0.99)
        target_policy = optai[0][:,:].copy()
        gamma = np.logspace(0, 100, base=0.99, num=100, endpoint=False)
        myopic = myopic_human(mdp)
        ans[iter, 0, 0] = opt_evaluator(mdp, myopic[0], tau=1)
        pois = sub_poison_model(target_policy, gamma, tau=1)
        myopic2 = myopic_human(pois[0])
        ans[iter, 0, 1] = opt_evaluator(mdp, myopic2[0], tau=1)
        ans[iter, 0, 2] = pois[1]
        discount = optimal_ai_discounted(mdp, gamma=discount_gamma)
        ans[iter, 1, 0] = opt_evaluator(mdp, discount[0], tau=1)
        gamma = np.logspace(0, 100, base=discount_gamma, num=100, endpoint=False)
        pois = sub_poison_model(target_policy, gamma, tau=10)
        discount2 = optimal_ai_discounted(pois[0], gamma=discount_gamma)
        ans[iter, 1, 1] = opt_evaluator(mdp, discount2[0], tau=1)
        ans[iter, 1, 2] = pois[1]
        bounded = bounded_rational_human_discounted(mdp, tau=bounded_horizon, gamma=0.99)
        ans[iter, 2, 0] = opt_evaluator(mdp, bounded[0], tau=bounded_horizon)
        gamma = np.logspace(0, 100, base=0.99, num=100, endpoint=False)
        pois = sub_poison_model(target_policy, gamma, tau=bounded_horizon)
        bounded2 = bounded_rational_human_discounted(pois[0], tau=bounded_horizon, gamma=0.99)
        ans[iter, 2, 1] = opt_evaluator(mdp, bounded2[0], tau=bounded_horizon)
        ans[iter, 2, 2] = pois[1]
        hyper = hyperbolic_human(mdp, k=hyperbolic_k)
        ans[iter, 3, 0] = opt_evaluator(mdp, hyper[0], tau=10)
        for i in range (100):
            gamma[i] = 1.0 / (1+ hyperbolic_k*i)
        pois = sub_poison_model(target_policy, gamma, tau=10)
        hyper2 = hyperbolic_human(pois[0], k=hyperbolic_k)
        ans[iter, 3, 1] = opt_evaluator(mdp, hyper2[0], tau=10)
        ans[iter, 3, 2] = pois[1]
        if np.mod(iter, 1) == 0:
            print("iter=", iter, " ,time=", time.time() - tstart)
            # np.savez(outp + '.npz', ans=ans, paras=Ntestseed)
    print("finish all", iter, " ,time=", time.time() - tstart)
    np.savez(path + outp + '.npz', ans=ans, paras=Ntestseed)
    return -1


if __name__ == '__main__':
    print("experiment codes for submission of Environment Design for Biased Decision Makers")
    print('---------------------------------------------------------------')
    argvs = sys.argv
    if argvs[1] == 'baseline':
        print("baseline test for bias agent")
        baseline_experiment()
    elif argvs[1] == 'action_nudge':
        print("action nudge test")
        action_nudge_experiment()
    elif argvs[1] == 'reward_modify':
        print("reward function modity test")
        reward_function_modificaiton()
    else:
        print("please check input argvs: baseline; action_nudge")

