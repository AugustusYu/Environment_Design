"""
Comment: principal model
----------
Parameters
----------
Returns
-------
"""

import numpy as np
import pulp
import time
import itertools
import jax.numpy as jnp
from jax import grad, jit, vmap
from MDPenvironment import MDPbasic
from Agent import general_human_backward


def optimal_ai_undiscounted(mdp):
    """
    the optimal ai with all information
    object is to max undiscounted AI
    Parameters
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    ans = backward_iter_Q(info)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q

def optimal_ai_discounted(mdp, gamma):
    """
    Parameters
    gamma:0~1, discounting factor
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    ans = backward_iter_Q_discounted(info, gamma)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q

def backward_iter(info):
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    policy = np.zeros([ T, Nstate])
    Rs = np.zeros([T, Nstate])
    for step in range (T):
        pos = T  - step - 1
        if pos == T -1: # greedy in last step
            for s in range(Nstate):
                v = np.sum(P[s,:,:] * R[pos, :],axis = 1)
                policy[pos,s] = np.argmax(v)
                Rs[pos,s] = np.max(v)
        else: # optimal in expectation
            for s in range (Nstate):
                v = np.sum(P[s,:,:] * (R[pos, :] + Rs[pos+1,:] ),axis = 1)
                policy[pos,s] = np.argmax(v)
                Rs[pos,s] = np.max(v)
    return policy.astype(int),Rs

def backward_iter_Q(info):
    # T here is planning horizon, rather than others
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    policy = np.zeros([ T, Nstate])
    Rs = np.zeros([T, Nstate])
    Q = np.zeros([T, Nstate, Naction])
    for step in range (T):
        pos = T  - step - 1
        if pos == T -1: # greedy in last step
            for s in range(Nstate):
                v = np.sum(P[s,:,:] * R[pos, :],axis = 1)
                policy[pos,s] = np.argmax(v)
                Q[pos, s, : ] = v
                Rs[pos,s] = np.max(v)
        else: # optimal in expectation
            for s in range (Nstate):
                v = np.sum(P[s,:,:] * (R[pos, :] + Rs[pos+1,:] ),axis = 1)
                Q[pos, s, :] = v
                policy[pos,s] = np.argmax(v)
                Rs[pos,s] = np.max(v)
    return policy.astype(int),Rs, Q

def backward_iter_Q_discounted(info, gamma):
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    policy = np.zeros([ T, Nstate])
    Rs = np.zeros([T, Nstate])
    Q = np.zeros([T, Nstate, Naction])
    for step in range (T):
        pos = T  - step - 1
        if pos == T -1: # greedy in last step
            for s in range(Nstate):
                v = np.sum(P[s,:,:] * R[pos, :],axis = 1)
                policy[pos,s] = np.argmax(v)
                Q[pos, s, : ] = v
                Rs[pos,s] = np.max(v)
        else: # optimal in expectation
            for s in range (Nstate):
                v = np.sum(P[s,:,:] * (R[pos, :] + gamma*Rs[pos+1,:] ),axis = 1)
                Q[pos, s, :] = v
                policy[pos,s] = np.argmax(v)
                Rs[pos,s] = np.max(v)
    return policy.astype(int),Rs, Q

def opt_evaluator(mdp, policy, tau=-1, cost = np.zeros(0)):
    """
    Parameters
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    ps_init = np.ones([Nstate]) / Nstate
    ps = ps_init.copy()
    sumr = 0
    sumcost = 0
    costflag = True
    if cost.size == 0:
        cost = np.zeros([T, Nstate])
        costflag = False
    if tau == -1:
        for t in range (T):
            p2p = np.zeros([Nstate, Nstate])
            for s in range(Nstate):
                p2p[s, :] = P[s, policy[t, s], :]
            ps = ps.dot(p2p)
            sumr = sumr + np.sum(ps*R[t,:])
            sumcost = sumcost + np.sum(ps*cost[t,:])
    elif tau<T:
        p2p = np.zeros([Nstate, Nstate])
        for s in range(Nstate):
            p2p[s, :] = P[s, policy[0, s], :]
        for t in range(T):
            ps = ps.dot(p2p)
            sumr = sumr + np.sum(ps*R[0,:])
            sumcost = sumcost + np.sum(ps * cost[t, :])
    else:
        print("check tau in opt_evaluator for more info")
        return -1
    if not costflag:
        return sumr
    else:
        return sumr, sumcost

def nudge_ai_without_budget():
    """
    Comment: nudge ai to optimal behavior
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    return -1

def nudge_ai_budget(mdp, Q, cost_budget = 100):
    """
    nudge ai find best policy under cost budget constraint
    by solving a cosntraint problem
    Parameters
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    for s in range (Nstate):
        for k in range (Naction):
            for i in range (Nstate):
                if P[s,k,i] < 0:
                    P[s, k, i] = 0
    def CMDP_cost_cal(Q):
        cost = np.zeros([Nstate, Naction])
        for s in range (Nstate):
            cost[s,:] = np.max(Q[0,s,:]) - Q[0,s,:]
        return cost
    cost = CMDP_cost_cal(Q)
    prob = pulp.LpProblem(sense=pulp.LpMaximize)
    x = pulp.LpVariable.dicts(name='x', indexs=range(Nstate * T * Naction), lowBound=0, cat=pulp.LpContinuous)
    prob += pulp.lpSum(x[i*T*Naction + j * Naction + k] * R[j,i] for i in range(Nstate)
                       for j in range(T) for k in range(Naction))
    prob += pulp.lpSum(x[i*T*Naction + j * Naction + k] * cost[i,k] for i in range(Nstate)
                       for j in range(T) for k in range(Naction)) <= cost_budget
    for j in range(T):
        for i in range(Nstate):
            prob += pulp.lpSum( x[i*T*Naction + j * Naction + k] for k in range(Naction) ) <= 1
            # if j == 0 and i != 0:
            #     # initial state limit---> initial state is 0
            #     prob += pulp.lpSum( x[i*T*Naction + j * Naction + k] for k in range(Naction) ) <= 0
            if j == 0 :
                # initial state limit---> initial state is randommly choosen
                prob += pulp.lpSum( x[i*T*Naction + j * Naction + k] for k in range(Naction) ) <= 1.0/Nstate
            if j > 0:
                prob += pulp.lpSum(x[i * T * Naction + j * Naction + k] for k in range(Naction)) \
                        == pulp.lpSum(x[st1 * T * Naction + (j-1) * Naction + k] * P[st1,k,i] for st1 in range(Nstate) for k in range(Naction))
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.value(prob.objective) < 1:
        # print('no feasible solution for constrain in ',cost_budget ) # should not happen
        # exit(0)
        pass
    policy = np.zeros([T, Nstate])
    cost_map = np.zeros([T, Nstate])
    for j in range(T):
        for i in range(Nstate):
            for k in range(Naction):
                if x[i * T * Naction + j * Naction + k].varValue > 0:
                    if policy[j, i] > 0:
                        print('repeated assigned task')
                    policy[j, i] = np.int(k)
                    cost_map[j,i] = cost[i,k]
    # np.save("tmpfile.npy", [policy, cost_map])
    return policy.astype(int), cost_map

def nudge_ai2policy(mdp, Q, target_policy):
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    init_s = 45
    sumr = 0
    sumcost = 0
    s = init_s
    alist = np.zeros(T)
    rlist = np.zeros(T)
    slist = np.zeros(T)
    for t in range (T):
        a = target_policy[t,s]
        nexts = np.argmax(P[s, a, :])
        if len(R.shape) == 3:
            sumr += R[t,s,a]
            rlist[t] = R[t, s,a]
        else:
            sumr += R[t,nexts]
            rlist[t] = R[t, nexts]
        sumcost += np.max(Q[t,s,:]) - Q[t,s,a]
        alist[t]= a
        slist[t] = s
        s = nexts
    return sumcost

def nudge_ai_budget_rsa(mdp, cost, cost_budget=100):
    """
    Comment:Q change over time
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    for s in range (Nstate):
        for k in range (Naction):
            for i in range (Nstate):
                if P[s,k,i] < 0:
                    P[s, k, i] = 0
    prob = pulp.LpProblem(sense=pulp.LpMaximize)
    x = pulp.LpVariable.dicts(name='x', indexs=range(Nstate * T * Naction), lowBound=0, cat=pulp.LpContinuous)
    prob += pulp.lpSum(x[i*T*Naction + j * Naction + k] * R[j,i] for i in range(Nstate)
                       for j in range(T) for k in range(Naction))
    prob += pulp.lpSum(x[i*T*Naction + j * Naction + k] * cost[j,i,k] for i in range(Nstate)
                       for j in range(T) for k in range(Naction)) <= cost_budget
    for j in range(T):
        for i in range(Nstate):
            prob += pulp.lpSum( x[i*T*Naction + j * Naction + k] for k in range(Naction) ) <= 1
            # if j == 0 and i != 0:
            #     # initial state limit---> initial state is 0
            #     prob += pulp.lpSum( x[i*T*Naction + j * Naction + k] for k in range(Naction) ) <= 0
            if j == 0 :
                # initial state limit---> initial state is randommly choosen
                prob += pulp.lpSum( x[i*T*Naction + j * Naction + k] for k in range(Naction) ) <= 1.0/Nstate
                if i == 0:
                    prob += pulp.lpSum(x[i * T * Naction + j * Naction + k] for k in range(Naction)) == 1
                else:
                    prob += pulp.lpSum(x[i * T * Naction + j * Naction + k] for k in range(Naction)) == 0
            if j > 0:
                prob += pulp.lpSum(x[i * T * Naction + j * Naction + k] for k in range(Naction)) \
                        == pulp.lpSum(x[st1 * T * Naction + (j-1) * Naction + k] * P[st1,k,i] for st1 in range(Nstate) for k in range(Naction))
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.value(prob.objective) < 1:
        # print('no feasible solution for constrain in ',cost_budget ) # should not happen
        # exit(0)
        pass
    policy = np.zeros([T, Nstate],dtype=int)
    policydata = np.zeros([T, Nstate, Naction])
    cost_map = np.zeros([T, Nstate])
    for j in range(T):
        for i in range(Nstate):
            for k in range(Naction):
                policydata[j,i,k] = x[i * T * Naction + j * Naction + k].varValue
            policy[j,i] = np.argmax(policydata[j,i,:] )
            cost_map[j, i] = cost[j, i, policy[j,i]]
    return policy.astype(int), cost_map

def poison_ai_to_policy(mdp, gamma, target_policy, consistent = True, testmode=False):
    """
    find minima reward change to the target policy; need reward to be fixed to state
    Parameters
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    prob = pulp.LpProblem(sense=pulp.LpMinimize)
    x = pulp.LpVariable.dicts(name='x', indexs=range(Nstate), lowBound=0, cat=pulp.LpContinuous)
    prob += pulp.lpSum(x[i] for i in range (Nstate ))
    tmpP = P.copy()
    mx = P.copy()
    tau = target_policy.shape[0]

    R = R[0,:]
    if consistent:
        for k in range (1,tau):
            p2p = np.zeros([Nstate, Nstate])
            for s in range(Nstate):
                p2p[s, :] = P[s, target_policy[0, s], :]
            tmpP = tmpP.dot( p2p)
            mx += gamma[k] * tmpP
        for s in range (Nstate):
            a_select = target_policy[0,s]
            for a in range (Naction):
                if a == a_select :
                    pass
                else:
                    prob += pulp.lpSum( mx[s,a_select,i] *(x[i] + R[i]) for i in range(Nstate)) >= \
                            pulp.lpSum( mx[s,a,i] *(x[i] + R[i]) for i in range(Nstate))

    else:   # constraint here should be written in backward iteration form
            # such that poison will change agent belief as well
        for t in range(tau):
            tmpP = P.copy()
            mx = gamma[t] * P.copy()
            for k in range(t+1, tau):
                p2p = np.zeros([Nstate, Nstate])
                for s in range(Nstate):
                    p2p[s, :] = P[s, target_policy[k, s], :]
                tmpP = tmpP.dot(p2p)
                mx += gamma[k] * tmpP
            for s in range(Nstate):
                a_select = target_policy[t,s]
                for a in range(Naction):
                    if a == a_select:
                        pass
                    else:
                        prob += pulp.lpSum(mx[s, a_select, i] * (x[i] + R[i]) for i in range(Nstate)) >= \
                                pulp.lpSum(mx[s, a, i] * (x[i] + R[i]) for i in range(Nstate)) - 0.1
    h = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.value(prob.objective) < 0:
        print('no feasible solution for constrain ') # should not happen
        return - 1
    c = np.zeros(Nstate)
    for s in range(Nstate):
        c[s] = x[s].varValue
    if testmode == True:
        rx = R + c
        print("rx is ", rx[:10])
        Q = np.zeros([tau, Nstate,Naction])
        for t in range(tau):
            tmpP = P.copy()
            mx = gamma[t] * P.copy()
            for k in range(t+1, tau):
                p2p = np.zeros([Nstate, Nstate])
                for s in range(Nstate):
                    p2p[s, :] = P[s, target_policy[k, s], :]
                tmpP = tmpP.dot(p2p)
                mx += gamma[k] * tmpP
            for s in range(Nstate):
                for a in range(Naction):
                    Q[t,s,a] = np.sum(mx[s,a]*rx)
        return c,Q
    return c

def poison_ai_budget(mdp, Q, poison_bugget):
    """
    poison reward within given budget constraint
    Parameters
    ----------
    Returns
    -------
    """
    return -1

def poison_ai_budget_gradient(mdp, human_t, lr, gamma, Niter, beta=1.0):
    """
    Comment
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    def loss_func(x,c,ps,r):
        sumloss = - gamma* np.sum(np.abs(c))
        for t in range (T):
            for s in range(Nstate):
                for a in range (Naction):
                    sumloss += ps[t,s] * x[s,a] * r[s,a]
        loss_full[iter, 0] = - gamma* np.sum(np.abs(c))
        loss_full[iter,1] = sumloss + gamma* np.sum(np.abs(c))
        loss_full[iter, 2] = sumloss
        return sumloss

    def partial_Q2c(s1,a1,s2,a2):
        if s1==s2 and a1 == a2:
            return 1
        else:
            return 0

    def partial_x2c(s1,a1,s2,a2):
        k = beta / (np.sum(eqsa[s1,:]))**2 * eqsa[s1,a1]
        h = np.zeros(Nstate)
        for a in range (Naction):
            h[a] = eqsa[s1,a] * (partialqc[s1,a1,s2,a2]-partialqc[s1,a,s2,a2])
        return k * np.sum(h)

    def partial_p2c(t1,s1,s2,a2):
        if t1 == 0:
            return 0
        else:
            h = np.zeros([Nstate, Naction])
            for s3, a3 in itertools.product(range(Nstate), range(Naction)):
                h[s3,a3] = P[s3,a3,s1] * (pst[t-1, s3]*partialxc[s3,a3,s2,a2] + partialpsc[t-1,s3,s2,a2]*xsa[s3,a3])
            return np.sum(h)

    def partial_z2c(s1,a1):
        if np.abs(c[s1,a1])<0.001:
            # return np.sign(c[s1,a1])
            return 0
        else:
            return np.sign(c[s1,a1])

    def partial_c(s1,a1):
        h = np.zeros([T,Nstate,Naction])
        for t,s2,a2 in itertools.product(range(T), range(Nstate), range(Naction)):
            h[t,s2,a2] = Reward[s1,a1] * (pst[t, s2]*partialxc[s2,a2,s1,a1] + partialpsc[t,s2,s1,a1]*xsa[s2,a2])
        return np.sum(h)


    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3] # r(s,a, T )
    P = info[4] # p(s,a,s')
    Reward = R[0,:,:]
    ps0 = np.ones(Nstate) / Nstate
    # np.random.seed(seed=1023)
    B = 3
    c0 = np.random.normal(0,1.0, size=Reward.shape)
    c = c0.copy()

    tstart = time.time()
    qsa = np.zeros([Nstate, Naction])
    pst = np.zeros([T,Nstate])
    xsa = np.zeros([Nstate, Naction])
    loss = np.zeros(Niter)
    loss_full = np.zeros([Niter, 3])
    for iter in range(Niter):
        rprime = Reward + c
        qsa = rprime
        eqsa = np.exp(beta * qsa)
        xsa = eqsa.copy()
        for s in range (Nstate):
            xsa[s,:] = xsa[s,:] / np.sum(xsa[s,:])
        pst = np.zeros([T, Nstate])
        for t in range(T):
            if t==0:
                pst[t,:] = ps0
            else:
                for s in range (Nstate):
                    for a in range(Naction):
                        for ss in range(Nstate):
                            pst[t,ss] += pst[t-1,s] * xsa[s,a] * P[s,a,ss]
        loss[iter] = loss_func(x=xsa,c=c,ps=pst,r=Reward)
        partialqc = np.zeros([Nstate, Naction, Nstate, Naction])
        partialxc = np.zeros([Nstate, Naction, Nstate, Naction])
        partialpsc = np.zeros([T, Nstate, Nstate, Naction])
        partialzc = np.zeros( [Nstate, Naction])
        # for s1,s2,a1,a2 in itertools.product(range(Nstate), range(Nstate),range(Naction), range(Naction)):
        #     partialqc[s1,a1,s2,a2] = partial_Q2c(s1,a1,s2,a2)
        # for s1, s2, a1, a2 in itertools.product(range(Nstate), range(Nstate), range(Naction), range(Naction)):
        #     partialxc[s1,a1,s2,a2] = partial_x2c(s1,a1,s2,a2)
        # for t, s1, s2, a2 in itertools.product(range(T), range(Nstate), range(Nstate), range(Naction)):
        #     partialpsc[t,s1, s2, a2] = partial_p2c(t,s1, s2, a2)
        # for s1,a1 in itertools.product(range(Nstate), range(Naction)):
        #     partialzc[s1,a1] = partial_z2c(s1,a1)
        # cgrad = -gamma * partialzc
        # for s1,a1 in itertools.product(range(Nstate), range(Naction)):
        #     cgrad[s1,a1] += partial_c(s1,a1)
        # c += lr * cgrad
        # stotastic gradient
        s = int(np.random.randint(Nstate))
        # s = int(np.random.randint(1,3))
        a = int(np.random.randint(Naction))
        for s1,s2,a1,a2 in itertools.product(range(Nstate), [s], range(Naction),[a]):
            partialqc[s1,a1,s2,a2] = partial_Q2c(s1,a1,s2,a2)
        for s1, s2, a1, a2 in itertools.product(range(Nstate), [s], range(Naction),[a]):
            partialxc[s1,a1,s2,a2] = partial_x2c(s1,a1,s2,a2)
        for t, s1, s2, a2 in itertools.product(range(T), range(Nstate), [s], [a]):
            partialpsc[t,s1, s2, a2] = partial_p2c(t,s1, s2, a2)
        for s1,a1 in itertools.product([s], [a]):
            partialzc[s1,a1] = partial_z2c(s1,a1)
        cgrad = -gamma * partialzc
        for s1,a1 in itertools.product(range(Nstate), range(Naction)):
            cgrad[s1,a1] += partial_c(s1,a1)
        c[s,a] += lr * cgrad[s,a]
    return c

def poison_ai_budget_gradient_auto(mdp, gt, budget,agenttype=1, lr=0.001, Niter=1000, beta=3.0):
    """
    Comment: an auto generated gradient descent solver for reward modification;
    poison_ai_budget_gradient(mdp, human_t, lr, gamma, Niter, beta=1.0)
    ----------
    Parameters
    gt: human discounting function
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3] # r(s,a, T )
    P = info[4] # p(s,a,s')
    Reward = R[0,:,:]  # Nstate Naciton
    Reward_ai = R[0, :, :]  # Nstate Naciton # this information should be given by out source

    ps0 = np.ones(Nstate) / Nstate
    np.random.seed(seed=1023)
    constraint_penalty = 10.0


    tstart = time.time()
    qsa = np.zeros([T,Nstate, Naction]) # updated Q value
    pst = np.zeros([T, Nstate]) # pst, probability of state every time
    csa = np.zeros([Nstate, Naction]) # reward modification
    csa = np.random.rand(Nstate, Naction) - 0.5
    csa = csa * (budget) / np.sum(np.abs(csa))  # initialize
    rho = np.zeros([T,Nstate, Naction])  # policy from Q
    trans = np.zeros([T,Nstate, Nstate]) # T*S*S transformer from probbaility of state
    transRst = np.zeros([T, Nstate])  # define cumulative reward
    # print(csa, gt)
    if np.size(gt) != T:
        print('check discounting function, not match with T')
        exit(0)

    def loss_detail(qsa,c,pst):
        gain = 0
        for t in range (T):
            for s in range(Nstate):
                for a in range (Naction):
                    gain += qsa[t,s,a] * pst[t,s] * Reward_ai[s,a]
        constraint =  budget - jnp.sum(jnp.abs(c))
        if constraint > 0:
            constraint = 0
        return gain + constraint_penalty * constraint

    def loss_byagent(csa):
        """
        Comment: calculate loss function by repeat agent process
        """
        agent_mdp = MDPbasic()
        agent_mdp.read_info([R,P])
        agent_mdp.poison_reward(csa)
        bounded = general_human_backward(agent_mdp, gt, agenttype=agenttype)
        ans = sub_evaluator_rsa(mdp=mdp, policy=bounded[0])
        constraint = budget - np.sum(np.abs(csa))
        if constraint > -0.01:
            constraint = 0
        return ans, constraint

    def softmax(q):
        A = jnp.exp(beta * q)
        return A / jnp.sum(A)

    def loss(c):
        # the loss only depends on current change to environment
        new_reward = Reward + c
        qsa = jnp.zeros([T, Nstate, Naction])  # updated Q value
        pst = jnp.zeros([T, Nstate])  # pst, probability of state every time
        rho = jnp.zeros([T, Nstate, Naction])  # policy from Q
        trans = jnp.zeros([T, Nstate, Nstate])  # T*S*S transformer from probbaility of state
        transRst = jnp.zeros([T, Nstate]) # define cumulative reward
        tmpRs = jnp.zeros([Nstate])
        # print(np.shape(new_reward), np.shape(qsa), np.shape(c))
        def naive_solver(tau):  # solve a problem in belief
            subpolicy = jnp.zeros([tau, Nstate,Naction])
            subRs = jnp.zeros([tau, Nstate])  # records of cumulative reward
            subQ = jnp.zeros([tau, Nstate, Naction])
            for step in range(tau):
                pos = tau - step - 1
                if pos == tau - 1:  # greedy in last step
                    for s in range(Nstate):
                        v = gt[pos] * R[pos, s, :]
                        subpolicy = subpolicy.at[pos, s, :].set(softmax(v))
                        subQ = subQ.at[pos, s, :].set(new_reward[s, :])
                        subRs = subRs.at[pos, s].set(jnp.sum(subpolicy[pos,s,:]*v))
                else:  # optimal in expectation
                    for s in range(Nstate):
                        v = gt[pos] * R[pos, s, :] + jnp.sum(P[s, :, :] * subRs[pos + 1, :], axis=1)
                        subQ = subQ.at[pos, s, :].set(new_reward[s, :])
                        subpolicy = subpolicy.at[pos, s, :].set(softmax(subQ[pos,s,:]))
                        subRs = subRs.at[pos, s].set(jnp.sum(subpolicy[pos,s,:]*v))
            return subpolicy, subQ

        # update qsa and rho based on gt and new_reward
        if agenttype == 0:
            # naive agent
            for step in range(T):
                subans = naive_solver(T - step)
                pos = T - step - 1
                qsa = qsa.at[pos, :, :].set(subans[1][0, :, :])
                for s in range(Nstate):
                    rho = rho.at[pos, s, :].set(softmax(qsa[pos, s, :]))
                    trans = trans.at[pos, s, :].set(jnp.dot(rho[pos, s, :], P[s, :, :]))
        elif agenttype == 1:
            # sophistacited
            for step in range(T):
                pos = T - step - 1
                if pos == T - 1:  # greedy in last step
                    for s in range(Nstate):
                        qsa = qsa.at[pos,s,:].set(new_reward[s,:])
                        rho = rho.at[pos,s,:].set(softmax(qsa[pos, s,:]))
                        trans = trans.at[pos,s,:].set(jnp.dot(rho[pos,s,:], P[s,:,:]))
                        transRst = transRst.at[pos,s].set(jnp.sum(rho[pos,s,:]*new_reward[s,:]))
                else: # new Q in first steps
                    tmpRs = tmpRs.at[:].set(jnp.sum(transRst[pos + 1:, :].T * gt[1:step + 1], axis=1))
                    for s in range(Nstate):
                        v = new_reward[s, :] + jnp.sum(P[s, :, :] * tmpRs[:], axis=1)
                        qsa = qsa.at[pos,s,:].set(v)
                        rho = rho.at[pos,s,:].set(softmax(qsa[pos, s,:]))
                        trans = trans.at[pos,s,:].set(jnp.dot(rho[pos,s,:], P[s,:,:]))
                        transRst = transRst.at[pos, s].set(jnp.sum(rho[pos, s, :] * new_reward[s, :]))
                    for j in range(pos+1,T):
                        transRst = transRst.at[j, :].set(jnp.dot(trans[pos, :, :], transRst[j, :].T))
        for t in range(T):
            if t == 0:
                pst = pst.at[t,:].set(ps0)
            else:
                pst = pst.at[t, :].set(jnp.dot(pst[t - 1, :], trans[t-1,:,:]))
        psta = pst.reshape([T, Nstate,1]) * rho
        gain = jnp.sum( psta *  Reward_ai.reshape([1,Nstate, Naction]) )
        constraint =  budget - jnp.sum(jnp.abs(c))
        if constraint > -0.01:
            constraint = 0
        loss = gain + constraint_penalty * constraint
        return loss

    def cons(c):
        return constraint_penalty * ( budget - np.sum(jnp.abs(c)))

    def grad_loss(c):
        # return grad(loss,argnums=1)(qsa,c,pst)
        return grad(loss, argnums=0)(c)

    c_grad = grad(loss, argnums=0)
    for iter in range(Niter):
        print('iter=', iter, ' time=',time.time()-tstart, ' seconds')
        # L = loss(csa)
        L = loss_byagent(csa)
        L2 = loss(csa)
        delta = c_grad(csa)
        csa = csa + lr * delta
        print('iter = ', iter, ' ,Loss=', L[0] + constraint_penalty *L[1], ' , gain/constraint=', L[0], L[1])
    print('finish all in ', time.time()-tstart, ' seconds')

    return -1

def sub_evaluator_rsa(mdp, policy, tau=-1):
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    ps_init = np.ones([Nstate]) / Nstate
    ps = ps_init.copy()
    sumr = 0
    if tau == -1:
        for t in range (T):
            p2p = np.zeros([Nstate, Nstate])
            for s in range(Nstate):
                p2p[s, :] = P[s, policy[t, s], :]
                sumr += ps[s] * R[t,s,policy[t, s]]
            ps = ps.dot(p2p)
    elif tau<T:
        p2p = np.zeros([Nstate, Nstate])
        Rs = np.zeros([Nstate])
        for s in range(Nstate):
            p2p[s, :] = P[s, policy[0, s], :]
            Rs[s] = R[0,s,policy[0, s]]
        for t in range(T):
            ps = ps.dot(p2p)
            sumr = sumr + np.sum(ps*Rs)
    return sumr

def optimal_ai_discounted_rsa(mdp, gamma):
    """
    Parameters
    gamma:0~1, discounting factor
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    ans = backward_iter_Q_discounted_rsa(info, gamma)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q

def backward_iter_Q_discounted_rsa(info, gamma, cost=np.array(0)):
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    policy = np.zeros([ T, Nstate])
    Rs = np.zeros([T, Nstate])
    Q = np.zeros([T, Nstate, Naction])
    if np.size(cost) < 2:
        for step in range (T):
            pos = T  - step - 1
            if pos == T -1: # greedy in last step
                for s in range(Nstate):
                    v = R[pos, s,:]
                    policy[pos,s] = np.argmax(v)
                    Q[pos, s, : ] = v
                    Rs[pos,s] = np.max(v)
            else: # optimal in expectation
                for s in range (Nstate):
                    v = R[pos, s,:] + np.sum(P[s,:,:] * (0 + gamma*Rs[pos+1,:] ),axis = 1)
                    Q[pos, s, :] = v
                    policy[pos,s] = np.argmax(v)
                    Rs[pos,s] = np.max(v)
    else: # with a cost function
        for step in range (T):
            pos = T  - step - 1
            if pos == T -1: # greedy in last step
                for s in range(Nstate):
                    v = R[pos, s,:] - cost[s,:]
                    policy[pos,s] = np.argmax(v)
                    Q[pos, s, : ] = v
                    Rs[pos,s] = np.max(v)
            else: # optimal in expectation
                for s in range (Nstate):
                    v = R[pos, s,:] - cost[s,:] + np.sum(P[s,:,:] * (0 + gamma*Rs[pos+1,:] ),axis = 1)
                    Q[pos, s, :] = v
                    policy[pos,s] = np.argmax(v)
                    Rs[pos,s] = np.max(v)
    return policy.astype(int),Rs, Q

def myopic_human_rsa(mdp):
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    policy = np.zeros([T, Nstate])
    Rs = np.zeros([T, Nstate])
    Q = np.zeros([T, Nstate, Naction])
    for step in range(T):
        pos = T - step -1
        for s in range (Nstate):
            v = R[pos,s,:]
            policy[pos, s] = np.argmax(v)
            Q[pos, s, :] = v
            Rs[pos, s] = np.max(v)
    return policy.astype(int), Q

def bounded_rational_human_discounted_rsa(mdp, tau=1, gamma=1):
    """
    Comment: based on bounded rational human discounted, but with rsa
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    def backward_iter_Q_discounted_finitehorizon_rsa(info, gamma, tau):
        Nstate = info[0]
        Naction = info[1]
        T = info[2]
        oriT=T
        R = info[3]
        P = info[4]
        if tau<T:
            T = tau
        policy = np.zeros([T, Nstate])
        Rs = np.zeros([T, Nstate])
        Q = np.zeros([T, Nstate, Naction])
        for step in range(T):
            pos = T - step - 1
            if pos == T - 1:  # greedy in last step
                for s in range(Nstate):
                    v = R[pos, s,:]
                    # v = np.sum(P[s, :, :] * R[pos, :], axis=1)
                    policy[pos, s] = np.argmax(v)
                    Q[pos, s, :] = v
                    Rs[pos, s] = np.max(v)
            else:  # optimal in expectation
                for s in range(Nstate):
                    v = R[pos, s,:]  + np.sum(P[s, :, :] * ( gamma * Rs[pos + 1, :]), axis=1)
                    Q[pos, s, :] = v
                    policy[pos, s] = np.argmax(v)
                    Rs[pos, s] = np.max(v)
        allpolicy = np.zeros([oriT, Nstate], dtype=int)
        allQ = np.zeros([oriT, Nstate, Naction], dtype=int)
        for t in range(oriT):
            if t < oriT - tau:
                allpolicy[t, :] = policy[0, :]
                allQ[t, :, :] = Q[0, :, :]
            else:
                allpolicy[t, :] = policy[t + tau - oriT, :]
                allQ[t, :, :] = Q[t + tau - oriT, :, :]
        return allpolicy.astype(int), Rs, allQ

    info = mdp.print_all_info()
    ans = backward_iter_Q_discounted_finitehorizon_rsa(info, gamma,tau)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q

def nudge_ai_without_budget_rsa(mdp, Q):
    """
    Comment
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    q = Q[0,:,:]
    c = q.copy()
    for s in range(q.shape[0]):
        c[s,:] = np.max(q[s,:]) - q[s,:]
    ans = backward_iter_Q_discounted_rsa(info,gamma = 1, cost=c)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q

def poison_ai_to_policy_rsa(mdp, gamma, target_policy, consistent = True, testmode=False, punishment =0 ):
    """
    Comment: given a target policy, try to change human to follow a certain policy
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    info = mdp.print_all_info()
    Nstate = info[0]
    Naction = info[1]
    T = info[2]
    R = info[3]
    P = info[4]
    prob = pulp.LpProblem(sense=pulp.LpMinimize)
    x = pulp.LpVariable.dicts(name='x', indexs=range(Nstate*Naction), lowBound=0, cat=pulp.LpContinuous)
    prob += pulp.lpSum( punishment[i] * x[i] for i in range (Nstate*Naction ))
    tmpP = P.copy()
    mx = P.copy()
    mxa = np.zeros([Nstate, Naction, Nstate, Naction])
    tau = target_policy.shape[0]
    for s1, a1, s2 in itertools.product(range(Nstate), range(Naction), range(Nstate)):
        if s2 == s1:
            mxa[s1, a1, s2, a1] = 1

    R = R[0,:]
    if consistent:
        for k in range (1,tau):
            mx = gamma[k] * tmpP
            p2p = np.zeros([Nstate, Nstate])
            for s in range(Nstate):
                p2p[s, :] = P[s, target_policy[k, s], :]
            tmpP = tmpP.dot( p2p)
            for s1,a1,s2 in itertools.product(range(Nstate), range(Naction), range(Nstate)):
                mxa[s1,a1,s2, target_policy[k, s2]] += mx[s1,a1,s2]
        for s in range (Nstate):
            a_select = target_policy[0,s]
            for a in range (Naction):
                if a == a_select :
                    pass
                else:
                    # itertools.product(range(20), range(10))
                    prob += pulp.lpSum( mxa[s,a_select,i, j] *(x[i*Naction+j] + R[i,j]) for i in range(Nstate) for j in range(Naction) ) >= \
                            pulp.lpSum( mxa[s,a,i, j] *(x[i*Naction+j] + R[i,j]) for i in range(Nstate) for j in range (Naction))
    else:   # constraint here should be written in backward iteration form
            # such that poison will change agent belief as well
        for t in range(tau):
            tmpP = P.copy()
            mx = gamma[t] * P.copy()
            for k in range(t+1, tau):
                p2p = np.zeros([Nstate, Nstate])
                for s in range(Nstate):
                    p2p[s, :] = P[s, target_policy[k, s], :]
                tmpP = tmpP.dot(p2p)
                mx += gamma[k] * tmpP
            for s in range(Nstate):
                a_select = target_policy[t,s]
                for a in range(Naction):
                    if a == a_select:
                        pass
                    else:
                        prob += pulp.lpSum(mx[s, a_select, i] * (x[i] + R[i]) for i in range(Nstate)) >= \
                                pulp.lpSum(mx[s, a, i] * (x[i] + R[i]) for i in range(Nstate))
    h = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.value(prob.objective) < 0:
        print('no feasible solution for constrain ') # should not happen
        return - 1
    c = np.zeros([Nstate, Naction])
    for s in range(Nstate):
        for a in range(Naction):
            c[s,a] = x[s*Naction+a].varValue
    if testmode == True:
        rx = R + c
        print("rx is ", rx[:10])
        Q = np.zeros([tau, Nstate,Naction])
        for t in range(tau):
            tmpP = P.copy()
            mx = gamma[t] * P.copy()
            for k in range(t+1, tau):
                p2p = np.zeros([Nstate, Nstate])
                for s in range(Nstate):
                    p2p[s, :] = P[s, target_policy[k, s], :]
                tmpP = tmpP.dot(p2p)
                mx += gamma[k] * tmpP
            for s in range(Nstate):
                for a in range(Naction):
                    Q[t,s,a] = np.sum(mx[s,a]*rx)
        return c,Q
    return c
