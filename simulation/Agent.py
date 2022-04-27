"""
Comment: biased agent model
----------
Parameters
----------
Returns
-------
"""
import numpy as np

def optimal_huamn_discounted(mdp, gamma):
    """
    Parameters
    ----------
    Returns
    -------
    """
    return optimal_ai_discounted(mdp, gamma)

def myopic_human(mdp):
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
    policy = np.zeros([T, Nstate])
    Rs = np.zeros([T, Nstate])
    Q = np.zeros([T, Nstate, Naction])
    for step in range(T):
        pos = T - step -1
        for s in range (Nstate):
            v = np.sum(P[s, :, :] * R[pos, :], axis=1)
            policy[pos, s] = np.argmax(v)
            Q[pos, s, :] = v
            Rs[pos, s] = np.max(v)
    return policy.astype(int), Q

def bounded_rational_human_discounted(mdp, tau=-1, gamma=1):
    """
    Parameters
    tau: is the planning time, tau=1 equals to myopic agent
    gamma: discounted factor
    ps: this human assume reward is fixed and invariant over time,
        so policy here will be  [tau, Nstate], representing his belief
    ----------
    Returns
    -------
    """
    def backward_iter_Q_discounted_finitehorizon(info, gamma, tau):
        Nstate = info[0]
        Naction = info[1]
        T = info[2]
        oriT = T
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
                    v = np.sum(P[s, :, :] * R[pos, :], axis=1)
                    policy[pos, s] = np.argmax(v)
                    Q[pos, s, :] = v
                    Rs[pos, s] = np.max(v)
            else:  # optimal in expectation
                for s in range(Nstate):
                    v = np.sum(P[s, :, :] * (R[pos, :] + gamma * Rs[pos + 1, :]), axis=1)
                    Q[pos, s, :] = v
                    policy[pos, s] = np.argmax(v)
                    Rs[pos, s] = np.max(v)
        allpolicy = np.zeros([oriT, Nstate], dtype=int)
        allQ = np.zeros([oriT, Nstate, Naction], dtype=int)
        for t in range(oriT):
            if t < oriT-tau:
                allpolicy[t,:] = policy[0,:]
                allQ[t,:,:] = Q[0,:,:]
            else:
                allpolicy[t, :] = policy[t+tau-oriT, :]
                allQ[t, :, :] = Q[t+tau-oriT, :, :]
        return allpolicy.astype(int), Rs, allQ

    info = mdp.print_all_info()
    ans = backward_iter_Q_discounted_finitehorizon(info, gamma,tau)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q

def hyperbolic_human(mdp, k):
    """
    Parameters
    ----------
    Returns
    -------
    """
    def backward_iter_Q_hyperbolic(info, k, tau):
        Nstate = info[0]
        Naction = info[1]
        T = info[2]
        R = info[3]
        P = info[4]
        if tau<T:
            T = tau
        policy = np.zeros([T, Nstate])
        Rs = np.zeros([T, Nstate])
        Q = np.zeros([T, Nstate, Naction])
        gamma =  np.zeros(T)
        for i in range (T):
            gamma[i] = 1.0 / (1+ k*i)
        for step in range(T):
            pos = T - step - 1
            if pos == T - 1:  # greedy in last step
                for s in range(Nstate):
                    v = gamma[pos] * np.sum(P[s, :, :] * R[pos, :], axis=1)
                    policy[pos, s] = np.argmax(v)
                    Q[pos, s, :] = v
                    Rs[pos, s] = np.max(v)
            else:  # optimal in expectation
                for s in range(Nstate):
                    v = np.sum(P[s, :, :] * (gamma[pos] * R[pos, :] + Rs[pos + 1, :]), axis=1)
                    Q[pos, s, :] = v
                    policy[pos, s] = np.argmax(v)
                    Rs[pos, s] = np.max(v)
        return policy.astype(int), Rs, Q

    info = mdp.print_all_info()
    T = info[2]
    ans = backward_iter_Q_hyperbolic(info, k, tau=T)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q

def general_human_backward(mdp, gt, agenttype):
    """
    Parameters:
    mdp: default is rsa
    gt: discounting function, must be in size of T
    agenttype: 0=naive; 1=sophisticated
    ----------
    Returns: policy, Q
    -------
    """
    info = mdp.print_all_info()
    T = info[2]
    if np.size(gt) != T:
        print('discouting function not match with T, check in human solver')

    def backward_iter_Q_general(info):
        Nstate = info[0]
        Naction = info[1]
        T = info[2]
        R = info[3]
        P = info[4]
        policy = np.zeros([T, Nstate],dtype=int)
        Rs = np.zeros([T, Nstate]) # records of cumulative reward
        Q = np.zeros([T, Nstate, Naction])
        Qtt = np.zeros([T, Nstate, Naction, T]) # this is for naive agent's belief
        trans = np.zeros([T, Nstate, Nstate]) # state transform probability by current policy
        Rst = np.zeros([T, Nstate]) # records of reward when following policy
        transRst = np.zeros([T, Nstate]) # records of transed reward (  trans [1] * trans[2] *Rst[3])


        def naive_solver(tau):  # solve a problem in belief
            subpolicy = np.zeros([tau, Nstate], dtype=int)
            subRs = np.zeros([tau, Nstate])  # records of cumulative reward
            subQ = np.zeros([tau, Nstate, Naction])
            for step in range(tau):
                pos = tau - step - 1
                if pos == tau - 1:  # greedy in last step
                    for s in range(Nstate):
                        v = gt[pos] * R[pos, s, :]
                        subpolicy[pos, s] = np.argmax(v)
                        subQ[pos, s, :] = v
                        subRs[pos, s] = np.max(v)
                else:  # optimal in expectation
                    for s in range(Nstate):
                        v = gt[pos] * R[pos, s, :] + np.sum(P[s, :, :] * subRs[pos + 1, :], axis=1)
                        subQ[pos, s, :] = v
                        subpolicy[pos, s] = np.argmax(v)
                        subRs[pos, s] = np.max(v)
            return subpolicy.astype(int), subQ

        if agenttype == 0: # naive agent
            for step in range(T):
                subans = naive_solver(T-step)
                Qtt[step, :,:,0] = subans[1][0,:,:] 
                Q[step, :, :] = Qtt[step, :,:,0]
                policy[step,:] = subans[0][0,:]
                for j in range(Nstate):
                    Rs[step,j] = Q[step, j, policy[step,j]]

        elif agenttype == 1: # sophisticated agent
            for step in range(T):
                pos = T - step - 1
                if pos == T - 1:  # greedy in last step
                    for s in range(Nstate):
                        v = R[pos, s, :]
                        policy[pos, s] = np.argmax(v)
                        Q[pos, s, :] = v
                        Rs[pos, s] = np.max(v)
                        Rst[pos, s] = np.max(v)
                        trans[pos,s,:] = P[s,policy[pos, s],:]
                    transRst[pos , :] = Rst[pos, :]
                else:  # optimal in expectation
                    tmpRs = np.sum( transRst[pos+1:, :].T * gt[1:step+1] , axis=1)
                    for s in range(Nstate):
                        v = R[pos, s, :] +  np.sum(P[s, :, :] * tmpRs[:], axis=1)
                        Q[pos, s, :] = v
                        policy[pos, s] = np.argmax(v)
                        Rst[pos, s] = R[pos, s, policy[pos,s]]
                        trans[pos,s,:] = P[s,policy[pos, s],:]
                    transRst[pos , :] = Rst[pos , :]
                    for j in range(pos+1,T):
                        transRst[j, :] = np.dot(trans[pos, :, :], transRst[j, :].T)
        else:
            print('a mistake in agent type')
            exit(0)
        return policy.astype(int), Rs, Q


    ans = backward_iter_Q_general(info)
    policy = ans[0]
    Rs = ans[1]
    Q = ans[2]
    return policy, Q


def general_discount(T,para1,para2 = 0, agenttype=0):
    """
    Comment: generate a discounting function based on agenttype
    ----------
    Parameters
    ----------
    Returns
    -------
    """
    gt = np.zeros(T)
    if agenttype == 0: # bounded
        # gt = np.logspace(0,T-1,num=T, base=para1)
        gt[:para2] = np.logspace(0, para2 - 1, num=para2, base=para1)
    elif agenttype == 1: # hyperbolic
        for t in range(T):
            gt[t] = 1.0 / (1 + para1 * t)
    else:
        print('misaligned agent type')
        exit(0)
    return gt
