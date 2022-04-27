"""
Comment: MDP environment
----------
Parameters
----------
Returns
-------
"""

### basic class for an MDP

import numpy as np

class MDPbasic():
    Nstate = 1 # start from 0
    Naction = 1 # start from 0
    T = 10
    Reward = np.zeros([T, Nstate, Naction])  
    Reward_ai = np.zeros([T, Nstate, Naction])  
    init_state = 0
    init_ps = np.ones(Nstate) / Nstate
    P = np.zeros(1)
    np.random.seed(1023)

    def __init__(self, K = 2, A = 3, T = 10):
        self.Nstate = K
        self.Naction = A
        self.Reward = np.zeros([self.T, self.Nstate])
        self.T = T

    def __call__(self):
        pass

    def init_game(self):
        self.cur_state = self.init_state
        self.sum_reward = 0
        self.cur_t = 0

    def reset(self):
        self.init_game()

    def random_init_transition(self):
        self.P = np.zeros([self.Nstate,self.Naction,  self.Nstate]) # transition probability
        self.P = np.random.rand(self.Nstate, self.Naction, self.Nstate)
        for i in range(self.Nstate):
            for j in range (self.Naction):
                self.P[i,j,:] = self.P[i,j,:] / np.sum(self.P[i,j,:])

    def one_hot_init_transition(self):
        self.P = np.zeros([self.Nstate,self.Naction,  self.Nstate]) # transition probability
        self.P = np.random.rand(self.Nstate, self.Naction, self.Nstate)
        for i in range(self.Nstate):
            for j in range (self.Naction):
                p = self.P[i,j,:]
                idx = np.argmax(p)
                p[:] = 0
                p[idx] = 1
                self.P[i,j,:] = p

    def random_init_reward(self):
        self.Reward = np.random.rand(self.T, self.Nstate)

    def init_game_random(self):
        self.init_game()
        self.random_init_transition()
        self.random_init_reward()

    def init_game_one_hot(self):
        self.init_game()
        self.one_hot_init_transition()
        self.random_init_reward()

    def default_setting1(self):
        K = 3
        A = 4
        T = 10
        self.__init__(K,A,T)
        self.init_game_random()

    def read_info(self,info):
        if len(info) == 2:
            self.Reward = info[0].copy()
            self.P = info[1].copy()
            self.Nstate = self.Reward.shape[1]
            self.T = self.Reward.shape[0]
            self.Naction = self.P.shape[1]
            self.reset()

    def poison_reward(self, poison):
        for t in range (self.T):
            self.Reward[t,:] = self.Reward[t,:] + poison

    def set_reward(self,r):
        self.Reward =r

    def default_setting2(self):
        K = 5
        A = 5
        T = 100
        self.__init__(K,A,T)
        self.init_game_random()

    def default_setting3(self):
        K = 25
        A = 5
        T = 100
        self.__init__(K, A, T)
        self.init_game_one_hot()

    def default_setting4(self):
        #     this model will fixed rewrad with state, no change over time
        K = 25
        A = 5
        T = 100
        self.__init__(K, A, T)
        self.init_game()
        self.random_init_transition()
        rr = np.random.rand(self.Nstate)
        self.Reward = np.zeros([self.T, self.Nstate])
        for t in range(T):
            self.Reward[t,:] = rr

    def simtestcase1(self, seed=1023):
        np.random.seed(seed)
        K = 100
        A = 4
        T = 20
        self.__init__(K, A, T)
        self.init_game()
        self.simtestcase1_init_transition()
        # rr = np.abs(np.random.normal(0.0,1.0,self.Nstate))
        rr = np.random.rand(K)/2
        idx = np.random.randint(2,K-2)
        rr[idx-2:idx+2] = np.random.rand(4)/2 + 0.5
        idx = np.random.randint(2,K-2)
        rr[idx-2:idx+2] = np.random.rand(4)/2 + 0.1
        idx = np.random.randint(2,K-2)
        rr[idx-2:idx+2] = np.random.rand(4)/2 + 0.2
        idx = np.random.randint(2,K-2)
        rr[idx-2:idx+2] = np.random.rand(4)/2 + 0.3
        self.Reward = np.zeros([self.T, self.Nstate])
        for t in range(T):
            self.Reward[t,:] = rr

    def simtestcase1_rsa(self, seed=1023):
        np.random.seed(seed)
        K = 100
        A = 4
        T = 20
        self.__init__(K, A, T)
        self.init_game()
        self.simtestcase1_init_transition()
        # rr = np.abs(np.random.normal(0.0,1.0,self.Nstate))
        rr = np.random.rand(K*A)/2
        idx = np.random.randint(8,K-8)
        rr[idx-8:idx+8] = np.random.rand(16)/2 + 0.5
        idx = np.random.randint(8,K-8)
        rr[idx-8:idx+8] = np.random.rand(16)/2 + 0.1
        idx = np.random.randint(8,K-8)
        rr[idx-8:idx+8] = np.random.rand(16)/2 + 0.2
        idx = np.random.randint(8,K-8)
        rr[idx-8:idx+8] = np.random.rand(16)/2 + 0.3
        self.Reward = np.zeros([self.T, self.Nstate, self.Naction])
        for t in range(T):
            self.Reward[t,:] = rr.reshape([self.Nstate, self.Naction])

    def pilot_testcase_simulate(self,seed=1023):
        np.random.seed(seed)
        K = 100
        A = 4
        T = 20
        self.__init__(K, A, T)
        self.init_game()
        self.P = self.init_P_for_2Dmove_nobound(K,A, 10,10)
        # self.P = self.init_P_for_2Dmove(K, A, 10, 10)
        rr = np.random.randint(1, 11, size=[K])
        self.Reward = np.zeros([self.T, self.Nstate])
        for t in range(T):
            self.Reward[t,:] = rr
        self.reward_ai = self.Reward.copy()
        return -1

    def pilot_testcase_simulate_0905pilot(self, seed = 1023):
        np.random.seed(seed)
        K = 100
        A = 4
        T = 20
        self.__init__(K, A, T)
        self.init_game()
        self.P = self.init_P_for_2Dmove(K, A, 10, 10)
        rr = self.reward_design_0905pilot_2digits()
        self.Reward = np.zeros([self.T, self.Nstate])
        for t in range(T):
            self.Reward[t,:] = rr
        self.reward_ai = self.Reward.copy()
        return -1

    def pilot_testcase_simulate_0906pilot(self, seed = 1023):
        np.random.seed(seed)
        K = 100
        A = 4
        T = 20
        init_s = 45
        self.__init__(K, A, T)
        self.init_game()
        self.init_s = init_s
        self.P = self.init_P_for_2Dmove(K, A, 10, 10)
        rr = self.reward_design_0906pilot()
        self.Reward = np.zeros([self.T, self.Nstate])
        for t in range(T):
            self.Reward[t,:] = rr
        self.reward_ai = self.Reward.copy()
        return -1

    def reward_design_0905pilot(self):
        # reward = np.zeros([10*10], dtype=int)
        reward = np.random.randint(1,5, size = [10,10], dtype=int)
        reward[0:2,:] = np.random.randint(3,7, size = [2,10], dtype=int)
        reward[:,0:2] = np.random.randint(3, 7, size=[10,2], dtype=int)
        for i in range(10):
            reward[i,i] = np.random.randint(2,6)
        reward[7:10, 0:3] = np.random.randint(6,9, size = [3,3], dtype=int)
        reward[0:3, 7:10] = np.random.randint(6, 9, size=[3, 3], dtype=int)
        idx = np.random.randint(4,9)
        reward[idx:idx+2, idx:idx+2] = np.random.randint(8,15, size=[2, 2], dtype=int)
        return reward.reshape(100)

    def reward_design_0905pilot_2digits(self):
        reward = np.random.randint(1,30, size = [10,10], dtype=int)
        reward[0:2,:] = np.random.randint(22,50, size = [2,10], dtype=int)
        reward[:,0:2] = np.random.randint(22, 50, size=[10,2], dtype=int)
        for i in range(10):
            reward[i,i] = np.random.randint(15,50)
        reward[7:10, 0:3] = np.random.randint(40,80, size = [3,3], dtype=int)
        reward[0:3, 7:10] = np.random.randint(40, 80, size=[3, 3], dtype=int)
        idx = np.random.randint(4,9)
        reward[idx:idx+2, idx:idx+2] = np.random.randint(70,100, size=[2, 2], dtype=int)
        return reward.reshape(100)

    def reward_design_0906pilot(self):
        # reward = np.zeros([10*10], dtype=int)
        reward = np.random.randint(10,30, size = [10,10], dtype=int)
        reward[:2,:] = np.random.randint(10,40, size = [2,10], dtype=int)
        reward[8:, :] = np.random.randint(10, 40, size=[2, 10], dtype=int)
        reward[:, :2] = np.random.randint(10, 40, size=[10,2], dtype=int)
        reward[:, 8:] = np.random.randint(10, 40, size=[10,2], dtype=int)
        corner_value = np.array([0,1,2,3]) # one high, one low, two middle
        np.random.shuffle(corner_value)

        def get_cornerreward(idx):
            if idx == 0:
                return np.random.randint(80,100, size = [2,2], dtype=int)
            elif idx == 1:
                return np.random.randint(60,80, size = [2,2], dtype=int)
            elif idx == 2:
                return np.random.randint(60,80, size = [2,2], dtype=int)
            elif idx == 3:
                return np.random.randint(40,60, size = [2,2], dtype=int)
        def get_linereward(idx):
            if idx == 0:
                return np.random.randint(10,30, size = [3,3], dtype=int)
            elif idx == 1:
                return np.random.randint(10,30, size = [3,3], dtype=int)
            elif idx == 2:
                return np.random.randint(30,50, size = [3,3], dtype=int)
            elif idx == 3:
                return np.random.randint(30,50, size = [3,3], dtype=int)
        reward[0:2,0:2] = get_cornerreward(corner_value[0])
        reward[8:, 0:2] = get_cornerreward(corner_value[1])
        reward[0:2, 8:] = get_cornerreward(corner_value[2])
        reward[8:, 8:] = get_cornerreward(corner_value[3])
        reward[2:5, 2:5] = get_linereward(corner_value[0])
        reward[5:8, 2:5] = get_linereward(corner_value[1])
        reward[2:5, 5:8] = get_linereward(corner_value[2])
        reward[5:8, 5:8] = get_linereward(corner_value[3])
        return reward.reshape(100)

    def pilot_testcase_simulate_rsa(self, seed=1023):
        np.random.seed(seed)
        K = 100
        A = 4
        T = 20
        self.__init__(K, A, T)
        self.init_game()
        # self.P = self.init_P_for_2Dmove_nobound(K, A, 10, 10)
        self.P = self.init_P_for_2Dmove(K, A, 10, 10)
        rr = np.random.randint(1, 11, size=[K,A])
        self.Reward = np.zeros([self.T, self.Nstate, self.Naction])
        for t in range(T):
            self.Reward[t, :,:] = rr
        self.reward_ai = self.Reward.copy()
        return -1

    def init_P_for_2Dmove(self, Nstate, Naction, la, lb):
        if Naction != 4:  # not include stay in pilot
            print('wrong action for 2D setting ')
            print('check init_P_for_2Dmove')
            exit(1)
        else:
            P = np.zeros([Nstate, Naction, Nstate])
            la = int(la)
            lb = int(lb)
            if Nstate != la * lb:
                print('wrong a * b for 2D setting ')
                exit(2)
            fail_rate = 0.0 # fail rate to keep every state will be shown
            for s in range(Nstate):
                for a in range(Naction):  # 0=up 1=down 2=left 3=right
                    # 4=stary, not include in this case
                    P[s,a,:] = fail_rate/4
                    if a == 0:
                        if s < Nstate - la:
                            P[s, a, s + la] += 1-fail_rate
                        else:
                            P[s, a, :] = -1
                    elif a == 1:
                        if s >= la:
                            P[s, a, s - la]  += 1-fail_rate
                        else:
                            P[s, a, :] = -1
                    elif a == 2:
                        if np.mod(s, la) == 0:
                            P[s, a, :] = -1
                        else:
                            P[s, a, s - 1] += 1-fail_rate
                    elif a == 3:
                        if np.mod(s, la) == la - 1:
                            P[s, a, :] = -1
                        else:
                            P[s, a, s + 1] += 1-fail_rate
                    # elif a == 4:
                    #     P[s, a, s] = 1
                    else:
                        P[s, a, :] = -1
                        print('illegal action in init_P_for_2Dmove')
            return P

    def init_P_for_2Dmove_nobound(self, Nstate, Naction, la, lb):
        if Naction != 4:  # not include stay in pilot
            print('wrong action for 2D setting ')
            print('check init_P_for_2Dmove')
            exit(1)
        else:
            P = np.zeros([Nstate, Naction, Nstate])
            la = int(la)
            lb = int(lb)
            if Nstate != la * lb:
                print('wrong a * b for 2D setting ')
                exit(2)
            fail_rate = 0.0 # fail rate to keep every state will be shown
            for s in range(Nstate):
                for a in range(Naction):  # 0=up 1=down 2=left 3=right
                    # 4=stary, not include in this case
                    P[s,a,:] = fail_rate/4
                    if a == 0:
                        if s < Nstate - la:
                            P[s, a, s + la] += 1-fail_rate
                        else:
                            P[s, a, s + la - Nstate] += 1-fail_rate
                    elif a == 1:
                        if s >= la:
                            P[s, a, s - la]  += 1-fail_rate
                        else:
                            P[s, a, s - la +Nstate ] += 1-fail_rate
                    elif a == 2:
                        if np.mod(s, la) == 0:
                            P[s, a, s-1 + la] += 1-fail_rate
                        else:
                            P[s, a, s - 1] += 1-fail_rate
                    elif a == 3:
                        if np.mod(s, la) == la - 1:
                            P[s, a, s+1-la] += 1-fail_rate
                        else:
                            P[s, a, s + 1] += 1-fail_rate
                    # elif a == 4:
                    #     P[s, a, s] = 1
                    else:
                        P[s, a, :] = -1
                        print('illegal action in init_P_for_2Dmove')
            return P

    def simtestcase1_init_transition(self):
        fail_rate = 0.3
        fail_move = np.zeros([self.Nstate])
        self.P = np.zeros([self.Nstate,self.Naction,  self.Nstate]) # transition probability
        for s in range(self.Nstate):
            fail_move = np.zeros([self.Nstate])
            fail_move[s] = fail_rate / 3
            if s != self.Nstate - 1:
                fail_move[s+1] = fail_rate / 3
            else:
                fail_move[0] = fail_rate / 3
            if s == 0:
                fail_move[self.Nstate - 1] = fail_rate / 3
            else:
                fail_move[s - 1] = fail_rate / 3
            for a in range(self.Naction):
                pmove = np.zeros([self.Nstate])
                pmove[np.random.randint(self.Nstate,size=3)] = np.random.rand(3) + 0.1
                pmove = pmove / np.sum(pmove) * (1-fail_rate)
                self.P[s, a, :] = pmove + fail_move
                self.P[s, a, :] = self.P[s, a, :] / np.sum(self.P[s, a, :])
        return -1

    def print_info(self):
        pass

    def print_all_info(self):
        # optimal information for ai
        return self.Nstate, self.Naction, self.T, self.Reward.copy(), self.P.copy()

    def save_info(self, p = 'tmp_out'):
        np.save(p, [self.Nstate, self.T, self.P, self.Reward])

    def step_once(self, action = - 1):
        # print(self.cur_state,action)
        if action < 0 or action >= self.Naction:
            print('wrong action in step_once')
            exit(2)
        else:
            w = self.P[self.cur_state, action,:]
            x = np.random.rand()
            next_state = - 1
            for i in range(self.Nstate):
                if x <= w[i]:
                    next_state = i
                    break
                else:
                    x -= w[i]
            previous_state = self.cur_state
            self.cur_state = next_state
            self.sum_reward += self.Reward[self.cur_t, self.cur_state]
            self.cur_t += 1
            return self.cur_state, self.Reward[self.cur_t-1, previous_state],\
                   self.sum_reward, self.cur_t

    def print_state(self):
        return self.cur_state

    def tmp_funs(a,v,c):
        pass
