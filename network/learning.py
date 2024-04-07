import logging
import numpy as np
from tqdm import trange
from transfer_functions import erf

logger = logging.getLogger(__name__)

class Learning(object):
    def __init__(self):
        self.actions = None
        self.behaviors = None
        self.transitions = None 
        self.values = None
        self.repes = None
    
class ReachingTask(Learning):
    def __init__(self):
        super().__init__()
        self.barrier = 0
        self.water = 0
        self.action = 0
        self.action_dur = 0
        self.reward = 0
        self.drank_water = 0
    
    # Detect whether mouse has acquired water
    def compute_barrier(self, a):
        if a == 0:
            self.barrier = 1
        elif self.barrier == 1 and (self.water == 0 and (a == 1 or a == 2)):
            pass
        else:
            self.barrier = 0
            
    def compute_water(self, a):
        if self.barrier == 1 and a == 1:
#             self.w = np.random.binomial(1, .9, 1)[0]
            self.water = 1
        else: self.water = 0
    
    # Detect whether mouse received reward
    def compute_reward(self, a, reward=1):
        if self.water == 1 and a == 2:
            self.reward = reward
        else:
            self.reward = 0

            
    # Environmental variabe
    def env(self, C, patterns):
        ret = np.zeros(len(patterns[0]))
        for i,v in enumerate(C):
            ret = ret + v * patterns[i]
        return ret.astype('float64')

    # Compute the reward rate over sessiosn 
    def learning_performance(self, correlations, interval):
        correlations = np.transpose(correlations)
        a0 = None
        a1 = None
        cnt = 0
        success = 0
        performance = [0]
        for i in range(len(correlations)):
            cur = self.get_action(np.argmax(correlations[i]))
            if cur != a1: 
                a0, a1 = a1, cur
                self.water(a0, a1)
            self.compute_reward(cur)
            if self.r == 1:
                success += 1
            cnt += 1
            if cnt % interval == 0:
                performance.append(success/(interval*0.001))
                cnt = 0
                success = 0
        return performance
    
    def compute_error_value(self, i, s0, s1, bs, ws, rs, value, rpe, gamma=1, alpha=1):
        s0, s1 = self.actions.index(s0), self.actions.index(s1)
        idx0, idx1 = int(8 * ws[i-1] + 4 * bs[i-1]), int(8 * ws[i] + 4 * bs[i])
        
        prediction = value[idx0+s0]
        outcome = rs[i] + gamma*value[idx1+s1]
#         print(prediction, outcome)
        rpe[idx1+s1] = outcome - prediction 
        
        if s0 == 3:
            pass
        else:
            value[idx0+s0] = value[idx0+s0] + alpha * (rs[i] + gamma * value[idx1+s1] - value[idx0+s0])
#         elif rs[i-1] == 1:
#             value[idx0+s0] = rs[i-1]
#         else:
#             value[idx0+s0] = rs[i-1] + gamma * value[idx1+s1]
        self.rpes[:,i] = rpe
        self.values[:,i] = value
        
        return rpe, value
        
      
class NetworkUpdateRule(object):
    def __init__(self):
        self.f = lambda x,y: (x + y)/2  
        self.rectifier = lambda x: 0 if x < 0 else x
        
                
                