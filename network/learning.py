import logging
import numpy as np
from tqdm import trange
from transfer_functions import erf

logger = logging.getLogger(__name__)

class Learning(object):
    def __init__(self, goal, alpha, gamma):
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma 
        self.reward = 0
        self.in_seq = False
        self.actions = None
        self.Qvalues = None
        self.behaviors = None
        self.behaviors_raw = None
        self.action_dur = 0
    
    def get_action(self, s):
        return self.actions[s]
    
    def compute_Qval(self, uc, f=lambda x,y:x+y, rectifier=lambda x:x):
        a0, a1, w0, w1 = uc[0], uc[1], uc[2], uc[3]
        Q = f(self.Qvalues[w0, a0],
              self.Qvalues[w1, a1])
        return rectifier(Q)
    
class ReachingTask(Learning):
    def __init__(self, goal, alpha=0.2, gamma=0.9, reward=0):
        super().__init__(goal, alpha, gamma)
        self.actions = ['aim', 'reach', 'lick', 'scavenge', 'null']
        self.Qvalues = np.zeros((2, 4))
        self.w = 0
    
    def water(self, a0, a1):
        if a0 == 'aim' and a1 == 'reach':
            return 1 
        return 0 
        
    def compute_reward(self, a, reward=1, penalty=-.05):
        if a == 'lick' and self.w:
            self.reward = reward
            self.w = 0
        elif a == 'lick':
            self.reward = penalty
        else:
            self.reward = 0
        
    def td_learning(self, a0, a1, w0, w1, f=lambda x:x):
        Q0, Q1 = self.Qvalues[w0,a0], self.Qvalues[w1,a1]
        self.Qvalues[w0,a0] =  f(Q0 + self.alpha * (self.reward + self.gamma * Q1 - Q0))
    
    
      
class NetworkUpdateRule(object):
    def __init__(self):
        self.f = lambda x,y: (x + y)/2  
        self.rectifier = lambda x: 0 if x < 0 else x
        
                
                