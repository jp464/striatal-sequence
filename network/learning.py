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
        self.behaviors1 = None
        self.behaviors2 = None
        self.action_dur1 = 0
        self.action_dur2 = 0
    
    def get_action(self, s):
        return self.actions[s]
    
class ReachingTask(Learning):
    def __init__(self, goal, alpha=0.2, gamma=0.9, reward=0):
        super().__init__(goal, alpha, gamma)
        self.actions = ['aim', 'reach', 'lick', 'scavenge', 'null']
        self.Qvalues = np.zeros((2, 4))
        self.w = 0
        self.fullness = 0
    
    def water(self, a0, a1):
        if a0 == 'aim' and a1 == 'reach':
            return 1 
        if a0 == 'reach' and a1 == 'lick':
            return self.w
        if a0 == a1:
            return self.w
        return 0 
        
    def compute_reward(self, a, reward=1, penalty=-.05):
        if a == 'lick' and self.w:
            self.reward = reward
            self.w = 0
        elif a == 'lick':
            self.reward = penalty
        else:
            self.reward = 0
    
    
      
class NetworkUpdateRule(object):
    def __init__(self):
        self.f = lambda x,y: (x + y)/2  
        self.rectifier = lambda x: 0 if x < 0 else x
        
                
                