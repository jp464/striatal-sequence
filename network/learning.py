import logging
import numpy as np
from tqdm import trange
from transfer_functions import erf

logger = logging.getLogger(__name__)

class Learning(object):
    def __init__(self):
        self.actions = None
        self.behaviors1 = None
        self.behaviors2 = None
        self.action_dur1 = 0
        self.action_dur2 = 0
        self.transitions1 = np.array([])
        self.transitions2 = np.array([])
    
    def get_action(self, s):
        return self.actions[s]
    
class ReachingTask(Learning):
    def __init__(self):
        super().__init__()
        self.actions = ['aim', 'reach', 'lick', 'scavenge', 'null']
        self.w = 0
        self.water_left = -1
        self.fullness = 0
    
    # Detect whether mouse has acquired water
    def water(self, a0, a1, water_left=300):
        if a0 == 'aim' and a1 == 'reach':
            self.water_left = water_left
            self.w = 1
        elif a0 == 'reach' and a1 == 'lick':
            return 
        elif a0 == a1:
            return
        else:
            self.w = 0
    
    # Detect whether mouse received reward
    def compute_reward(self, a, reward=1):
        if self.water_left == -1:
            self.reward = 0
        elif self.water_left == 0:
            self.w = 0
            self.reward = reward
            self.water_left -= 1
        elif a == 'lick' and self.w:
            self.water_left -= 1
        else:
            self.reward = 0
    
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
            if self.reward == 1:
                success += 1
            cnt += 1
            if cnt % interval == 0:
                performance.append(success/(interval*0.001))
                cnt = 0
                success = 0
        return performance
    
      
class NetworkUpdateRule(object):
    def __init__(self):
        self.f = lambda x,y: (x + y)/2  
        self.rectifier = lambda x: 0 if x < 0 else x
        
                
                