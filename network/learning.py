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
        self.transitions1 = np.array([])
        self.transitions2 = np.array([])
    
    def get_action(self, s):
        return self.actions[s]
    
class ReachingTask(Learning):
    def __init__(self, goal, alpha=0.2, gamma=0.9, reward=0):
        super().__init__(goal, alpha, gamma)
        self.actions = ['aim', 'reach', 'lick', 'scavenge', 'null']
        self.Qvalues = np.zeros((2, 4))
        self.w = 0
        self.water_left = -1
        self.fullness = 0
    
    def water(self, a0, a1):
        if a0 == 'aim' and a1 == 'reach':
            self.water_left = 200
            self.w = 1
        elif a0 == 'reach' and a1 == 'lick':
            return 
        elif a0 == a1:
            return
        else:
            self.w = 0
        
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
            
    def calibrate(self):
        transitions1 = [self.transitions1[i] for i in range(len(self.transitions1)) 
               if self.transitions1[i] - self.transitions1[i-1] > 10]
        transitions2 = [self.transitions2[i] for i in range(len(self.transitions2)) 
               if self.transitions2[i] - self.transitions2[i-1] > 10]
        transitions1 = transitions1[2:]
        transitions2 = transitions2[:-2]
        delta_t = np.mean(np.subtract(transitions1, transitions2))
        
        eta = 0
        tau_e = 0
        for i in range(len(transitions1)):
            if i % 2 == 1:
                eta += (transitions1[i] - transitions1[i-1]) + (transitions2[i] - transitions2[i-1])
            if i % 8 == 7:
                tau_e += (transitions1[i] - transitions1[i-7]) + (transitions2[i] - transitions2[i-7])
        eta = len(transitions1) / eta
        tau_e = tau_e / (len(transitions1) / 7)
        
        print("tau_e: " + str(tau_e) + " eta: " + str(eta) + " delta_t: " + str(delta_t))
        return tau_e, eta, delta_t
                
        
    
    
      
class NetworkUpdateRule(object):
    def __init__(self):
        self.f = lambda x,y: (x + y)/2  
        self.rectifier = lambda x: 0 if x < 0 else x
        
                
                