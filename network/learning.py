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
    
    def get_action(self, s):
        return self.actions[s]
    
    def compute_Qval(self, uc, f=lambda x,y:x+y, rectifier=lambda x:x):
        Q = f(self.Qvalues[uc[0][2],self.states.index(uc[0][0]),uc[0][1]],
              self.Qvalues[uc[1][2],self.states.index(uc[1][0]),uc[1][1]])
        return rectifier(Q)
    
class ReachingTask(Learning):
    def __init__(self, goal, alpha=0.2, gamma=0.9, reward=0, f=lambda x:erf(x/2)):
        super().__init__(goal, alpha, gamma)
        self.states = ['out', 'in']
        self.actions = ['aim', 'reach', 'retract', 'lick']
        self.legal_actions = {'out': [0, 1, 2, 3], 
                        'in': [0, 1, 2, 3]}
        self.Qvalues = np.zeros((2, 2, 4))
    
    def state_transition(self, s, a, w):
        if s == 'out':
            if a == 0:
                return 'in', 0
            elif a == 4:
                return 'out', w
            else:
                return 'out', 0
        if s == 'in':
            if a == 4:
                return 'out', w
            elif a == 1:
                return 'in', 1
            elif a == 2:
                return 'in', w
            else:
                return 'in', 0
        
    def detect_reward(self, s, a, w, reward=1, penalty=-.3):
        if a == 3 and w:
            self.reward = reward
        elif a == 3:
            self.reward = penalty
        elif a == 1 and s == 'in':
            self.reward = .3 * reward
        else:
            self.reward = -.0001
        
    def td_learning(self, s0, a0, w0, s1, a1, w1, f=lambda x:erf(x/2)):
        s0 = self.states.index(s0)
        s1 = self.states.index(s1)
        Q0 = self.Qvalues[w0,s0,a0]
        Q1 = self.Qvalues[w1,s1,a1]
        self.Qvalues[w0,s0,a0] =  f(Q0 + self.alpha * (self.reward + self.gamma * Q1 - Q0))

class NetworkUpdateRule(object):
    def __init__(self):
        self.f = lambda x,y: 0.1 * x + y  
        self.rectifier = lambda x: 0 if x < 0 else x
        
                
                