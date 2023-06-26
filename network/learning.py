import logging
import numpy as np
from tqdm import trange

logger = logging.getLogger(__name__)

class Learning(object):
    def __init__(self, goal, alpha, gamma):
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma 
        self.Qvalues = np.array((goal, goal))
        self.in_seq = False
    
    def get_actions(self, s, actions):
        return actions[s]
        
class ReachingTask(Learning):
    def __init__(self, goal, alpha=0.2, gamma=0.9):
        super().__init__(goal, alpha, gamma)
        self.states = ['out', 'in']
        self.actions = {'out': [0, 2], 'in': [0, 1, 2, 3]}
        self.Qvalues = np.zeros((2, 3))
    
    def state_transition(s, a, w):
        
        if s == 'out':
            if a == 2:
                return ('out', False)
            else:
                return ('in', False)
        if s == 'out':
            if a == 3:
                return ('out', w)
            elif a == 1:
                return ('in', True)
            else:
                return ('in', False)
        
    def detect_reward(self, s0, s1):
        # Check if in sequence
        self.in_seq = (self.in_seq and (s1 == s0 or s1 == s0+1 or s1 == -1)) or s1 == 0
            
        # Determine reward 
        if self.in_seq and s1 == self.goal and s1 == s0+1:
            self.in_seq = False
            return 1
        elif s1 == self.goal:
            return -1
        return 0
    
    def td_learning():
        
                
                