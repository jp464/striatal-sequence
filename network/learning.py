import logging
import numpy as np
from tqdm import trange

logger = logging.getLogger(__name__)

class Learning(object):
    def __init__(self, goal, alpha, gamma):
        self.goal = goal
        self.alpha = alpha
        self.gamma = gamma 
        self.reward = 0
        self.Qvalues = np.array((goal, goal))
        self.in_seq = False
    
    def get_actions(self, s, actions):
        return actions[s]
        
class ReachingTask(Learning):
    def __init__(self, goal, alpha=0.2, gamma=0.9, reward=0):
        super().__init__(goal, alpha, gamma)
        self.states = ['out', 'in']
        self.actions = ['aim', 'reach', 'lick', 'back']
        self.actions = {'out': ['aim', 'reach', 'lick', 'back'], 
                        'in': ['aim', 'reach', 'lick', 'back']}
        self.Qvalues = np.zeros((2, 2, 4))
    
    def state_transition(self, s, a, w):
        if s == 'out':
            if a == 0:
                return 'in', 0
            elif a == 3:
                return 'out', w
            else:
                return 'out', 0
        if s == 'in':
            if a == 3:
                return 'out', w
            elif a == 1:
                return 'in', 1
            else:
                return 'in', 0
        
    def detect_reward(self, s, a, w, reward=1, penalty=-.3):
        if a == 2 and w:
            self.reward = reward
        elif a == 2:
            self.reward = penalty
        else:
            self.reward = -.0001
        
    def td_learning(self, s0, a0, w0, s1, a1, w1):
        s0 = self.states.index(s0)
        s1 = self.states.index(s1)
        Q0 = self.Qvalues[w0,s0,a0]
        Q1 = self.Qvalues[w1,s1,a1]
        self.Qvalues[w0,s0,a0] =  Q0 + self.alpha * (self.reward + self.gamma * Q1 - Q0)
        
                
                