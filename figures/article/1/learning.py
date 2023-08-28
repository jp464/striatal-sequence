import sys
sys.path.insert(0, '../../../network')

import logging
import argparse
import numpy as np
from network import Population, RateNetwork
from learning import ReachingTask
from transfer_functions import ErrorFunction
from connectivity import SparseConnectivity, LinearSynapse, ThresholdPlasticityRule, set_connectivity 
from sequences import GaussianSequence
import matplotlib.pyplot as plt
import seaborn as sns
logging.basicConfig(level=logging.INFO)

### Input 
filename = sys.argv[1]

### Set up network
phi = ErrorFunction(mu=0.22, sigma=0.1).phi
plasticity = ThresholdPlasticityRule(x_f=0.5, q_f=0.8)

# populations
ctx = Population(N=1000, tau=1e-2, phi=phi, name='ctx')
d1 = Population(N=1000, tau=1e-2, phi=phi, name='d1')
d2 = Population(N=1000, tau=1e-2, phi=phi, name='d2')
pops = np.array([ctx, d1])

# patterns 
S, P = 1, 3
sequences_ctx = [GaussianSequence(P,ctx.size, seed=114) for i in range(S)]
patterns_ctx = np.stack([s.inputs for s in sequences_ctx])
sequences_d1 = [GaussianSequence(P,d1.size, seed=29) for i in range(S)]
patterns_d1 = np.stack([s.inputs for s in sequences_d1])
patterns = [patterns_ctx, patterns_d1]

# connectivity probabilities
cp = np.array([[0.05,  0.05], 
               [0.05, 0.05]])
cw = np.array([[0, 0],
               [0, 0]])
A = np.array([[5, 1],
             [0, 5]])
plasticity_rule = np.array([[0, 0],
                          [1, 0]])
J = set_connectivity(pops, cp, cw, A, plasticity_rule, patterns, plasticity)

network = RateNetwork(pops, J, formulation=4)

### Simiulation
init_input_ctx = np.random.RandomState().normal(0,1,size=patterns_ctx[0][0].shape)
init_input_d1 = np.random.RandomState().normal(0,1,size=patterns_d1[0][0].shape)
T=500 #ms
mouse = ReachingTask()
network.simulate_learning(mouse, T, init_input_ctx, init_input_d1, 
                          patterns_ctx[0], patterns_d1[0], plasticity, 
                          delta_t=500, eta=0.0005, tau_e=1600, lamb=0.5, 
                          noise1=.13, noise2=.13, etrace=True, print_output=False)

### Save
overlaps_ctx = sequences_ctx[0].overlaps(network.pops[0])
overlaps_d1 = sequences_d1[0].overlaps(network.pops[1])
correlations_ctx = sequences_ctx[0].overlaps(network.pops[0], correlation=True)
correlations_d1 = sequences_d1[0].overlaps(network.pops[1], bg, correlation=True)
np.savez('/work/jp464/striatum-sequence/' + filename + '.npz', 
         overlaps_ctx=overlaps_ctx, overlaps_d1=overlaps_d1, 
         correlations_ctx=correlations_ctx, correlations_d1=correlations_d1, 
         state_ctx=network.pops[0].state, state_d1=network.pops[1].state)
