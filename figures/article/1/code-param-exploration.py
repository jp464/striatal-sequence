import sys
sys.path.insert(0, '../../../network')
import logging
import argparse
import numpy as np
from network import Population, RateNetwork
from learning import ReachingTask
from transfer_functions import ErrorFunction
from connectivity import SparseConnectivity, LinearSynapse, ThresholdPlasticityRule, set_connectivity, cmatrix
from sequences import GaussianSequence
import matplotlib.pyplot as plt
import seaborn as sns

filename, A0, A1, A2, A3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
filename = filename + '-' + A0 + '-' + A1 + '-' + A2 + '-' + A3
A0, A1, A2, A3 = float(A0), float(A1), float(A2), float(A3)
#=======================INITIALIZATION=======================
# Pop size
N_ctx, N_d1 = 3000, 3000
N = [N_ctx, N_d1]

# Gaussian patterns
S, P = 1, 4
sequences_ctx = [GaussianSequence(P,N_ctx, seed=1) for i in range(S)]
patterns_ctx = np.stack([s.inputs for s in sequences_ctx])
sequences_d1 = [GaussianSequence(P,N_d1, seed=41) for i in range(S)]
patterns_d1 = np.stack([s.inputs for s in sequences_d1])

sequences = np.array([sequences_ctx, sequences_d1], dtype=object)
patterns = np.array([patterns_ctx, patterns_d1], dtype=object)

###### Learning rule: 
# Type 0: zeros; 1: symmetric; 2: asymmetric
A00, A01 = cmatrix((P,P), A0, 1), cmatrix((P,P), A1, 0)
A0 = np.array([A00,A01])


A10, A11, = cmatrix((P,P), A2, 1), cmatrix((P,P), A3, 1) #.15
A1 = np.array([A10,A11])


A = np.array([A0,A1])

# Connection probabilities
cp = np.array([[0.1, 0.1],
               [0.1, 0.1]])

cw = np.array([[0,1],
                [0,-1]])

phi = ErrorFunction(mu=0.22, sigma=0.1).phi
plasticity = ThresholdPlasticityRule(x_f=0.5, q_f=0.8)

# populations
ctx = Population(N=N[0], tau=1e-2, phi=phi, name='ctx')
d1 = Population(N=N[1], tau=1e-2, phi=phi, name='d1')
# d2 = Population(N=N[2], tau=1e-2, phi=phi, name='d2')

J = set_connectivity([ctx, d1], cp, cw, A, patterns, plasticity)
network = RateNetwork([ctx, d1], J, formulation=4, disable_pbar=False)

#=======================SIMULATION=======================
init_inputs = [np.zeros(ctx.size),
               np.zeros(d1.size)]
input_patterns = [p[0] for p in patterns]
T=8 #ms
mouse = ReachingTask()
network.simulate_learning(mouse, T, init_inputs, input_patterns, plasticity, 
                          delta_t=300, eta=0.05, tau_e=800, lamb=0.3, 
                          noise=[0.13,0.13,0.13], a_cf=0, e_bl = [0.05,0.003,0.045,0.07], 
                          alpha=0, gamma=0, etrace=False, #[0.05,0.003,0.045,0.07]
                          hyper=False, r_ext=[lambda t:0, lambda t: .5], print_output=False)

#=======================SAVE=======================
overlaps_ctx = sequences[0][0].overlaps(network.pops[0])
overlaps_d1 = sequences[1][0].overlaps(network.pops[1])
np.savez('/work/jp464/striatum-sequence/exploration/' + filename + '.npz', 
         overlaps_ctx=overlaps_ctx, overlaps_d1=overlaps_d1)
