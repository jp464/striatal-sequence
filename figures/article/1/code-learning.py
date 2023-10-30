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
filename, delta_t, eta, tau_e, lamb, alpha, gamma = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
filename = filename + '-' + delta_t + '-' + eta + '-' + tau_e + '-' + lamb + '-' + alpha + '-' + gamma
delta_t, eta, tau_e, lamb, gamma, alpha = int(delta_t), float(eta), float(tau_e), float(lamb), float(gamma), float(alpha)

# Load parameters
params = np.load("./ctx_str_params.npz", allow_pickle=True) 
N, sequences, patterns, cp, cw, A = params['N'], params['sequences'], params['patterns'], params['cp'], params['cw'], params['A']

phi = ErrorFunction(mu=0.22, sigma=0.1).phi
plasticity = ThresholdPlasticityRule(x_f=0.5, q_f=0.8)

# populations
ctx = Population(N=N[0], tau=1e-2, phi=phi, name='ctx')
d1 = Population(N=N[1], tau=1e-2, phi=phi, name='d1')

J = set_connectivity([ctx, d1], cp, cw, A, patterns, plasticity)
network = RateNetwork([ctx, d1], J, formulation=4, disable_pbar=False)

### Simiulation
init_inputs = [np.zeros(ctx.size),
               np.zeros(d1.size)]
input_patterns = [p[0] for p in patterns]

T=15 #ms
mouse = ReachingTask()
network.simulate_learning(mouse, T, init_inputs, input_patterns, plasticity, 
                          delta_t=delta_t, eta=eta, tau_e=tau_e, lamb=lamb, 
                          noise=[0.13,0.13,0.13], e_bl = [0.06,0.031,0.04,0.07],
                          alpha=alpha, gamma = gamma, adap=0, env=2.4, etrace=True, 
                          r_ext=[lambda t:0, lambda t: .5], print_output=False)

### Save
overlaps_ctx = sequences[0][0].overlaps(network.pops[0])
overlaps_d1 = sequences[1][0].overlaps(network.pops[1])
np.savez('/work/jp464/striatum-sequence/' + filename + '.npz', 
         overlaps_ctx=overlaps_ctx, overlaps_d1=overlaps_d1,
         state_ctx=network.pops[0].state, state_d1=network.pops[1].state,
         behaviors=mouse.behaviors, values=mouse.values, rpes=mouse.rpes)
