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
from scipy.signal import find_peaks
import pandas as pd
import h5py

filename, A0, A1, A2, A3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
filename = filename + '-' + A0 + '-' + A1 + '-' + A2 + '-' + A3
A0, A1, A2, A3 = float(A0), float(A1), float(A2), float(A3)
#=======================HELPER FUNCTIONS=======================
def fpeaks(overlaps):
    peaks = np.array([])
    for m_p in overlaps:
        peaks = np.append(peaks, find_peaks(m_p, height=.3, prominence=.05)[0])
    peaks.sort()
    return peaks 

def retrieval_speed(overlaps, tau=10):
    peaks = fpeaks(overlaps)
    if len(peaks) == 0:
        return 0
    if len(peaks) == 1:
        return -1
    sum = 0
    for i in range(1, len(peaks)):
        sum += (peaks[i] - peaks[i-1])
    return tau / (sum / (len(peaks)-1))

def plot_peaks(overlaps):
    peaks = fpeaks(overlaps)
    const = [.5 for i in peaks]
    for m in overlaps:
        plt.plot(m)
    plt.scatter(peaks, const, color='r')
    plt.show()
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
A10, A11, = cmatrix((P,P), A2, 1), cmatrix((P,P), A3, 1) #.15
A = np.array([np.array([A00,A01]),
              np.array([A10,A11])])
# Connection probabilities
cp = np.array([[0.05, 0.05],
               [0.05, 0.05]])

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
init_inputs = [patterns[0][0][3],
               np.zeros(d1.size)]
input_patterns = [p[0] for p in patterns]
T=8 #ms
mouse = ReachingTask()
network.simulate_learning(mouse, T, init_inputs, input_patterns, plasticity, 
                          delta_t=300, eta=0.05, tau_e=800, lamb=0.3, 
                          noise=[0.13,0.13,0.13], e_bl = [0.055,0.0022,0.04,0.07], 
                          alpha=0, gamma=0, adap=0, env=0, etrace=False, #[0.05,0.003,0.045,0.07]
                          r_ext=[lambda t:0, lambda t: .5], print_output=False)

#=======================SAVE=======================
overlaps_ctx = sequences[0][0].overlaps(network.pops[0])
overlaps_d1 = sequences[1][0].overlaps(network.pops[1])
v_ctx, v_d1 = retrieval_speed(overlaps_ctx), retrieval_speed(overlaps_d1) 

def detect_seq(pre, post):
    if pre == 3:
        if post != 0: return -1
    elif post - pre != 1: return -1
    if 0 == post: return 1
    return 0
    
behavior = mouse.behaviors[0][1:]
score = 0
for i in range(len(behavior)):
    pre, post = behavior[i], behavior[i+1]
    if post == None: break
    else: pre, post = pre[0], post[0]
    score += detect_seq(pre, post)

seq = score > 0 and v_ctx > 0
if not seq and max(overlaps_ctx[:,-1]) > 0.3: att = True
else: att = False
    
df = pd.read_hdf('/work/jp464/striatum-sequence/output/retrieval_speed.h5', 'data')
df.loc[len(df)] = [A0, A1, A2, A3, v_ctx, v_d1, seq, att]
df.to_hdf('/work/jp464/striatum-sequence/output/retrieval_speed.h5', 'data')