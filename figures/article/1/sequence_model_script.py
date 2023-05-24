import sys
sys.path.insert(0, '../../../network')
import logging
import argparse
import numpy as np

from network import Population, RateNetwork
from transfer_functions import ErrorFunction
from connectivity import SparseConnectivity, LinearSynapse, ThresholdPlasticityRule
from sequences import GaussianSequence
import matplotlib.pyplot as plt 
from scipy import stats
from openpyxl import load_workbook
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO)

#===========================================
# Initial Parameters
#===========================================
### Transfer function
phi_mu = 0
phi_sigma = 0.2

### Cortex 
N_ctx = 10000
S_ctx, P_ctx = 1, 10

### BG 
N_bg = 10000
S_bg, P_bg = 1, 10

### Network
sparsity_cc, sparsity_bc, sparsity_bb, sparsity_cb = 0.05, 0.05, 0.05, 0.05

### Transfer function
phi = ErrorFunction(mu=phi_mu, sigma=phi_sigma).phi

### Cortical network
ctx = Population(N=N_ctx, tau=1e-2, phi=phi, name="ctx")
sequences_ctx = [GaussianSequence(P_ctx, ctx.size, seed=11) for i in range(S_ctx)]
patterns_ctx = np.stack([s.inputs for s in sequences_ctx])

### BG network
bg = Population(N=N_bg, tau=1e-2, phi=phi, name="bg")
sequences_bg = [GaussianSequence(P_bg, bg.size, seed=367) for i in range(S_bg)]
patterns_bg = np.stack([s.inputs for s in sequences_bg])

T=1
dt = 1e-3
t = np.arange(0, T, dt)

#===========================================
# Build Netowrk
#===========================================
def build_network(A_cc, A_bc, A_bb, A_cb):
    plasticity = ThresholdPlasticityRule(x_f=0, q_f=0.5)

    ### J_cc: attractors
    J_cc = SparseConnectivity(source=ctx, target=ctx, p=sparsity_cc)
    synapse_cc = LinearSynapse(J_cc.K, A=A_cc)
    J_cc.store_attractors(patterns_ctx[0], synapse_cc.h_EE, plasticity.f, plasticity.g)

    ### J_bc: sequences 
    J_bc = SparseConnectivity(source=ctx, target=bg, p=sparsity_bc)
    synapse_bc = LinearSynapse(J_bc.K, A=A_bc)
    J_bc.store_sequences([patterns_bg, patterns_ctx], synapse_bc.h_EE, plasticity.f, plasticity.g, seq=True)

    ### J_bb: attractors
    J_bb = SparseConnectivity(source=bg, target=bg, p=sparsity_bb)
    synapse_bb = LinearSynapse(J_bb.K, A=A_bb)
    J_bb.store_attractors(patterns_bg[0], synapse_bb.h_EE, plasticity.f, plasticity.g)

    ### J_cb: sequences 
    J_cb = SparseConnectivity(source=bg, target=ctx, p=sparsity_cb)
    synapse_cb = LinearSynapse(J_cb.K, A=A_cb)
    J_cb.store_sequences([patterns_ctx, patterns_bg], synapse_cb.h_EE, plasticity.f, plasticity.g, seq=False)

    return J_cc, J_bc, J_bb, J_cb

def simulate(J_cc, J_bc, J_bb, J_cb):
    ### Initial input for simulation
    init_input_ctx = patterns_ctx[0, 0, :]
    init_input_bg = patterns_bg[0, 0, :]

    from tqdm import tqdm, trange
    net_ctx = RateNetwork(ctx, c_EE=J_cc, c_EI=J_cb, formulation=1)
    net_bg = RateNetwork(bg, c_EE=J_bb, c_EI=J_bc, formulation=1)


    fun_ctx = net_ctx._fun(T)
    fun_bg = net_bg._fun(T)

    r0 = ctx.phi(init_input_ctx)
    r1 = bg.phi(init_input_bg)


    state_ctx = np.zeros((len(t)+1, N_ctx))
    state_bg = np.zeros((len(t)+1, N_bg))
    for i in trange(len(t)):
        r0 += fun_ctx(T, r0, r1)*dt
        r1 += fun_bg(T, r1, r0)*dt
        state_ctx[i] = r0
        state_bg[i] = r1

    return state_ctx, state_bg

def find_corr(state, phi_patterns):
    corr = []
    for p in phi_patterns[0]:
        corr.append([stats.pearsonr(state[i], p)[0] for i in range(int(T/dt))])
    return corr

def maximize(A, values):
    maximum = -float('inf')
    ind = -1
    for i in A:
        if values[i] > maximum:
            maximum = values[i]
            ind = i
    return maximum, ind

def average(S, length=len(patterns_ctx[0])):
    tot = np.sum(S)
    return tot / length

def overlap(corr, patterns):
    peaks_index = [0]
    peaks = [1]
    for i in range(len(patterns)):
        if i == 0:
            continue
        else:
            try:
                ind = find_peaks(corr[i], height=0.5)[0][0]
                val = corr[i][ind]
                peaks_index.append(ind)
                peaks.append(val)
            except:
                continue
    return peaks, peaks_index

def output(A_cc, A_bc, A_bb, A_cb):
    phi_patterns_ctx = phi(patterns_ctx)
    corr_ctx = find_corr(state_ctx, phi_patterns_ctx)

    phi_patterns_bg = phi(patterns_bg)
    corr_bg = find_corr(state_bg, phi_patterns_bg)

    lim = 1000
    fig, axes = plt.subplots(2, figsize=[8,10])

    lim = lim - T
    for i, v in enumerate(corr_ctx):
        axes[0].plot(t[:lim], v[:lim], label=str(i))
        axes[0].legend()
        axes[0].set_title("Cortex")

    for i, v in enumerate(corr_bg):
        axes[1].plot(t[:lim], v[:lim], label=str(i))
        axes[1].legend()
        axes[1].set_title("BG")
    fig.supxlabel('Time (s)')
    fig.supylabel('Correlation')
    fig.tight_layout()
    # plt.show()
    plt.savefig("../../output/" + str(round(A_cc, 1)) + "," + str(round(A_bc, 1)) + "," + str(round(A_bb, 1)) + "," + str(round(A_cb, 1)) + ".jpg", bbox_inches='tight', dpi=300)

    peaks_ctx, peaks_ind_ctx = overlap(corr_ctx, patterns_ctx[0])
    peaks_bg, peaks_ind_bg = overlap(corr_bg, patterns_bg[0])
    peak_intervals_ctx = [peaks_ind_ctx[i+1]-peaks_ind_ctx[i] for i in range(len(peaks_ind_ctx)-1)]
    peak_intervals_bg = [peaks_ind_bg[i+1]-peaks_ind_bg[i] for i in range(len(peaks_ind_bg)-1)]

    corr_ctx, corr_bg = average(peaks_ctx), average(peaks_bg)
    spd_ctx, spd_bg = np.mean(peak_intervals_ctx), np.mean(peak_intervals_bg)

    return corr_ctx, spd_ctx, corr_bg, spd_bg



if __name__ == "__main__":
    wb = load_workbook(filename="/Users/stan.park712/MyDrive/striatum/hebbian_sequence_learning/figures/output/learning_rate_data.xlsx")
    sheet = wb.active
    As = [[3.4, i, 0, i] for i in np.arange(1.0, 5.2, 0.2)]
    for A in As:
        Acc, Abc, Abb, Acb = A
        J_cc, J_bc, J_bb, J_cb = build_network(Acc, Abc, Abb, Acb)
        state_ctx, state_bg = simulate(J_cc, J_bc, J_bb, J_cb)
        corr_ctx, spd_ctx, corr_bg, spd_bg = output(Acc, Abc, Abb, Acb)
        sheet.append([Acc, Abc, Abb, Acb, corr_ctx, spd_ctx, corr_bg, spd_bg])
        wb.save(filename="/Users/stan.park712/MyDrive/striatum/hebbian_sequence_learning/figures/output/learning_rate_data.xlsx")


