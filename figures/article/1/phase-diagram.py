import sys
sys.path.insert(0, '../../../network')
import logging
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from network import Population, RateNetwork
from transfer_functions import ErrorFunction
from connectivity import SparseConnectivity, LinearSynapse, ThresholdPlasticityRule
from sequences import GaussianSequence
from learning import ReachingTask
import numpy as np

logging.basicConfig(level=logging.INFO)

As = []
for i in np.arange(2.5, 10, .1):
    As += [[i,i,1,j] for j in np.arange(0, 5, .2)]

for i in As:
    print(i)
    Acc, Abb, Acb, Abc = round(i[0], 1), round(i[1], 1), round(i[2], 1), round(i[3], 1)
    phi = ErrorFunction(mu=0.22, sigma=0.1).phi
    ctx = Population(N=1000, tau=1e-2, phi=phi, name='ctx')
    bg = Population(N=1000, tau=1e-2, phi=phi, name='bg')
    plasticity = ThresholdPlasticityRule(x_f=0.5, q_f=0.8) 

    S, P = 1, 3
    sequences_ctx = [GaussianSequence(P,ctx.size, seed=114) for i in range(S)]
    patterns_ctx = np.stack([s.inputs for s in sequences_ctx])
    sequences_bg = [GaussianSequence(P,ctx.size, seed=29) for i in range(S)]
    patterns_bg = np.stack([s.inputs for s in sequences_bg])

    J_cc = SparseConnectivity(source=ctx, target=ctx, p=0.05)
    synapse_cc = LinearSynapse(J_cc.K, A=Acc)
    J_cc.store_attractors(patterns_ctx[0], patterns_ctx[0], synapse_cc.h_EE, 
                        plasticity.f, plasticity.g)
    J_bb = SparseConnectivity(source=bg, target=bg, p=0.05)
    synapse_bb = LinearSynapse(J_bb.K, A=Abb)
    J_bb.store_attractors(patterns_bg[0], patterns_bg[0], synapse_bb.h_EE, 
                        plasticity.f, plasticity.g)
    J_cb  = SparseConnectivity(source=bg, target=ctx, p=0.05)
    synapse_cb = LinearSynapse(J_cb.K, A=Acb)
    J_cb.store_attractors(patterns_bg[0], patterns_ctx[0], synapse_cb.h_EE, 
                        plasticity.f, plasticity.g)

    J_bc = SparseConnectivity(source=ctx, target=bg, p=0.05)
    synapse_bc = LinearSynapse(J_cc.K, A=0)
    J_bc.store_sequences(patterns_ctx, patterns_bg, synapse_bc.h_EE, plasticity.f, plasticity.g)
    J_bc.update_sequences(patterns_ctx[0][0], patterns_bg[0][1],
                    Abc, lamb=1,f=plasticity.f, g=plasticity.g)
    J_bc.update_sequences(patterns_ctx[0][1], patterns_bg[0][2],
                    Abc, lamb=1,f=plasticity.f, g=plasticity.g)
    J_bc.update_sequences(patterns_ctx[0][2], patterns_bg[0][0],
                    Abc, lamb=1,f=plasticity.f, g=plasticity.g)
    net_ctx = RateNetwork(ctx, c_EE=J_cc, c_IE=J_bc, formulation=4)
    net_bg = RateNetwork(bg, c_II=J_bb, c_EI=J_cb, formulation=4)

    init_input_ctx = phi(patterns_ctx[0][0])
    init_input_bg = phi(patterns_bg[0][0])
    T=5
    mouse = ReachingTask(3, alpha=0.5)
    net_ctx.simulate_euler2(mouse, net_bg, T, init_input_ctx, init_input_bg, 
                            patterns_ctx[0], patterns_bg[0], detection_thres=.23)
    overlaps_ctx = sequences_ctx[0].overlaps(net_ctx, ctx)
    overlaps_bg = sequences_bg[0].overlaps(net_bg, bg)  

    sns.set_style('dark') 
    colors = sns.color_palette('deep')
    fig, axes = plt.subplots(2,1, sharex=True, sharey=True, tight_layout=True, figsize=(20,13))
    axes[0].plot(overlaps_ctx[0], linestyle='solid', linewidth=4, color=colors[8])
    axes[0].plot(overlaps_ctx[1], linestyle='dashed', linewidth=4, color=colors[0])
    axes[0].plot(overlaps_ctx[2], linestyle='dotted', linewidth=4, color=colors[3])
    axes[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    axes[0].set_title("CTX", fontsize=25)
    axes[1].plot(overlaps_bg[0], linestyle='solid', linewidth=4, color=colors[8])
    axes[1].plot(overlaps_bg[1], linestyle='dashed', linewidth=4, color=colors[0])
    axes[1].plot(overlaps_bg[2], linestyle='dotted', linewidth=4, color=colors[3])
    axes[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    axes[1].set_title("BG", fontsize=25)
    axes[1].set_xlabel('Time (ms)', fontsize=20)
    fig.text(-0.01, 0.5, 'Overlap', va='center', rotation='vertical', fontsize=20)
    plt.setp(axes, xlim=(0, 1000))
    plt.figlegend(labels=['Aim', 'Reach', 'Lick'], fontsize=20)
    plt.savefig('./output/' + str(Acc) + '-' + str(Abb) + '-' + str(Acb) + '-' + str(Abc) + '.jpg', dpi=300, bbox_inches = "tight", format='jpg')
