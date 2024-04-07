import sys
sys.path.insert(0, '../../../network')
import numpy as np
import matplotlib.pyplot as plt 
from transfer_functions import ErrorFunction
from connectivity import ThresholdPlasticityRule
from scipy.stats import norm
from scipy.special import erf
from numpy import exp, sqrt, pi

mu = 0.22
sigma = 0.1
tau = 1e-2
dt = 1e-3
q_f = 0.8
x_f = abs(norm.ppf(1-q_f))
phi = lambda x: 0.5 * (1 + erf((x - mu) / (sqrt(2) * sigma)))
f = lambda x: np.where(x < x_f, -(1-q_f), q_f)
states = [f(0), f(1)]

def prob(etas, thres):
    ret = 1
    for eta in etas:
        if eta < 0: ret *= norm.cdf(thres)
        else: ret *= (1 - norm.cdf(thres))
    return ret 

def ctx_dynamical_mft(q, l, Acc, Acs, x_f=x_f, states=states, tau=1e-2):
    ret = 0
    states = [f(0), f(1)]
    for etas in [(p1, p2, p3, p4) for p1 in states for p2 in states for p3 in states for p4 in states]:
        ret += prob(etas, x_f) * etas[l] * phi(np.sum([etas[mu] * (Acc * q[mu] + Acs * q[mu+4]) for mu in range(4)]))
    return (-q[l] + ret)/tau

def str_dynamical_mft(q, l, Ass, Asc, x_f=x_f, states=states, tau=1e-2):
    ret = 0
    states = [f(0), f(1)]
    for etas in [(p1, p2, p3, p4) for p1 in states for p2 in states for p3 in states for p4 in states]:
        ret += prob(etas, x_f) * etas[l%4] * phi(np.sum([Ass * etas[mu] * q[mu+4] for mu in range(4)]) + 
                                               np.sum([Asc * etas[mu+1] * q[mu] for mu in range(3)]) +  Asc * etas[0] * q[3])
    return (-q[l] + ret)/tau 

def rate(y, t, Acc, Asc, Ass, Acs):
    dydt = np.zeros(len(y))
    for i in range(len(y)):
        if i < 4:
            dydt[i] = ctx_dynamical_mft(y, i, Acc, Acs)
        else:
            dydt[i] = str_dynamical_mft(y, i, Ass, Asc)
    return dydt