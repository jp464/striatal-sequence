import pdb
import logging
from numba import jit, njit
import numpy as np
import scipy.sparse
from tqdm import trange
from scipy.stats import lognorm, norm
from itertools import cycle
from helpers import adder 
import sys

logger = logging.getLogger(__name__)

# params: size: number of patterns, val: value of synaptic strength A, z: 1 if symmetric and 0 if asymmetric
# returns the matrix of synaptic strength A between every pair of patterns for a given connectivity matrix 
def cmatrix(M, idxs, val, loop=False):
    for group in idxs:
        for i in range(len(group)):
            if i == len(group)-1:
                if loop: M[group[i]][0] = val
            else: M[group[i]][group[i+1]] = val


# params: pops: array of networks, cp: connection probabilty matrix, cw: sign constraint matrix, patterns: array of patterns, plasticity: learning rule functions 
# returns block matrix of connectivity matrices 
def set_connectivity(pops, cp, cw, A, patterns, plasticity):
    Jmat = np.array([])
    for pop1 in range(len(pops)):
        rowblock = np.array([])
        for pop2 in range(len(pops)):
            J = SparseConnectivity(source=pops[pop1], target=pops[pop2], p=cp[pop1][pop2])
            sign = cw[pop1][pop2]
            
            Atemp = A[pop1][pop2]
            for i in range(Atemp.shape[0]):
                for j in range(Atemp.shape[1]):
                    if Atemp[i][j] == 0: continue
                    synapse = LinearSynapse(J.K, Atemp[i][j])
                    J.update_sequences(patterns[pop1][i], patterns[pop2][j], synapse.h_EE, plasticity.f, plasticity.g)

                # sign constraint
                if sign == 1:
                    J.W.data[J.W.data < 0] = 0
                elif sign == -1:
                    J.W.data[J.W.data < 0] -= 0
                    J.W.data[J.W.data > 0] = 0 
                else:
                    J.W.data[J.W.data < 0] -= 0.01
            rowblock = np.append(rowblock, J)

        Jmat = np.vstack((Jmat, rowblock)) if Jmat.size else rowblock
    return Jmat

def corticostriatal_baby(J, patterns, p1, p2): 
    ij = J.ij
    W = J.W.toarray()
    sum = 0 
    cnt = 0
    for i in range(len(ij)):
        J_i, pre, post = W[ij[i],i], patterns[0][p1][i], np.full(len(ij[i]), patterns[1][p2][ij[i]])
        sum += np.sum(J_i * pre * post)
        cnt += len(ij[i])
    return sum / cnt

def corticostriatal(J, patterns):
    ret = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            ret[j][i] = corticostriatal_baby(J, patterns, i, j)
    return ret 

def reset_connectivity(self):
    raise NotImplementedError
        
class Connectivity(object):
    def __init__(self):
        self.W = None
        self.disable_pbar = False

    def scale_all(self, value):
        self.W *= valuelearning-env-100-1-300-0.9-1-1

    @staticmethod
    def _store_sequences(ij, inputs_pre, inputs_post, f, g, disable_pbar=False):
        """
        inputs: S x P x N
        Store heteroassociative connectivity
        """
        S, P, N  = inputs_post.shape
        row = []
        col = []
        data = []
        for n in trange(len(inputs_post), disable=disable_pbar):
            seq_pre = inputs_pre[n]
            seq_post = inputs_post[n]
            for j in trange(seq_post.shape[1], disable=disable_pbar):
                i = ij[j]
                # $f(xi_i^{\mu+1}) * g(xi_j^{\mu})$
                w = np.sum(f(seq_post[1:,i]) * g(seq_pre[:-1,j][:,np.newaxis]), axis=0) 
                row.extend(i)
                col.extend([j]*len(i))
                data.extend(w)
        return data, row, col

    @staticmethod
    def _update_sequences(ij, inputs_pre, inputs_post, f, g, disable_pbar=False):
        """
        inputs: S x P x N
        Store heteroassociative connectivity
        """
        N  = inputs_post.shape[0]
        row = []
        col = []
        data = []
        for j in trange(N, disable=disable_pbar):
            i = ij[j]
            w = f(inputs_post[i]) * g(inputs_pre[j])
            row.extend(i)
            col.extend([j]*len(i))
            data.extend(w)
        return data, row, col 

    @staticmethod
    def _store_attractors(ij, inputs_pre, inputs_post, f, g, vary_A, A, disable_pbar=False):
        """
        inputs: P x N
        Store autoassociative connectivity
        """
        P, N  = inputs_post.shape
        row = []
        col = []
        data = []
        for j in trange(inputs_post.shape[1], disable=disable_pbar):
            i = ij[j]
            if vary_A:
                w = np.sum(f(inputs_post[:,i]) * g(inputs_pre[:,j][:,np.newaxis]) * A[:,np.newaxis], axis=0) 
            else:
                w = np.sum(f(inputs_post[:,i]) * g(inputs_pre[:,j][:,np.newaxis]), axis=0) 
            row.extend(i)
            col.extend([j]*len(i))
            data.extend(w)
        return data, row, col

    @staticmethod
    def _set_all(ij, value):
        row = []
        col = []
        data = []
        for j in range(len(ij)):
            i = ij[j]
            row.extend(i)
            col.extend([j]*len(i))
            data.extend([value]*len(i))
        return data, row, col
                

class SparseConnectivity(Connectivity):
    def __init__(self, source, target, p=0.005, fixed_degree=False, seed=42, disable_pbar=False):
        self.W = scipy.sparse.csr_matrix((target.size, source.size), dtype=np.float32)
        self.E = None
        self.p = p
        self.K = p*target.size
        self.ij = []
        self.data0 = 0
        self.disable_pbar = disable_pbar
        logger.info("Building connections from %s to %s" % (source.name, target.name))
        if fixed_degree:
            n_neighbors = np.asarray([int(p*target.size)]*source.size)
        else:
            n_neighbors = np.random.RandomState(seed).binomial(target.size, p=p, size=source.size)

        @njit
        def func(source_size, target_size):
            np.random.seed(seed)
            ij = []
            for j in range(source_size): # j --> i
                # Exclude self-connections
                arr = np.arange(target_size)
                arr2 = np.concatenate((arr[:j], arr[j+1:]))
                j_subset = np.random.choice(arr2, size=n_neighbors[j], replace=False)
                ij.append(j_subset)
            return ij

        self.ij = func(source.size, target.size)

    def store_sequences(self, inputs_pre, inputs_post, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        N = inputs_post.shape[2]
        logger.info("Storing sequences")
        data, row, col = Connectivity._store_sequences(self.ij, inputs_pre, inputs_post, f, g, self.disable_pbar)
        logger.info("Applying synaptic transfer function")
        #pdb.set_trace()
        data = h(data)
        logger.info("Building sparse matrix")
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()
    
    def update_sequences(self, inputs_pre, inputs_post, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        N = inputs_post.shape[0]
        data, row, col = Connectivity._update_sequences(self.ij, inputs_pre, inputs_post, f, g, disable_pbar=True)
        data = h(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()
        
    def update_etrace(self, t, etrace, inputs_pre, inputs_post, eta, tau_e, edata, etime, R=0, f=lambda x:x, g=lambda x:x, disable_pbar=True):
        next_data = []
        for j in trange(inputs_pre.shape[0], disable=disable_pbar):
            i = self.ij[j]
            w = f(inputs_post[i]) * g(inputs_pre[j])
            next_data.extend(w)
        diff = [next-current for next, current in zip(next_data, edata)]            
        for i, synapse in enumerate(diff):
            if abs(synapse) > sys.float_info.epsilon or R:
                etrace[i] = etrace[i] * np.exp(-(t-etime[i])/tau_e) + eta * next_data[i] * (1 - np.exp(-(t-etime[i])/tau_e))
                etime[i] = t
        return etrace, next_data, etime

    def reward_etrace(self, etrace, lamb, R, inputs_pre):   
        row = []
        col = []
        for j in range(inputs_pre.shape[0]):
            i = self.ij[j]
            row.extend(i)
            col.extend([j]*len(i))
        self.W = lamb * self.W + R * scipy.sparse.coo_matrix((etrace, (row, col)), dtype=np.float32)
        
    def store_attractors(self, inputs_pre, inputs_post, h=lambda x:x, f=lambda x:x, g=lambda x:x, vary_A=False, A=None):
        logger.info("Storing attractors")
        data, row, col = Connectivity._store_attractors(self.ij, inputs_pre, inputs_post, f, g, vary_A, A, self.disable_pbar)
        data = h(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()

    def store_gaussian(self, mu=0, sigma=1, seed=2):
        w = mu + sigma*np.random.RandomState(seed).randn(self.W.data.size)
        self.W.data[:] += w

    def set_random(self, mu, var, h=lambda x:x):
        data, row, col = Connectivity._set_all(self.ij, 1)
        data = np.asarray(data, dtype=float)
        data[:] = np.sqrt(var)*(np.random.randn(data.size) + mu)
        data = h(data)
        print(data)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()

    def set_weights(self, data, row, col):
        "NOTE: Adds to, but does not overwrite existing weights"
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()

    def set_all(self, value):
        "NOTE: Adds to, but does not overwrite existing weights"
        data, row, col = Connectivity._set_all(self.ij, value)
        W = scipy.sparse.coo_matrix((data, (row, col)), dtype=np.float32)
        self.W += W.tocsr()


class DenseConnectivity(Connectivity):
    def __init__(self, source, target, seed=42, disable_pbar=False):
        self.disable_pbar = disable_pbar
        self.W = np.zeros((target.size, source.size), dtype=np.float32)
        self.K = target.size-1

    def store_sequences(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        """
        inputs: S x P x N
        """
        N = inputs.shape[2]
        logger.info("Storing sequences")
        for n in trange(len(inputs), disable=self.disable_pbar):
            seq = inputs[n]
            for mu in trange(seq.shape[0]-1, disable=self.disable_pbar):
                W = h(np.outer(f(seq[mu+1,:]), g(seq[mu,:])))
                diag = np.diagonal(W)
                diag.setflags(write=True)
                diag.fill(0)
                self.W += W

    def store_attractors(self, inputs, h=lambda x:x, f=lambda x:x, g=lambda x:x):
        pass

    def set_weights(self, data, row, col):
        pass

    def set_all(self, value):
        pass


class ThresholdPlasticityRule(object):
    def __init__(self, x_f, q_f, x_g=None, rv=scipy.stats.norm):
        if not x_g:
            x_g = x_f
        q_g = rv.cdf(x_g)
        self.x_f, self.q_f = x_f, q_f
        self.x_g, self.q_g = x_g, q_g
        self.f = lambda x: np.where(x < x_f, -(1-q_f), q_f)
        self.g = lambda x: np.where(x < x_g, -(1-q_g), q_g)

class SynapticTransferFunction(object):
    def __init__(self, K):
        pass

class LinearSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, A):
        super(LinearSynapse, self).__init__(self)
        self.A = A
        self.K_EE = K_EE

    def h_EE(self, J):
        return self.A * np.asarray(J) / self.K_EE

    def h(self, J):
        return self.h_EE(J)


class RectifiedSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, K_IE, K_EI, K_II, alpha, plasticity, A=1, g=1., o=0):
        super(RectifiedSynapse, self).__init__(self)

        @np.vectorize
        def rectify(x):
            if x<0:
                return 0
            else:
                return x

        self.K_EE, self.K_IE, self.K_EI, self.II = K_EE, K_IE, K_EI, K_II
        f_fun, x_f, q_f = plasticity.f, plasticity.x_f, plasticity.q_f
        g_fun, x_g, q_g = plasticity.g, plasticity.x_g, plasticity.q_g
        self.A, self.g, self.o, = A, g, o

        gamma = norm.expect(lambda x: f_fun(x)**2) * \
                norm.expect(lambda x: g_fun(x)**2)
        self.E_w = norm.expect(
                lambda x: rectify(g*A*np.sqrt(alpha*gamma)*x + o),
                scale=1)
        self.E_w_2 = norm.expect(
                lambda x: rectify(g*A*np.sqrt(alpha*gamma)*x + o)**2,
                scale=1)
    
    def h_EE(self, J):
        return self.A * (self.g * (J / np.sqrt(self.K_EE)) + self.o).clip(min=0) / np.sqrt(self.K_EE)

    def h_IE(self, J):
        return J * self.E_w / self.K_IE

    def h_EI(self, J):
        return -J * 1. / np.sqrt(self.K_EI)

    def h_II(self, J):
        return -J


class ExponentialSynapse(SynapticTransferFunction):
    def __init__(self, K_EE, K_IE, K_EI, K_II, alpha, plasticity, A=1, g=1., o=0):
        super(ExponentialSynapse, self).__init__(self)

        @np.vectorize
        def exp(x):
            return np.exp(x)

        self.K_EE, self.K_IE, self.K_EI, self.II = K_EE, K_IE, K_EI, K_II
        f_fun, x_f, q_f = plasticity.f, plasticity.x_f, plasticity.q_f
        g_fun, x_g, q_g = plasticity.g, plasticity.x_g, plasticity.q_g
        self.A, self.g, self.o, = A, g, o

        gamma = norm.expect(lambda x: f_fun(x)**2) * \
                norm.expect(lambda x: g_fun(x)**2)
        self.E_w = norm.expect(
                lambda x: exp(g*A*np.sqrt(alpha*gamma)*x + o),
                scale=1)
        self.E_w_2 = norm.expect(
                lambda x: exp(g*A*np.sqrt(alpha*gamma)*x + o)**2,
                scale=1)
    
    def h_EE(self, J):
        return self.A * (self.g * (J / np.sqrt(self.K_EE)) + self.o).clip(min=0) / np.sqrt(self.K_EE)

    def h_IE(self, J):
        return J * self.E_w / self.K_IE

    def h_EI(self, J):
        return -J * 1. / np.sqrt(self.K_EI)

    def h_II(self, J):
        return -J

