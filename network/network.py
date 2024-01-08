import pdb
import logging
import numpy as np
import scipy.sparse
import scipy.integrate
from tqdm import tqdm, trange
import progressbar
from numba import jit, njit
from connectivity import Connectivity, LinearSynapse, corticostriatal
from helpers import spike_to_rate, determine_action, action_transition, hyperpolarize, overlap
from scipy.stats import pearsonr
from learning import NetworkUpdateRule
from transfer_functions import erf, ErrorFunction
import pandas as pd
import h5py

logger = logging.getLogger(__name__)

class Population(object):
    def __init__(self, N, tau, phi=lambda x:x, name="exc"):
        self.name = name
        self.size = N
        self.state = np.array([], ndmin=2).reshape(N,0)
        self.noise = np.array([], ndmin=2).reshape(N,0)
        self.field = np.array([], ndmin=2).reshape(N,0)
        self.tau = tau
        self.phi = phi
        


class SpikingNeurons(Population):
    def __init__(self, N, tau_mem, tau_syn, thresh, reset, name="exc"):
        super(SpikingNeurons, self).__init__(N, None, None)
        self.name = name
        self.tau_mem = tau_mem
        self.tau_syn = tau_syndf1 = df1.assign(e=pd.Series(np.random.randn(sLength)).values)
        self.thresh = thresh
        self.reset = reset


class Network(object):
    def __init__(self, pops, J, inh=None):
        self.pops = pops 
        self.Np = len(pops)
        self.J = J
        self.inh = inh
        self.t = np.array([])
        if inh:
            self.W_EI = c_EI.W
            self.W_IE = c_IE.W
            self.W_II = c_II.W
            self.tau = np.concatenate([
                np.array([exc.tau]*exc.size),
                np.array([inh.tau]*inh.size)
            ])
            self.size += self.inh.size
        else:
            self.tau = pops[0].tau
        self.xi = None
        self.r_ext = [lambda t:0 for i in range(len(pops))]

class RateNetwork(Network):
    def __init__(self, pops, J, inh=None, formulation=1, disable_pbar=False):
        super(RateNetwork, self).__init__(pops, J, inh)
        self.formulation = formulation
        self.disable_pbar = disable_pbar
        self.hyperpolarize_dur = [0 for i in range(len(pops))] 
        if self.formulation == 1:
            self._fun = self._fun1
        elif self.formulation == 2:
            self._fun = self._fun2
        elif self.formulation == 3:
            self._fun = self._fun3
        elif self.formulation == 4:
            self._fun = self._fun4
        elif self.formulation == 5:
            self._fun = self._fun5
            
    def simulate_learning(self, mouse, t, r, patterns, plasticity, delta_t, eta, tau_e, lamb, noise, e_bl, alpha, gamma, adap, env, 
                          etrace=True, t0=0, dt=1e-3, r_ext=lambda t: 0, detection_thres=0.3, print_output=False, track=False):
        logger.info("Integrating network dynamics")
        if self.disable_pbar:
            pbar = progressbar
            fun = self._fun(pbar,t)
        else:
            fun = self._fun(tqdm(total=int(t/dt)-1),t)
        
        # ===================================================================================================================================
        # INITIALIZATION
        # ===================================================================================================================================
        states = np.zeros(self.Np, dtype=object)
        prev_actions = np.zeros(self.Np, dtype=np.int8)
        prev_idxs = np.zeros(self.Np, dtype=np.int32)
        mouse.behaviors = np.zeros(self.Np, dtype=object)
        for i in range(self.Np):
            states[i] = np.zeros((self.pops[i].size, int((t-t0)/dt)))
            states[i][:,0] = r[i]
            e = np.zeros(len(e_bl), dtype='float')
            prev_actions[i] = -1
            prev_idxs[i] = 0
            mouse.behaviors[i] = np.empty(int((t-t0)/dt), dtype=object)
            mouse.behaviors[i][prev_idxs[i]] = [prev_actions[i],i]
        adaptation = np.zeros((self.pops[0].size))    
            
        mouse.values = np.zeros(((len(mouse.actions)-1)*4, int((t-t0)/dt)))
        value = np.zeros((len(mouse.actions)-1)*4)
        mouse.rpes = np.zeros(((len(mouse.actions)-1)*4, int((t-t0)/dt)))
        rpe = np.zeros((len(mouse.actions)-1)*4)
        ws = np.zeros(int((t-t0)/dt))
        bs = np.zeros(int((t-t0)/dt))
        rs = np.zeros(int((t-t0)/dt))
        mouse.evars = np.zeros((len(e_bl), int((t-t0)/dt)))
        mouse.corticostriatal = np.zeros((4,4), dtype=float)
        check = False
        
        # eligibility trace parameters  
        N_synapse, row, col = 0, [], []
        for j in range(self.pops[0].size):
            i = self.J[0][1].ij[j]
            row.extend(i)
            col.extend([j]*len(i))
            N_synapse += len(i)
            
        self.J[0][1].E = np.zeros(N_synapse)
        eprev = None
        ecnt = 0
        data = np.zeros(N_synapse)
        time_ls = np.zeros(N_synapse)
        self.r_ext = r_ext

        # ===================================================================================================================================
        # SIMULATION
        # ===================================================================================================================================
        for i, t in enumerate(np.arange(t0, t, dt)[0:-1]):
            
            mouse.evars[:,i] = np.array(e)
            
            ### Update firing rate 
            m_ctx = overlap(states[0][:,i], patterns[0])
            dr, de, da = fun(i, states, e, e_bl, m_ctx, adaptation, patterns, mouse, env, adap)
                
            for k in range(self.Np):
                white_noise = np.random.normal(size=self.pops[k].size) * noise[k]
                states[k][:,i+1] = states[k][:,i] + dt * dr[k] + white_noise
            for k in range(len(e)):
                e[k] += dt * de[k]
            adaptation += dt * da

            ### Update eligibility traces 
            if i - delta_t > 0:
                if print_output:
                    pre = determine_action(states[0][:,i-delta_t], patterns[0], thres=0.15)
                    post = determine_action(states[1][:,i+1], patterns[1], thres=0.15)
                    
                    if eprev == [pre, post]:
                        ecnt += 1
                    else:
                        print(eprev, ecnt)
                        ecnt = 0
                        eprev = [pre, post]
                if etrace:
                    data, time_ls = self.J[0][1].update_etrace(i, states[0][:,i-delta_t], states[1][:,i+1], eta=eta, tau_e=tau_e, 
                                                               data_prev=data, time_ls=time_ls, row=row, col=col, f=plasticity.f, g=plasticity.g)
                
            ### Update mouse behavior 
            mouse.action_dur += 1 
            transitions = action_transition(i, mouse, prev_actions, prev_idxs, states, patterns, thres=detection_thres)
            
            if prev_idxs[0] > 0: a0, a1 = mouse.get_action(mouse.behaviors[0][prev_idxs[0]-1][0]), mouse.get_action(mouse.behaviors[0][prev_idxs[0]][0])
            else: continue
            
            if transitions[0]:
                mouse.compute_reward(a1)
                mouse.water(a0, a1)
                mouse.barrier(a1)
                check = True
        
                if print_output:
                    print(a0 + "-->" + a1)
                bs[prev_idxs[0]] = mouse.b
                ws[prev_idxs[0]] = mouse.w
                rs[prev_idxs[0]] = mouse.r
                
                
                ### Compute RPE, V
                if a1 != None:
                    rpe, value = mouse.compute_error_value(prev_idxs[0], a0, a1, bs, ws, rs, value, rpe, gamma=gamma, alpha=alpha)

            ### Reward
            if mouse.r and mouse.action_dur > 100 and check == True:
                check = False
                print('Mouse drank water')
                if etrace:
                    data, time_ls = self.J[0][1].update_etrace(i, states[0][:,i-delta_t], states[1][:,i+1], eta=eta, tau_e=tau_e, 
                                           data_prev=data, time_ls=time_ls, R=1, f=plasticity.f, g=plasticity.g)
                    self.J[0][1].reward_etrace(lamb=lamb, R=1)
                mouse.corticostriatal = np.vstack((mouse.corticostriatal, corticostriatal(self.J[0][1], patterns)))
            
        # ===================================================================================================================================
        # SAVE 
        # ===================================================================================================================================
        for k in range(self.Np):
            self.pops[k].state = np.hstack([self.pops[k].state, states[k][:self.pops[k].size,:]])
        mouse.values = mouse.values[:,0:prev_idxs[0]]
        mouse.rpes = mouse.rpes[:,0:prev_idxs[0]]
    
    def simulate_euler2(self, mouse, t, r1, r2, r3, y2, y3, patterns_ctx, patterns_d1, patterns_d2, detection_thres, noise1, noise2, noise3, t0=0, dt=1e-3, r_ext=lambda t: 0):
        logger.info("Integrating network dynamics")
        
        if self.disable_pbar:
            pbar = progressbar.NullBar()
            fun = self._fun(pbar,t)
        else:
            fun = self._fun(tqdm(total=int(t/dt)-1),t)
            
        # Initial conditions                
        state1 = np.zeros((self.pops[0].size, int((t-t0)/dt)))
        state1[:,0] = r1
        state2 = np.zeros((self.pops[1].size, int((t-t0)/dt)))
        state2[:,0] = r2
        state3 = np.zeros((self.pops[2].size, int((t-t0)/dt)))
        state3[:,0] = r3
                         
        for i, t in enumerate(np.arange(t0, t, dt)[0:-1]):
            dr1, dr2, dr3, dy2, dy3 = fun(i, state1[:,i], state2[:,i], state3[:,i], depression2[:,i], depression3[:,i], beta=.01)    
            state1[:,i+1] = state1[:,i] + dt * dr1 + np.random.normal(size=self.pops[0].size) * noise1
            state2[:,i+1] = state2[:,i] + dt * dr2 + np.random.normal(size=self.pops[1].size) * noise2
            state3[:,i+1] = state3[:,i] + dt * dr3 + np.random.normal(size=self.pops[2].size) * noise3
            depression2[:,i+1] = depression2[:,i] + dt * dy2 
            depression3[:,i+1] = depression3[:,i] + dt * dy3

        self.pops[0].state = np.hstack([self.pops[0].state, state1[:self.pops[0].size,:]])
        self.pops[1].state = np.hstack([self.pops[1].state, state2[:self.pops[1].size,:]])
        self.pops[2].state = np.hstack([self.pops[2].state, state3[:self.pops[2].size,:]])
        self.pops[1].depression = np.hstack([self.pops[1].depression, depression2[:self.pops[1].size,:]])
        self.pops[2].depression = np.hstack([self.pops[2].depression, depression3[:self.pops[2].size,:]])
        
        
    def simulate(self, t, r0, t0=0, dt=1e-3, r_ext=lambda t: 0):
        """
        Runge-Kutta 2nd order
        """
        logger.info("Integrating network dynamics")
        if self.disable_pbar:
            pbar = progressbar.NullBar()
        else:
            pbar = progressbar.ProgressBar(
                maxval=t,
                widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        pbar.start()
        self.r_ext = r_ext
        sol = scipy.integrate.solve_ivp(
            self._fun(pbar,t),
            t_span=(t0,t),
            t_eval=np.arange(0,t,dt),
            y0=r0,
            method="RK23")
        pbar.finish()
        self.t = sol.t
        state = sol.y

        # Save network state
        self.pops[0].state = np.hstack([self.pops[0].state, state[:self.pops[0].size,:]])
        if self.inh: 
            self.inh.state = np.hstack([self.inh.state, state[-self.inh.size:,:]])

    def simulate_euler(self, t, r0, t0=0, dt=1e-3, r_ext=lambda t: 0, save_field=True):
        """
        Euler-Maryama scheme
        """
        logger.inmeansfo("Integrating network d1ynamics")
        self.r_ext = r_ext
        state = np.zeros((self.exc.size, int((t-t0)/dt)))
        field = np.zeros_like(state)
        state[:,0] = r0
        if self.disable_pbar:
            pbar = progressbar.NullBar()
            fun = self._fun(pbar,t)
        else:
            fun = self._fun(tqdm(total=int(t/dt)),t)
        for i, t in enumerate(np.arange(t0,t-dt,dt)[:-1]):
            r = state[:,i]
            dr, field_i = fun(t, r, return_field=save_field)
            if self.xi:
                sigma = np.sqrt(field.var()) / self.tau
                dr += self.xi.value(dt,self.tau,self.exc.size) * sigma
            state[:,i+1] = state[:,i] + dt * dr
            if save_field:
                field[:,i] = field_i
        self.exc.state = np.hstack([self.exc.state, state[:self.exc.size,:]])
        if save_field:
            self.exc.field = np.hstack([self.exc.field, field[:self.exc.size,:]])
        if self.inh: 
            self.inh.state = np.hstack([self.inh.state, state[-self.inh.size:,:]])
            if save_field:
                self.inh.field = np.hstack([self.inh.fHelloield, field[:self.inh.size,:]])

    def add_noise(self, xi, pop):
        self.xi = xi

    def _fun1(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate meansformulation 1
            """
            # $ \frac{dx}{dt} = -x + \phi( \sum_{j} J_{ij} x_j + I_0 ) $
            pbar.update(1)

            if self.inh:
                raise NotImplemented
            else:
                phi_r = self.pops[0].phi
            r_ext = self.r_ext
            r_sum = phi_r(self.J[0].W.dot(r) + r_ext(t))
            dr = (-r + r_sum) / self.tau
            if return_field:
                return dr, r_sum
            else:
                return dr
        return f

    def _fun2(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 2
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij} /phi(x_j) + I_0 $

            pbar.update(t)

            if self.inh:
                phi_r = np.zeros_like(r)
                phi_r[:self.exc.size] = self.exc.phi(r[:self.exc.size])
                phi_r[-self.inh.size:] = self.inh.phi(r[-self.inh.size:])
            else:
                phi_r = self.exc.phi(r)
            r_ext = self.r_ext
            r_sum = self.W.dot(phi_r)
            dr = (-r + r_sum + r_ext(t)) / self.tau
            if return_field:
                return dr, r_sum
            else:
                return dr
        return f

    def _fun3(self, pbar, t_max):
        def f(t, r, return_field=False):
            """
            Rate formulation 3: instantaneous inhibition
            """
            # $ \frac{dx}{dt} = -x + \sum_{j} J_{ij} /phi(x_j) + I_inh + I_0 $

            pbar.update(t%t_max)

            if self.inh is None:
                return NotImplementedError

            r_ext = self.r_ext
            phi_r_exc = self.exc.phi(r[:self.exc.size])
            r_sum_exc = self.W_EE.dot(phi_r_exc)
            r_sum_inh = 20*self.W_EI.dot(self.W_IE.dot(phi_r_exc)) #FIXME: 20 is g_phi^I. Do not hardcode
            dr = np.zeros(self.size)
            dr[:self.exc.size] = (-r[self.exc.size] + r_sum_exc + r_sum_inh + r_ext(t)) / self.tau[:self.exc.size]

            return dr
        return f
    
    def _fun4(self, pbar, t_max):
        def f(t, states, e, e_bl, overlaps, a, patterns, mouse, env, adap, return_field=False):
            """
            Rate formulation 4
            """
            # $ \frac{dx}{dt} = -x + \phi( \sum_{j} J_{ij} x_j + I_0 ) $
            try:
                pbar.update(1)
            except:
                pass
            if self.inh:
                raise NotImplemented
            else:
                phi_r = self.pops[0].phi
            r1, r2 = states[0][:,t], states[1][:,t]
            
            r_sum1 = phi_r((self.J[0][0].W.dot(r1) + self.J[1][0].W.dot(r2) + self.r_ext[0](t)) + env * mouse.env(e, patterns[0]) - adap*a)
            r_sum2 = phi_r(self.J[1][1].W.dot(r2) + self.J[0][1].W.dot(r1) + self.r_ext[1](t))
            dr1 = (-r1 + r_sum1) / self.tau
            dr2 = (-r2 + r_sum2) / self.tau  
            de = np.zeros(len(e), dtype='float')
            
            if env:
                cur_action = np.argmax(overlaps)
                ctau = 25
                cf_a = cf_r = cf_l = 0.8
                cf_s = 0.8
                
                if cur_action == 0:
                    de[0] = (-e[0] + e_bl[0] - overlaps[0]*cf_a) / (self.tau * ctau)
                    de[1] = (-e[1] + e_bl[1] + overlaps[0]*.1) / (self.tau * ctau)
                    de[2] = (-e[2] + e_bl[2] + overlaps[0]*.1) / (self.tau * ctau)
                    de[3] = (-e[3] + e_bl[3] - overlaps[3]*cf_s) / (self.tau * ctau)
                elif cur_action == 1:
                    de[0] = (-e[0] + e_bl[0] - overlaps[0]*cf_a) / (self.tau * ctau)
                    de[1] = (-e[1] + e_bl[1] - overlaps[1]*cf_r) / (self.tau * ctau)
                    de[2] = (-e[2] + e_bl[2] + overlaps[1]*.1) / (self.tau * ctau)
                    de[3] = (-e[3] + e_bl[3] - overlaps[3]*cf_s) / (self.tau * ctau)         
                else:
                    de[0] = (-e[0] + e_bl[0] - overlaps[0]*cf_a) / (self.tau * ctau)
                    de[1] = (-e[1] + e_bl[1] - overlaps[1]*cf_r) / (self.tau * ctau)
                    de[2] = (-e[2] + e_bl[2] - overlaps[2]*cf_l) / (self.tau * ctau)
                    de[3] = (-e[3] + e_bl[3] - overlaps[3]*cf_s) / (self.tau * ctau)            
            da = (-a + r1) / (25 * self.tau)
            
            return [dr1, dr2], de, da

        return f

    def _fun5(self, pbar, t_max):
        def f(t, states, deps, beta, return_field=False):
            """
            Rate formulation 5
            """
            # $ \frac{dx}{dt} = -x + \phi( \sum_{j} J_{ij} x_j + I_0 ) $
            try:
                pbar.update(1)
            except:
                pass

            if self.inh:
                raise NotImplemented
            else:
                phi_r = self.pops[0].phi
            sig = lambda x: 0 if x < .3 else x
            
            if t > 0 and t < 150:
                self.r_ext[2] = lambda t: 0
            else:
                self.r_ext[2] = lambda t: 1
                
            
            r1, r2, r3 = states[0][:,t], states[1][:,t], states[2][:,t]
            s1, s2, s3 = deps[0][:,t], deps[1][:,t], deps[2][:,t]
            r_sum1 = phi_r(self.J[0][0].W.dot(r1) + self.J[1][0].W.dot(r2) + self.J[2][0].W.dot(r3) + self.r_ext[0](t))
            r_sum2 = phi_r(self.J[1][1].W.dot(r2) + self.J[0][1].W.dot(r1) + self.J[2][1].W.dot(np.vectorize(sig)(r3)*s3)*.1 + self.r_ext[1](t))
            r_sum3 = phi_r(self.J[2][2].W.dot(r3) + self.J[0][2].W.dot(r1) + self.J[1][2].W.dot(np.vectorize(sig)(r2)*s2)*.1 + self.r_ext[2](t))

            
            dr1 = (-r1 + r_sum1) / self.tau
            dr2 = (-r2 + r_sum2) / self.tau
            dr3 = (-r3 + r_sum3) / self.tau 
            
            ds1 = np.zeros(len(r_sum1))
            
#             sig = lambda x:x
            ds2 = (-(s2-1)*(1-r_sum2) - (s2-beta)*r_sum2) / (10*self.tau)
            ds3 = (-(s3-1)*(1-r_sum3) - (s3-beta)*r_sum3) / (10*self.tau)
            

            return np.array([dr1, dr2, dr3]), np.array([ds1, ds2, ds3])

        return f
    
    def overlap_with(self, vec, pop, spikes=False):
        """
        Compute the overlap of network activity with a given input vector
        """
#         ret = np.zeros(pop.state.shape[1])
#         for i, row in enumerate(pop.state.T):
#             ret[i] = np.sum((row - vec)**2) / (self.exc.size-1)
#         return ret    
        return pop.state.T.dot(vec) / self.exc.size

    def clear_state(self):
        self.exc.state = np.array([], ndmin=2).reshape(self.exc.size,0)
        self.exc.field = np.array([], ndmin=2).reshape(self.exc.size,0)
        if self.inh:
            self.inh.state = np.array([], ndmin=2).reshape(self.inh.size,0)
            self.inh.field = np.array([], ndmin=2).reshape(self.inh.size,0)


    
    def reward_etrace(self, W, E, lamb, R):
        return lamb * W+ R * E.tocsr()
        
class PoissonNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(), 
            c_EI=Connectivity(), c_II=Connectivity(),
            r_max=100.):
        super(PoissonNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II)
        self.r_max=r_max

    def simulate(self, t, r0, t0=0, dt=1e-3, exact=False):
        tau = self.tau
        neighbors = self.c_EE.ij
        W = self.W
        r = r0*1
        phi = self.exc.phi

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size))
        spikes = np.zeros_like(state)

        #@njit -- no speedup with compilation
        def func(state, spikes, W, r, r_max, t_size, tau, dt, phi, neighbors):
            for n in range(t_size-1):
                rv = np.random.random(size=r.size)
                spks = np.nonzero(rv < phi(r)*dt*r_max)[0]

                # Propagate spike to neighbors
                for j in spks:
                    idxs = neighbors[j]
                    r[idxs] += 1./r_max*W[idxs,j].flatten()/tau

                # Decay intensity function for all units
                r = r * np.exp(-dt/tau)

                # Record state and spikes
                state[:,n+1] = r
                spikes[spks,n] = 1

        # Much faster with dense matrix
        func(state, spikes, np.asarray(W.todense()), r, self.r_max,
             t_size, tau, dt, phi, neighbors)

        # Save network state
        self.exc.state = state[:self.exc.size,:]
        self.exc.spikes = spikes[:self.exc.size,:]
        if self.inh: 
            self.inh.state = state[self.exc.size:,:]
            self.inh.spikes = spikes[self.exc.size:,:]

    def rates(self, pop):
	    r = spike_to_rate(pop.spikes)
	    return r

    def overlap_with(self, vec, pop, spikes=False):
        "Compute the overlap of network activity with a given input vector"
        if spikes:
            r = self.rates(pop)
        else:
            r = pop.state
        return r.T.dot(vec) / self.exc.size


class SpikingNetwork(Network):
    def __init__(self, exc, inh=None,
            c_EE=Connectivity(), c_IE=Connectivity(),
            c_EI=Connectivity(), c_II=Connectivity(),
            r_max=100.):
        super(SpikingNetwork, self).__init__(exc, inh, c_EE, c_IE, c_EI, c_II)

    def simulate(self, t, s0, v0, t0=0, dt=1e-3, tau_rp=1e-3, exact=False):
        neighbors = self.c_EE.ij
        W_EE = np.asarray(self.W_EE.todense())
        s = s0
        v = v0
        thresh = self.exc.thresh
        reset = self.exc.reset
        tau_mem = self.exc.tau_mem
        tau_syn = self.exc.tau_syn

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size, 2))
        spikes = np.zeros_like(state[:,:,0])

        for n in range(t_size-1):
            spks = np.nonzero(v > thresh)[0]

            # Decay intensity function for all units
            v = v * np.exp(-dt/tau_mem) + s * (1 - np.exp(-dt/tau_mem))
            s = s * np.exp(-dt/tau_syn)
            v[spks] = reset

            # Propagate spike to neighbors
            for j in spks:
                idxs = neighbors[j]
                s[idxs] += W_EE[idxs,j].flatten()/tau_syn

            if n > 0:
                idxs = spikes[:,n-1].nonzero()[0]
                v[idxs] = reset

            # Record state and spikes
            state[:,n+1,0] = s
            state[:,n+1,1] = v
            spikes[spks,n] = 1

        # Save network state
        self.exc.state = state[:self.exc.size,:,:]
        self.exc.spikes = spikes[:self.exc.size,:]

    # TODO: Note that refractory period is hardcoded to 1ms right now
    def simulate_two_pop(self, 
            t,
            s0_exc,
            v0_exc,
            s0_inh,
            v0_inh,
            t0=0,
            dt=1e-3,
            tau_rp=1e-3,
            sigma_lif_E=0,
            sigma_lif_I=0,
            v_exc_lower_bound=None,
            white_noise_seed=100):
        """
        Simulate two population LIF spiking network

        Inputs:
            t: 
            s0_exc:
            v0_exc:
            s0_exc:
            v0_exc: 
            t0:
            dt:
            tau_rp:
            v_exc_lower_bound:
        """

        neighbors_EE = self.c_EE.ij
        neighbors_EI = self.c_EI.ij
        neighbors_IE = self.c_IE.ij
        neighbors_II = self.c_II.ij

        W_EE = np.asarray(self.W_EE.todense())
        W_EI = np.asarray(self.W_EI.todense())
        W_IE = np.asarray(self.W_IE.todense())
        W_II = np.asarray(self.W_II.todense())

        s_exc = s0_exc
        v_exc = v0_exc
        s_inh = s0_inh
        v_inh = v0_inh

        thresh_exc = self.exc.thresh
        reset_exc = self.exc.reset
        tau_mem_exc = self.exc.tau_mem
        tau_syn_exc = self.exc.tau_syn

        thresh_inh = self.inh.thresh
        reset_inh = self.inh.reset
        tau_mem_inh = self.inh.tau_mem
        tau_syn_inh = self.inh.tau_syn

        logger.info("Integrating network dynamics")
        t_size = int(t/dt)
        state = np.zeros((self.size, t_size, 2))
        spikes = np.zeros_like(state[:,:,0])

        xi = np.random.RandomState(seed=white_noise_seed)

        for n in trange(t_size-1):

            # Get indices of neurons that crossed spiking threshold in the last timestep
            spks_exc = np.nonzero(v_exc >= thresh_exc)[0]
            spks_inh = np.nonzero(v_inh >= thresh_inh)[0]

            # Evolve membrane voltage in time
            v_exc = v_exc * np.exp(-dt/tau_mem_exc) + \
                    s_exc * (1 - np.exp(-dt/tau_mem_exc))
            v_inh = v_inh * np.exp(-dt/tau_mem_inh) + \
                    s_inh * (1 - np.exp(-dt/tau_mem_inh))

            # Evolve synaptic currents in time
            s_inh = s_inh * np.exp(-dt/tau_syn_inh)
            s_exc = s_exc * np.exp(-dt/tau_syn_exc)

            # Propagate exc spikes to neighbors
            for j in spks_exc:
                idxs_EE = neighbors_EE[j]
                idxs_IE = neighbors_IE[j]
                s_exc[idxs_EE] += W_EE[idxs_EE,j].flatten()/tau_syn_exc
                s_inh[idxs_IE] += W_IE[idxs_IE,j].flatten()/tau_syn_inh
            # Propagate inh spikes to neighbors
            for j in spks_inh:
                idxs_II = neighbors_II[j]
                idxs_EI = neighbors_EI[j]
                s_inh[idxs_II] += W_II[idxs_II,j].flatten()/tau_syn_inh
                s_exc[idxs_EI] += W_EI[idxs_EI,j].flatten()/tau_syn_exc

            # Reset neurons that spiked in last timestep
            v_exc[spks_exc] = reset_exc
            v_inh[spks_inh] = reset_inh

            # Inject white noise into membrane potentials
            if sigma_lif_E > 0:
                v_exc += np.sqrt(dt/tau_mem_exc)*sigma_lif_E*xi.normal(size=self.exc.size)
            if sigma_lif_I > 0:
                v_inh += np.sqrt(dt/tau_mem_inh)*sigma_lif_I*xi.normal(size=self.inh.size)

            # Clamp membrane potentials above threshold to threshold
            v_exc[np.nonzero(v_exc > thresh_exc)[0]] = thresh_exc
            v_inh[np.nonzero(v_inh > thresh_inh)[0]] = thresh_inh

            # Clamp membrane potentials below voltage floor to floor
            if v_exc_lower_bound:
                v_exc[v_exc < v_exc_lower_bound] = v_exc_lower_bound

            # Refractory period: Clamp voltage of neurons that spiked 1 timestep ago to reset/rest
            if n > 0:
                idxs_exc = spikes[:,n-1][:self.exc.size].nonzero()[0]
                idxs_inh = spikes[:,n-1][self.exc.size:].nonzero()[0]
                v_exc[idxs_exc] = reset_exc
                v_inh[idxs_inh] = reset_inh

            # Record state and spikes
            state[:,n+1,0] = np.concatenate([s_exc,s_inh])
            state[:,n+1,1] = np.concatenate([v_exc,v_inh])
            spikes[spks_exc,n] = 1
            spikes[self.exc.size+spks_inh,n] = 1

        # Save network state
        self.exc.state = state[:self.exc.size,:,:]
        self.exc.spikes = spikes[:self.exc.size,:]
        if self.inh: 
            self.inh.state = state[self.exc.size:,:,:]
            self.inh.spikes = spikes[self.exc.size:,:]

    def rates(self, pop):
	    r = spike_to_rate(pop.spikes)
	    return r

    def overlap_with(self, vec, pop, spikes=False):
        "Compute the overlap of network activity with a given input vector"
        if spikes:
            r = self.rates(pop)
        else:
            r = pop.state[:,:,0]
        return r.T.dot(vec) / self.exc.size

