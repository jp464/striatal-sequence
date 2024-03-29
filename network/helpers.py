import numpy as np
import scipy.stats

def spike_to_rate(spikes, window_std=20):
    window_size = np.arange(-3*window_std,3*window_std,1)
    window = scipy.stats.norm.pdf(window_size, 0, window_std)
    window /= window.sum()
    n_units = spikes.shape[0]
    estimate = np.zeros_like(spikes) # Create an empty array of the same size as spikes
    for i in range(n_units):
        y = np.convolve(window, spikes[i,:], mode='same')
        estimate[i,:] = y
    return estimate

def compute_overlap(state, patterns):
    return [state.T.dot(p) / state.shape[0] for p in patterns]

# Determines which action mouse is currently in 
def determine_action(state, patterns, thres=0.3, correlations=False):
    if correlations:
        measure = [scipy.stats.pearsonr(state, p)[0] for p in patterns]
    else:
        measure = [state.T.dot(p) / state.shape[0] for p in patterns]
    maxind = np.argmax(measure)
    if measure[maxind] > thres and abs(np.sort(measure)[-2]-measure[maxind]) > .01:
        return maxind
    return -1

def action_transition(i, mouse, states, patterns, thres):
    next_action = determine_action(states[0][:,i], patterns[0][0], thres)
    if mouse.action != next_action and next_action != -1:
        mouse.compute_reward(next_action)
        mouse.compute_water(mouse.action, next_action)
        mouse.compute_barrier(next_action)
        
        mouse.action_dur = 0
        mouse.action = next_action
        return True 
    return False 

# Hyperpolarizing current if at an action for thres ms 
def hyperpolarize(hyperpolarize_dur, cur_action, action_dur, r_ext, thres=500, h_dur=50, cur1=lambda t:-10, cur2=lambda t:0):
    if hyperpolarize_dur > 0 and hyperpolarize_dur < h_dur:
        hyperpolarize_dur += 1
    else:
        hyperpolarize_dur = 0
        if action_dur > thres and cur_action != -1:
            r_ext = cur1
            hyperpolarize_dur += 1
        else:
            r_ext = cur2
    return r_ext, hyperpolarize_dur

def adder(L, i):
    return np.sum(L[0:i])

    
        
        
        
            
