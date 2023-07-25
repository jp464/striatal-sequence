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

# Determines which action mouse is currently in 
def determine_action(state, patterns, thres=0.6, correlations=False):
    if correlations:
        measure = [scipy.stats.pearsonr(state, p)[0] for p in patterns]
    else:
        measure = [state.T.dot(p) / state.shape[0] for p in patterns]
    maxind = np.argmax(measure)
    if measure[maxind] > thres and abs(np.sort(measure)[-2]-measure[maxind]) > .01:
        return maxind
    return -1

# Hyperpolarizing current if at an action for thres ms 
def hyperpolarizing_current(action_dur, cur_action=-1, thres=200, cur=-10):
    if action_dur > thres and cur_action != -1:
        return lambda t:cur 
    else:
        return lambda t:0


        
        
        
            
