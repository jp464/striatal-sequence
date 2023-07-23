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

def determine_action(state, patterns, thres=0.6, correlations=False):
    if correlations:
        measure = [scipy.stats.pearsonr(state, p)[0] for p in patterns]
    else:
        measure = [state.T.dot(p) / state.shape[0] for p in patterns]
    maxind = np.argmax(measure)
    if measure[maxind] > thres and abs(np.sort(measure)[-2]-measure[maxind]) > .01:
        return maxind
    return -1

def hyperpolarizing_current(action_dur, cur_action=-1, thres=200, cur=-10):
    if action_dur > thres and cur_action != -1:
        return lambda t:cur 
    else:
        return lambda t:0

def learning_performance(mouse, correlations, interval):
    correlations = np.transpose(correlations)
    a0 = None
    a1 = None
    cnt = 0
    success = 0
    performance = [0]
    for i in range(len(correlations)):
        cur = mouse.get_action(np.argmax(correlations[i]))
        if cur != a1: 
            a0, a1 = a1, cur
            mouse.water(a0, a1)
        mouse.compute_reward(cur)
        if mouse.reward == 1:
            success += 1
        cnt += 1
        
        if cnt % interval == 0:
            performance.append(success/(interval*0.001))
            cnt = 0
            success = 0
    return performance
        
        
        
            
