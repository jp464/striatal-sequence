import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import sys
import h5py

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

if __name__ == '__main__':
    filename, A0, A1, A2, A3 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    filename = filename + '-' + A0 + '-' + A1 + '-' + A2 + '-' + A3
    path = '/work/jp464/striatum-sequence/exploration/' + filename + '.npz'
    data = np.load(path) 
    overlaps_ctx = data['overlaps_ctx']
    overlaps_d1 = data['overlaps_d1']
    v_ctx, v_d1 = retrieval_speed(overlaps_ctx), retrieval_speed(overlaps_d1) 
    
    df = pd.read_hdf('/work/jp464/striatum-sequence/output/retrieval_speed.h5', 'data')
    df.loc[len(df)] = [A0, A1, A2, A3, v_ctx, v_d1, v_ctx > 0 and v_d1 > 0]
    df.to_hdf('/work/jp464/striatum-sequence/output/retrieval_speed.h5', 'data')