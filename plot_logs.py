from matplotlib import pyplot as plt
import numpy as np
import argparse


rough = np.load('data/rough.npy')
rough = rough.reshape(rough.shape[0])
window_len = 101
s=np.r_[rough[window_len-1:0:-1], rough, rough[-2:-window_len-1:-1]]
w=np.ones(window_len,'d')
rough=np.convolve(w/w.sum(),s,mode='valid')

smooth = np.load('data/gradient.npy')
smooth = smooth.reshape(smooth.shape[0])
s=np.r_[smooth[window_len-1:0:-1], smooth, smooth[-2:-window_len-1:-1]]
w=np.ones(window_len,'d')
smooth=np.convolve(w/w.sum(),s,mode='valid')

size = min(len(rough), len(smooth)) - 1
rough = rough[0 : size]
smooth = smooth[0 : size]
rough = rough.reshape((rough.shape[0],1))
smooth = smooth.reshape((smooth.shape[0],1))
data = np.concatenate((rough, smooth), axis=1)
plt.plot(data)
plt.title('Landscape Effect on Generalisation')
plt.xlabel('Steps')
plt.ylabel('Generalisation (sgima)')
plt.legend(['Rough landscape', 'Smooth landscape'])
plt.show()
