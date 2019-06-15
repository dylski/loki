#!/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np
import pickle

data_file = 'output_maxOfResMatch_reproRand_relfectMut_better/loki_data_t000235919.pkl'
plot_data = True

with open(data_file, 'rb') as h:
    data = pickle.load(h)

def stats(vals):
    vals = np.array(vals)
    return vals.min(), vals.mean(), vals.max()

energy_hist = []
max_energy_hist = []
repo_hist = []
mut_means_hist = []
mut_sigmas_hist = []
agg = False
if agg:
    energy_hist.append(stats(data['energy']))
    repo_hist.append(stats(data['reproduction_threshold']))
    mut_means_hist.append([stats(np.array(data['mut_means'])[:,0])[1],
            stats(np.array(data['mut_means'])[:,1])[1]])
    mut_sigmas_hist.append(stats(data['mut_sigmas']))
else:
    energy_hist.append(data['energy'])
    repo_hist.append(data['reproduction_threshold'])
    mut_means_hist.append(data['mut_means'])
    mut_sigmas_hist.append(data['mut_sigmas'])
# max_energy_hist.append(max_energy)
data['energy_hist'] = energy_hist
# data['max_energy_hist'] = max_energy_hist
data['repo_hist'] = repo_hist
data['mut_means_hist'] = mut_means_hist
data['mut_sigmas_hist'] = mut_sigmas_hist

if plot_data:
    ax = plt.subplot(3,2,1)
    ax.scatter(range(len(energy_hist[0])), energy_hist)
    # ax.plot(max_energy_hist)
    ax.set_title('energy')
    ax = plt.subplot(3,2,2)
    ax.scatter(range(len(repo_hist[0])), repo_hist)
    ax.set_title('repro threshold')
    ax = plt.subplot(3,2,3)
    means = np.array(data['means'])
    ax.scatter(means[:,0], means[:,1])
    # ax.scatter(current_res[0], current_res[1])
    ax.set_title('means')
    ax = plt.subplot(3,2,4)
    sigmas = np.array(data['sigmas'])
    ax.scatter(sigmas[:,0], sigmas[:,1])
    ax.set_title('sigmas')
    ax = plt.subplot(3,2,5)
    means = np.array(data['mut_means_hist'])[0]
    ax.scatter(means[:,0], means[:,1])
    ax.set_title('mut means')
    ax.set_ylim([0, 1])
    ax = plt.subplot(3,2,6)
    means = np.array(data['mut_sigmas_hist'])[0]
    ax.scatter(means[:,0], means[:,1])
    ax.set_title('mut sigmas')
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.show()
    # plt.draw()
    # plt.pause(0.0001)
    # plt.savefig('output/loki_plot_t{:09d}.png'.format(t)) 
    # plt.clf()

# print(np.array(data['reproduction_threshold']).max())
