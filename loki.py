#!/usr/bin/python3

import colorsys
import copy
from matplotlib import pyplot as plt
import numpy as np
import pickle
import pygame
from pygame import surfarray
import random
import scipy.misc
import PIL
from PIL import Image
from scipy.stats import logistic

pygame.init()
plt.ion()


# gui = 'console'
gui = 'headless'
# gui = 'pygame'

max_energy = 1
efficiency = 0.6
energy_drain = 1.0  # 0.99

plot_data = True
save_data = True
save_video_frames = True
save_history_images = True
testing = True
# testing = False

fast = False

if fast:
    plot_data = False
    save_video_frames = False
    save_data = False

if gui == 'headless':
    plot_data = False

if testing:
    land_size = 320
    history = 240
else:
    #land_size = 1680
    #history = 1050
    land_size = 840
    history = 525

display_w = land_size * 2
display_h = history * 2
num_resources = 2
resources = np.zeros(num_resources)
resources[0] = 0.
resources[1] = 0.

resource_mutability = np.ones(resources.shape) * 0.002
sqrt_2_pi = np.sqrt(2 * np.pi)

# pygame.display.toggle_fullscreen()
if gui == 'pygame':
    if testing:
        display = pygame.display.set_mode((display_w, display_h))
    else:
        display = pygame.display.set_mode((display_w, display_h),pygame.FULLSCREEN)

class Agent(object):
    def __init__(self, resource_size):

        # Mutable data
        self._means = np.random.uniform(size=resource_size)
        self._sigmas = np.ones(resource_size) * 4
        self._mutability_means = np.random.uniform(size=resource_size)
        self._mutability_sigmas = np.random.uniform(size=resource_size)
        self._reproduction_threshold = np.random.uniform() * 5
        self._colour = np.random.uniform(size=(3,))
        # self._mutation_level_means = 0.1
        # self._mutation_level_sigmas = 0.1
        self._mutation_level_repro = np.random.uniform()

        # Params
        self._energy = 0
        self._mutation_level_colour = 0.01
        self._age = 0
        self._gen = 0
        #print(self._means)
        #print(self._sigmas)

    def extract_energy(self, resources):
        global max_energy
        # env is list of resources
        dist_squared = np.square(self._means - resources)
        energy = (
                (np.exp(-dist_squared / (2*self._sigmas*self._sigmas)))
                / (self._sigmas * sqrt_2_pi))
        # print('energy', dist_squared, self._sigmas, energy)
        self._energy += energy.max() * efficiency
        self._energy *= energy_drain
        # self._energy = min(max(self._energy, 0.), max_energy)
        if self._energy > max_energy:
            max_energy = self._energy
            # print('max_energy = ', max_energy)
        self._age += 1

    def reproduce_stochastic(self, agents, neighbour_indices):
        if self._energy >= self._reproduction_threshold:
            choose = random.sample(range(len(neighbour_indices)), 1)[0]
            neighbour_idx = neighbour_indices[choose]
            prob = logistic.cdf(self._energy - 6)
            if np.random.uniform() < prob:
                agents[neighbour_idx] = self._make_offspring()
            # else:
            #     self._energy /= 2

    def reproduce_energy(self, agents, neighbour_indices):
        if self._energy >= self._reproduction_threshold:
        # if self._age >= self._reproduction_threshold:
            choose = random.sample(range(len(neighbour_indices)), 1)[0]
            neighbour_idx = neighbour_indices[choose]
            if (agents[neighbour_idx] is None
                or agents[neighbour_idx]._energy < self._energy):
                agents[neighbour_idx] = self._make_offspring()

    def _make_offspring(self):
        clone = copy.deepcopy(self)
        clone.mutate()
        clone._energy /= 2
        clone._gen += 1
        clone._age = 0
        self._energy /= 2
        self._age = 0
        return clone


    def mutate(self):
        mutate_array(self._means, self._mutability_means)
        mutate_array(self._sigmas, self._mutability_sigmas, lower=1.0)
        self._reproduction_threshold = mutate_value(
                self._reproduction_threshold , self._mutation_level_repro, 
                lower=0.0)
        mutate_array(self._colour, self._mutation_level_colour, 
                lower=0.0, higher=1.0, reflect=True)
        mutate_array(self._mutability_means, 0.01, lower=0.0, higher=1.0, 
                reflect=True)
        mutate_array(self._mutability_sigmas, 0.01, lower=0.0, higher=1.0, 
                reflect=True)

        # self._mutation_level_means = mutate_value(
        #         self._mutation_level_means, 0.01, lower=0.0)
        # self._mutation_level_sigmas = mutate_value(
        #         self._mutation_level_sigmas, 0.01, lower=0.0)
        self._mutation_level_repro = mutate_value(
                self._mutation_level_repro, 0.01, lower=0.0)

        
    def gen_to_char(self):
        chars = list('.:-=+*#%@')
        # '.,-+*oO$0#$@
        return chars[self._gen % len(chars)]

    def energy_to_char(self):
        return str(int(self._energy))

    def __rgb(self):
        return self._colour * 255

    @property
    def rgb(self):
        er = 0.9
        self._colour[2] = ((self._energy * er)/max_energy) + (1-er)
        # self._colour[2] = 1-(((self._energy * er)/max_energy) + (1-er))
        sr = 1.0
        self._colour[1] = self._colour[1]*sr + (1-sr)
        return np.array(colorsys.hsv_to_rgb(*self._colour.tolist()))

    def _rgb(self):
        v = self._energy / max_energy
        s = self._colour[1]
        return np.array(colorsys.hsv_to_rgb(
            self._colour[0], self._colour[1], v)) * 255


def mutate_array(arr, level, lower=None, higher=None, reflect=False):
    arr += (np.random.normal(size=arr.shape) * level)
    if lower is not None:
      if reflect:
          arr[arr < lower] = 2 * lower - arr[arr < lower]
      else:
        arr[arr < lower] = lower
    if higher is not None:
      if reflect:
          arr[arr > higher] = 2 * higher - arr[arr > higher]
      else:
        arr[arr > higher] = higher
    return arr

def mutate_value(val, level, lower=None, higher=None):
    val += np.random.normal() * level
    if lower is not None and val < lower:
      val = lower
    if higher is not None and val > higher:
      val = higher
    return val

# agents = [Agent(len(resources)) for _ in range(10)]
world = np.zeros((land_size, num_resources))
# agents = [None for _ in range(land_size)]
agents = [Agent(num_resources) for _ in range(land_size)]
agents[int(land_size/2)] = Agent(num_resources)
# import pdb; pdb.set_trace()

bitmap = np.zeros((land_size, history ,3)).astype(np.uint8)

def step_world(world, incoming_resource, agents):
    world[:] = incoming_resource
    # print('incoming', incoming_resource)
    world_size = world.shape[0]
    indices = list(range(world_size))
    random.shuffle(indices)
    for pos in indices:
        agent = agents[pos]
        resource = world[pos]
        if agent is not None:
          agent.extract_energy(resource)
    deaths = 0
    for pos in indices:
        agent = agents[pos]
        if agent is not None:
            if pos == 0:
                neighbour_indices = [1]
            elif pos == world_size - 1:
                neighbour_indices = [world_size - 2]
            else:
                neighbour_indices = [pos - 1, pos + 1]
            agent.reproduce_stochastic(agents, neighbour_indices)
            # if agent._energy > max_energy:
            #     agents[pos] = None
            #     deaths += 1
            # print(agent._sigmas)
            # print(agent._means)
    if deaths > 0:
        print('deaths', deaths)


def show(world, agents):
    out = ''
    for i in range(world.shape[0]):
        if agents[i] is not None:
            out += agents[i].gen_to_char()
        else:
            out += ' '
    out += ' | '
    for i in range(world.shape[0]):
        if agents[i] is not None:
            out += agents[i].energy_to_char()
        else:
            out += ' '
    print(out)

def draw_agents(bitmap, row, world, agents):
    bitmap[:,row] = np.zeros((3), dtype=np.uint8)
    for i in range(world.shape[0]):
        if agents[i] is not None:
            bitmap[i,row] = agents[i].rgb * 255

def draw_agents_roll(bitmap, world, agents):
    bitmap = np.roll(bitmap, 1, axis=1)
    bitmap[:,0] = np.zeros((3), dtype=np.uint8)
    for i in range(world.shape[0]):
        if agents[i] is not None:
            bitmap[i,0] = agents[i].rgb * 255
    return bitmap
    # print(bitmap[:,row])


def get_data(agents):
    means = []
    mut_means = []
    sigmas = []
    mut_sigmas = []
    reproduction_threshold = []
    energy = []
    for agent in agents:
        if agent is not None:
            means.append(agent._means)
            mut_means.append(agent._mutability_means)
            sigmas.append(agent._sigmas)
            mut_sigmas.append(agent._mutability_sigmas)
            reproduction_threshold.append(agent._reproduction_threshold)
            energy.append(agent._energy)
    return dict(means=means, sigmas=sigmas, 
            reproduction_threshold=reproduction_threshold,
            energy=energy, mut_means=mut_means, mut_sigmas=mut_sigmas)

def stats(vals):
    vals = np.array(vals)
    return vals.min(), vals.mean(), vals.max()

stop = False
#for t in range(1000):
energy_hist = []
max_energy_hist = []
repo_hist = []
mut_means_hist = []
mut_sigmas_hist = []
t = 0
current_res = np.zeros_like(resources)

with open('output/loki_data_t{}.pkl'.format(t), 'wb') as handle:
    data = get_data(agents)
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
two_pi = 2 * np.pi
while True:

    changed = False
    if np.random.uniform() < resource_mutability[0]:
        resources[0] += np.random.uniform(-1,1) * 5
        # resources[0] += np.random.normal() * 5
        # resources[0] += np.random.standard_cauchy() * 5
        changed = True
    if np.random.uniform() < resource_mutability[1]:
        resources[1] += np.random.uniform(-1,1) * 5
        # resources[1] += np.random.normal() * 5
        # resources[1] += np.random.standard_cauchy() * 5
        changed = True
    current_res[0] = resources[0]
    current_res[1] = resources[1]
    if changed:
        print('Resources at {} = {} (mutabilitt {})'.format(
            t, current_res, resource_mutability))
    #current_res[0] = (np.sin((t*two_pi)/300.) + 1.) / 2 + resources[0]
    #current_res[1] = (np.cos((t*two_pi)/500.) + 1.) / 2 + resources[1]



    # resources[2] = (np.cos((t*two_pi)/800.) + 1.)
    # resources[3] = (np.cos((t*two_pi)/1300.) + 1.)
    # resources[4] = (np.cos((t*two_pi)/2100.) + 1.)
    # resources[5] = (np.cos((t*two_pi)/3400.) + 1.)
    if gui == 'pygame':
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stop = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    stop = True
                    break
    if stop:
        break
    max_energy *= 0.99
    step_world(world, current_res, agents)
    if not fast:
        bitmap = draw_agents_roll(bitmap, world, agents)
    else:
        draw_agents(bitmap, t % history, world, agents)
    if gui == 'console':
       show(world, agents)
    elif gui == 'pygame':
        # pygame.transform.scale(final_surf, (width*scale, height*scale), DISPLAYSURF)
        if not fast:
            bbitmap = scipy.misc.imresize(bitmap, (display_w, display_h),
                    interp='nearest')
        else:
            bbitmap = bitmap
        surfarray.blit_array(display, bbitmap)
        pygame.display.flip()

    if save_video_frames and t % int(history/8) == 0:
        img = Image.fromarray(bitmap.swapaxes(0,1))
        img = img.resize((img.width * 2, img.height * 2))
        img.save('output/loki_frame_t{:09d}.png'.format(t))

    if t % history == history - 1:

        if save_history_images:
            img = Image.fromarray(bitmap.swapaxes(0,1))
            img = img.resize((img.width * 2, img.height * 2))
            img.save('output/loki_image_t{:09d}.png'.format(t))

        if plot_data or save_data:
            data = get_data(agents)

            agg = True
            if agg:
                energy_hist.append(stats(data['energy']))
                repo_hist.append(stats(data['reproduction_threshold']))
                mut_means_hist.append([stats(np.array(data['mut_means'])[:,0])[1],
                        stats(np.array(data['mut_means'])[:,1])[1]])
                mut_sigmas_hist.append(stats(data['mut_sigmas']))
            else:
                energy_hist.append(data['energy'])
                repo_hist.append(data['reproduction_threshold'])
                mut_means_hist.append(
                    np.concatenate(
                        (np.array(data['mut_means'])[:,0],
                            np.array(data['mut_means'])[:,1])))
                mut_sigmas_hist.append(np.concatenate(
                    (np.array(data['mut_sigmas'])[:,0],
                        np.array(data['mut_sigmas'])[:,1])))
            max_energy_hist.append(max_energy)
            data['energy_hist'] = energy_hist
            data['max_energy_hist'] = max_energy_hist
            data['repo_hist'] = repo_hist
            data['mut_means_hist'] = mut_means_hist
            data['mut_sigmas_hist'] = mut_sigmas_hist

        if save_data:
            with open(
                    'output/loki_data_t{:09d}.pkl'.format(t), 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if plot_data:
            ax = plt.subplot(3,2,1)
            ax.plot(energy_hist)
            ax.plot(max_energy_hist)
            ax.set_title('energy')
            ax = plt.subplot(3,2,2)
            ax.plot(repo_hist)
            ax.set_title('repro threshold')
            ax = plt.subplot(3,2,3)
            means = np.array(data['means'])
            ax.scatter(means[:,0], means[:,1])
            ax.scatter(current_res[0], current_res[1])
            ax.set_title('means')
            ax = plt.subplot(3,2,4)
            sigmas = np.array(data['sigmas'])
            ax.scatter(sigmas[:,0], sigmas[:,1])
            ax.set_title('sigmas')
            ax = plt.subplot(3,2,5)
            means = np.array(data['mut_means_hist'])
            ax.plot(means)
            ax.set_title('mut means')
            ax.set_ylim([0, 1])
            ax = plt.subplot(3,2,6)
            means = np.array(data['mut_sigmas_hist'])
            ax.plot(means)
            ax.set_title('mut sigmas')
            ax.set_ylim([0, 1])
            plt.tight_layout()
            plt.draw()
            plt.pause(0.0001)
            plt.savefig('output/loki_plot_t{:09d}.png'.format(t)) 
            plt.clf()
    t += 1


