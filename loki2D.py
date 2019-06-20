#!/usr/bin/python3

import colorsys
import copy
import math
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

sqrt_2_pi = np.sqrt(2 * np.pi)

# Energy colour, uniform5 delta on res, high res

pygame.init()
plt.ion()



class Agent(object):
    def __init__(self, resource_size):

        # Mutable data
        self._means = np.random.uniform(-5, 5, size=resource_size)
        self._sigmas = np.ones(resource_size) * 4
        self._mutability_means = np.random.uniform(size=resource_size)
        self._mutability_sigmas = np.random.uniform(size=resource_size)
        self._reproduction_threshold = np.random.uniform() * 5
        self._colour = np.random.uniform(size=(3,))
        self._mutation_level_repro = np.random.uniform()

        # Params
        self._energy = 0
        self._mutation_level_colour = 0.01
        self._age = 0
        self._gen = 0

    def extract_energy(self, resources, efficiency, max_energy):
        # env is list of resources
        dist_squared = np.square(self._means - resources)
        energy = (
                (np.exp(-dist_squared / (2*self._sigmas*self._sigmas)))
                / (self._sigmas * sqrt_2_pi))
        # print('energy', dist_squared, self._sigmas, energy)
        self._energy += energy.max() * efficiency
        # self._energy = min(max(self._energy, 0.), max_energy)
        self._age += 1

    def reproduce_stochastic(self, world, agents, neighbour_indices):
        if self._energy >= self._reproduction_threshold:
            choose = random.sample(range(len(neighbour_indices)), 1)[0]
            neighbour_idx = neighbour_indices[choose]
            index = neighbour_idx[0] + neighbour_idx[1] * world.shape[0]
            prob = logistic.cdf(self._energy - 6)
            if np.random.uniform() < prob:
                agents[index] = self._make_offspring()
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

    def rgb(self, max_energy):
        colour_energy = False
        if colour_energy:
            er = 0.9
            self._colour[2] = ((self._energy * er)/max_energy) + (1-er)
            # self._colour[2] = 1-(((self._energy * er)/max_energy) + (1-er))
        else:
            er = 1.0
            self._colour[2] = self._colour[2] * er + (1-er)
        sr = 0.9
        self._colour[1] = self._colour[1] * sr + (1-sr)
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

def neighbours_2D_9(x, y, width, height):
    neighbours = []
    if y > 0:
        # Row above.
        if x > 0:
            neighbours.append((x-1, y-1))
        if x < width - 1:
            neighbours.append((x+1, y-1))
        neighbours.append((x, y-1))
    # Left/right
    if x > 0:
        neighbours.append((x-1, y))
    if x < width - 1:
        neighbours.append((x+1, y))
    # Row below
    if y < height - 1:
        # Row above.
        if x > 0:
            neighbours.append((x-1, y+1))
        if x < width - 1:
            neighbours.append((x+1, y+1))
        neighbours.append((x, y+1))
    return neighbours

def neighbours_2D(x, y, width, height):
    neighbours = []
    if y > 0:
        # above.
        neighbours.append((x, y-1))
    # Left/right
    if x > 0:
        neighbours.append((x-1, y))
    if x < width - 1:
        neighbours.append((x+1, y))
    # Row below
    if y < height - 1:
        # below
        neighbours.append((x, y+1))
    return neighbours


def step_world(world, incoming_resource, agents, efficiency, max_energy):
    world[:,:] = incoming_resource
    # print('incoming', incoming_resource)
    world_width = world.shape[0]
    world_height = world.shape[1]
    indices = list(range(len(agents)))
    random.shuffle(indices)
    for index in indices:
        y = math.floor(index / world_width)
        x = index % world_width
        agent = agents[index]
        resource = world[x][y]
        if agent is not None:
            agent.extract_energy(resource, efficiency, max_energy)
            if agent._energy > max_energy:
                max_energy = agent._energy
                # print('max_energy = ', max_energy)
    deaths = 0
    for index in indices:
        y = math.floor(index / world_width)
        x = index % world_width
        agent = agents[index ]
        if agent is not None:
            neighbour_indices  = neighbours_2D(
                    x, y, world_width, world_height)
            agent.reproduce_stochastic(world, agents, neighbour_indices)
            # if agent._energy < max_energy / 10:
            #     agents[pos] = None
            #     deaths += 1
            # print(agent._sigmas)
            # print(agent._means)
    if deaths > 0:
        print('deaths', deaths)
    return max_energy


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


def draw_agents(bitmap, world, agents, max_energy):
    indices = list(range(len(agents)))
    world_width = world.shape[0]
    world_height = world.shape[1]
    for index in indices:
        y = math.floor(index / world_width)
        x = index % world_width
        if agents[index] is not None:
            bitmap[x,y] = agents[index].rgb(max_energy) * 255


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

def main():
    # gui = 'console'
    gui = 'headless'
    # gui = 'pygame'

    max_energy = 0
    efficiency = 0.6

    plot_data = True
    save_data = True
    save_video_frames = True
    save_land_height_images = True
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
        land_width = 64*4
        land_height = 48*4
    else:
        #land_width = 1680
        #land_height = 1050
        land_width = 840
        land_height = 525

    #display_w = land_width * 2
    #display_h = land_height * 2
    display_w = 640
    display_h = 480
    num_resources = 2
    resources = np.zeros(num_resources)
    resources[0] = 0.
    resources[1] = 0.

    resource_mutability = np.ones(resources.shape) * 0.002

    # pygame.display.toggle_fullscreen()
    if gui == 'pygame':
        if testing:
            display = pygame.display.set_mode((display_w, display_h))
        else:
            display = pygame.display.set_mode((display_w, display_h),pygame.FULLSCREEN)

    # agents = [Agent(len(resources)) for _ in range(10)]
    world = np.zeros((land_width, land_height, num_resources))
    # agents = [None for _ in range(land_width)]
    agents = [Agent(num_resources) for _ in range(land_width * land_height)]
    # agents[int(land_width/2)] = Agent(num_resources)
    # import pdb; pdb.set_trace()

    bitmap = np.zeros((land_width, land_height ,3)).astype(np.uint8)


    stop = False
    #for t in range(1000):
    energy_hist = []
    max_energy_hist = []
    repo_hist = []
    mut_means_hist = []
    mut_sigmas_hist = []
    t = 0
    current_res = np.zeros_like(resources)

    with open('output2D/loki_data_t{}.pkl'.format(t), 'wb') as handle:
        data = get_data(agents)
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    two_pi = 2 * np.pi
    while True:

        changed = False
        if np.random.uniform() < resource_mutability[0]:
            # resources[0] = np.random.uniform(-5,5)
            resources[0] += np.random.uniform(-1,1) * 5
            # resources[0] += np.random.normal() * 5
            # resources[0] += np.random.standard_cauchy() * 5
            changed = True
        if np.random.uniform() < resource_mutability[1]:
            # resources[1] = np.random.uniform(-5,5)
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
        max_energy = step_world(world, current_res, agents, efficiency,
                max_energy)
        draw_agents(bitmap, world, agents, max_energy)
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

        if save_video_frames:
            img = Image.fromarray(bitmap.swapaxes(0,1))
            img = img.resize((display_w, display_h))
            img.save('output2D/loki_frame_t{:09d}.png'.format(t))

        if t % land_height == land_height - 1:

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
                        'output2D/loki_data_t{:09d}.pkl'.format(t), 'wb') as handle:
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
                plt.savefig('output2D/loki_plot_t{:09d}.png'.format(t)) 
                plt.clf()
        t += 1

if __name__== "__main__":
    main()

#import cProfile
#cProfile.run('main()')
