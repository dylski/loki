#!/usr/bin/python3

import colorsys
import copy
import numpy as np
import pickle
import pygame
from pygame import surfarray
import random
import scipy.misc
import PIL
from PIL import Image

pygame.init()

# gui = 'console'
gui = 'headless'
gui = 'pygame'

max_energy = 10
efficiency = 0.5

testing = True
testing = False

if testing:
    land_size = 320
    history = 100
else:
    #land_size = 1680
    #history = 1050
    land_size = 840
    history = 525

display_w = 1680
display_h = 1050
num_resources = 2
resources = np.zeros(num_resources)
sqrt_2_pi = np.sqrt(2 * np.pi)

# pygame.display.toggle_fullscreen()
if gui == 'pygame':
    if testing:
        display = pygame.display.set_mode((land_size, history))
    else:
        display = pygame.display.set_mode((display_w, display_h),pygame.FULLSCREEN)

class Agent(object):
    def __init__(self, resource_size):

        # Mutable data
        self._means = np.random.uniform(size=resource_size)
        self._sigmas = np.ones(resource_size) * 4
        self._mutability = np.random.uniform(size=resource_size)
        self._reproduction_threshold = 3
        self._colour = np.random.uniform(size=(3,))

        # Params
        self._energy = 0
        self._mutation_level_means = 0.1
        self._mutation_level_sigmas = 0.1
        self._mutation_level_repro = 0.1
        self._mutation_level_colour = 0.01
        self._gen = 0
        #print(self._means)
        #print(self._sigmas)

    def extract_energy(self, resources):
        # env is list of resources
        dist_squared = np.square(self._means - resources)
        energy = (
                (np.exp(-dist_squared / (2*self._sigmas*self._sigmas)))
                / (self._sigmas * sqrt_2_pi))
        # print('energy', dist_squared, self._sigmas, energy)
        self._energy += energy.sum() * efficiency
        self._energy = min(max(self._energy, 0.), 10.)

    def reproduce(self, agents, neighbour_indices):
        if self._energy >= self._reproduction_threshold:
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
        self._energy /= 2
        return clone

    def mutate(self):
        self._means += (np.random.normal(size=self._means.shape)
                * self._mutation_level_means)
        self._sigmas += (np.random.normal(size=self._sigmas.shape)
                * self._mutation_level_sigmas)
        too_low = self._sigmas < 1.0
        self._sigmas[too_low] = 1.0
        self._reproduction_threshold += (np.random.normal()
                * self._mutation_level_repro)
        self._colour += (np.random.normal(size=self._colour.shape)
                * self._mutation_level_colour)
        self._colour[self._colour < 0.] = 0.0
        self._colour[self._colour > 1.] = 1.0

        
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
        self._colour[2] = (self._energy * er)/max_energy + (1-er)
        sr = 0.8
        self._colour[1] = self._colour[1]*sr + (1-sr)
        return np.array(colorsys.hsv_to_rgb(*self._colour.tolist())) * 255

    def _rgb(self):
        v = self._energy / max_energy
        s = self._colour[1]
        return np.array(colorsys.hsv_to_rgb(
            self._colour[0], self._colour[1], v)) * 255

# agents = [Agent(len(resources)) for _ in range(10)]
world = np.zeros((history, land_size, num_resources))
agents = [None for _ in range(land_size)]
agents[int(land_size/2)] = Agent(num_resources)
# import pdb; pdb.set_trace()

bitmap = np.zeros((land_size, history ,3)).astype(np.uint8)

def step_world(world, incoming_resource, agents):
    depth = 0
    world[depth] = incoming_resource
    # print('incoming', incoming_resource)
    world_size = world.shape[1]
    indices = list(range(world_size))
    random.shuffle(indices)
    for pos in indices:
        agent = agents[pos]
        resource = world[depth][pos]
        if agent is not None:
          world[depth + 1][pos] = agent.extract_energy(resource)
        else:
          world[depth + 1][pos] = resource
    for pos in indices:
        agent = agents[pos]
        if agent is not None:
            if pos == 0:
                neighbour_indices = [1]
            elif pos == world_size - 1:
                neighbour_indices = [world_size - 2]
            else:
                neighbour_indices = [pos - 1, pos + 1]
            agent.reproduce(agents, neighbour_indices)
            # print(agent._sigmas)
            # print(agent._means)


def show(world, agents):
    out = ''
    for i in range(world.shape[1]):
        if agents[i] is not None:
            out += agents[i].gen_to_char()
        else:
            out += ' '
    out += ' | '
    for i in range(world.shape[1]):
        if agents[i] is not None:
            out += agents[i].energy_to_char()
        else:
            out += ' '
    print(out)

def draw_agents(bitmap, row, world, agents):
    line = np.zeros((world.shape[1], 1, 3))
    for i in range(world.shape[1]):
        if agents[i] is not None:
            bitmap[i,row] = agents[i].rgb

def get_data(agents):
    means = []
    sigmas = []
    reproduction_threshold = []
    for agent in agents:
        if agent is not None:
            means.append(agent._means)
            sigmas.append(agent._sigmas)
            reproduction_threshold.append(agent._reproduction_threshold)
    return dict(means=means, sigmas=sigmas, 
            reproduction_threshold=reproduction_threshold)

resources[0] = 1.
resources[1] = 5.

stop = False
#for t in range(1000):
t = 0
with open('output/loki_data_t{}.pkl'.format(t), 'wb') as handle:
    data = get_data(agents)
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
while True:
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
    step_world(world, resources, agents)
    draw_agents(bitmap, t % history, world, agents)
    if gui == 'console':
       show(world, agents)
    elif gui == 'pygame':
        # pygame.transform.scale(final_surf, (width*scale, height*scale), DISPLAYSURF)
        bbitmap = scipy.misc.imresize(bitmap, (display_w, display_h),
                interp='nearest')
        surfarray.blit_array(display, bbitmap)
        pygame.display.flip()
    if t % history == history - 1:
        img = Image.fromarray(bitmap.swapaxes(0,1))
        img = img.resize((img.width * 2, img.height * 2))
        img.save('output/loki_image_t{}.png'.format(t))

        # scipy.misc.toimage(bitmap.swapaxes(0,1), cmin=0.0, cmax=255.).save('output/loki_image_t{}.png'.format(t))
        with open('output/loki_data_t{}.pkl'.format(t), 'wb') as handle:
            data = get_data(agents)
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if testing:
            break

    #bitmap = np.roll(bitmap, 1)
    #draw_agents(bitmap, 0, world, agents)

    # surf = pygame.surfarray.make_surface(bitmap)
    # display.blit(surf, (0,0))
    # pygame.display.update()

    t += 1


