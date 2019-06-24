#!/usr/bin/python3

import colorsys
from enum import IntEnum
import functools
from matplotlib import pyplot as plt
import numpy as np
import operator
import pickle
from PIL import Image
import pygame
from pygame import surfarray
import random
from scipy.stats import logistic


sqrt_2_pi = np.sqrt(2 * np.pi)

class Key(IntEnum):
    mean = 0
    sigma = 1
    mean_mut = 2
    sigma_mut = 3
    energy = 4
    _num = 5


class State(IntEnum):
    energy = 0
    age = 1
    repo_threshold = 2
    repo_threshold_mut = 3
    _colour_start = 4
    red = 4
    green = 5
    blue = 6
    _colour_end = 7
    colour_mut = 7
    _num = 8


config = dict(
        num_resources=2,
        # map_size=(640,),
        map_size=(320,),
        num_1d_history = 240,

        # map_size=(64,48),
        # map_size=(32,24),
        display_size=(640,480),

        # map_size=(1280,),
        # num_1d_history = 720,
        # display_size=(1280,720),

        save_frames=False,
        # gui = 'console',
        # gui = 'headless',
        gui = 'pygame',
        )
config['num_agents'] = functools.reduce(operator.mul, config['map_size'])
config['world_d'] = len(config['map_size'])


def init_agents(config):
    """Creates dict with all agents' key and state data."""

    agent_data = dict(
        keys = np.zeros((config['num_agents'], config['num_resources'], 
            int(Key._num))),
        state = np.zeros((config['num_agents'], int(State._num)))
        )
    keys = agent_data['keys']
    # #agents, #resources
    keys[:,:,Key.mean] = np.random.uniform(size=keys.shape[0:2])  
    keys[:,:,Key.sigma] = np.random.uniform(size=keys.shape[0:2]) * 4
    keys[:,:,Key.mean_mut] = np.random.uniform(size=keys.shape[0:2])  
    keys[:,:,Key.sigma_mut] = np.random.uniform(size=keys.shape[0:2])  
    state = agent_data['state']
    state[:,State.repo_threshold] = np.random.uniform(size=state.shape[0]) * 5
    state[:,State.repo_threshold_mut] = np.random.uniform(size=state.shape[0])
    state[:,State._colour_start:State._colour_end] = np.random.uniform(
            size=(state.shape[0],3))
    return agent_data


def extract_energy(agent_data, resources):
    # env is list of resources
    dist_squared = np.square(agent_data['keys'][:,:,Key.mean] - resources)
    sigmas = agent_data['keys'][:,:,Key.sigma]
    agent_data['keys'][:,:,Key.energy] = (
            (np.exp(-dist_squared / (2 * sigmas * sigmas)))
            / (sigmas * sqrt_2_pi))
    agent_data['state'][:,State.energy] += agent_data[
            'keys'][:,:,Key.energy].max(axis=1) #  * 0.1


def agents_to_render_data(agent_data, render_data, row=0):
    # Updata RGB matrix
    new_render_data = agent_data['state'][
            :,[State.red, State.green, State.blue, State.energy]]
    if new_render_data.shape[0] == render_data.shape[0]:
        # new_render_data is 1D, i.e. a row 
        render_data[:, row] = new_render_data
    else:
        # new_render_data is 2D, i.e. data for the whole render_data map.
        render_data[:] = new_render_data.reshape(render_data.shape)


def render_data_to_bitmap(render_data, bitmap, method='flat_rgb'):
    if method == 'flat_rgb':
        # Just RGB
        bitmap[:] = (render_data[:,:,0:3] * 255).astype(np.uint8)
    elif method == 'energy_up':
        print(render_data[:,:,3].max())
        bitmap[:] = (render_data[:,:,0:3] 
                * (render_data[:,:,3] 
                    / (render_data[:,:,3].max() + 0.001))[:, :, np.newaxis]
                * 255).astype(np.uint8)


def bitmap_to_image(rgb_data, display_size):
    return Image.fromarray(rgb_data.swapaxes(0,1)).resize(display_size)


def display_image(image, display):
    bitmap = np.array(image).swapaxes(0,1).astype(np.uint8)
    surfarray.blit_array(display, bitmap)
    pygame.display.flip()


def memoize(obj):
    cache = obj.cache = {}
    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = obj(*args, **kwargs)
        return cache[key]
    return memoizer


@memoize
def init_indices(map_size):
    indices = []
    neighbours = []
    width = map_size[0]
    if len(map_size) == 1:
        print('Calculating 1D indices.')
        for x in range(1, width - 1):
            indices.append(x)
            neighbours.append([x - 1, x + 1])
    elif len(map_size) == 2:
        print('Calculating 2D indices.')
        height = map_size[1]
        for x in range(1, width - 1):
            for y in range(1, height - 1):
                indices.append(y * width + x)
                neighbours.append([
                    (y - 1) * width + x,
                    y * width + x - 1,
                    y * width + x + 1,
                    (y + 1) * width + x])
    else:
        raise ValueError('map_size dim {} not supported'.format(len(map_size)))
    return indices, neighbours


def replication(agent_data, map_size):
    agent_indices, agent_neighbours = init_indices(config['map_size'])
    indices = list(range(len(agent_indices)))
    random.shuffle(indices)
    for index in indices:
        replicate_stochastic(agent_data, agent_indices[index], 
                agent_neighbours[index]) 


def replicate_stochastic(agent_data, agent_index, neighbours):
    energy = agent_data['state'][agent_index, State.energy]
    reproduction_threshold = agent_data['state'][
            agent_index, State.repo_threshold]
    if energy >= reproduction_threshold:  
        choose = random.sample(range(len(neighbours)), 1)[0]
        neighbour_idx = neighbours[choose]
        if np.random.uniform() < logistic.cdf(energy - 6):
            make_offspring(agent_data, agent_index, neighbour_idx)


def make_offspring(agent_data, agent_index, target_index):
    agent_data['state'][agent_index, State.energy] /= 2
    agent_data['keys'][target_index, :] = agent_data['keys'][agent_index, :]
    agent_data['state'][target_index, :] = agent_data['state'][agent_index, :]

    mutate_array(agent_data['keys'][target_index, :, Key.mean],
            agent_data['keys'][target_index, :, Key.mean_mut])
    mutate_array(agent_data['keys'][target_index, :, Key.mean_mut],
            0.01, lower=0.0, higher=1.0, reflect=True)
    mutate_array(agent_data['keys'][target_index, :, Key.sigma],
            agent_data['keys'][target_index, :, Key.sigma_mut], lower=1.0)
    mutate_array(agent_data['keys'][target_index, :, Key.sigma_mut],
            0.01, lower=0.0, higher=1.0, reflect=True)
    agent_data['state'][target_index, State.repo_threshold] = mutate_value(
            agent_data['state'][target_index, State.repo_threshold],
            agent_data['state'][target_index, State.repo_threshold_mut],
            lower=0.0)
    mutate_array(agent_data['state'][
        agent_index, State._colour_start:State._colour_end], 0.01,
        lower=0.0, higher=1.0, reflect=True)


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


def mutate_value(val, level, lower=None, higher=None):
    val += np.random.normal() * level
    if lower is not None and val < lower:
      val = lower
    if higher is not None and val > higher:
      val = higher
    return val


pygame.init()
plt.ion()

if config['gui'] == 'pygame':
    display = pygame.display.set_mode(config['display_size'])

agent_data = init_agents(config)

# if config['world_d'] == 1:
#     rgb_history = np.zeros((config['map_size'][0], config['num_1d_history'], 3),
#             dtype=np.uint8)

if config['world_d'] == 1:
    render_data = np.zeros((config['map_size'][0], config['num_1d_history'], 4))
    bitmap = np.zeros((config['map_size'][0], config['num_1d_history'], 3),
            dtype=np.uint8)
else:
    render_data = np.zeros((config['map_size'] + (4,)))
    bitmap = np.zeros((config['map_size'] + (3,)),
            dtype=np.uint8)

resources = np.random.uniform(-5, 5, size=config['num_resources'])
t = 0
stop = False
while True:
    if config['gui'] == 'pygame':
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
    
    render_data = np.roll(render_data, 1, axis=1)
    agents_to_render_data(agent_data, render_data)
    if True:
        render_data_to_bitmap(render_data, bitmap, method='energy_up')
    else:
        render_data_to_bitmap(render_data, bitmap, method='flat_rgb')
    if config['gui'] == 'pygame' or config['save_frames']:
        image = bitmap_to_image(bitmap, config['display_size']) 

    if config['gui'] == 'pygame':
        display_image(image, display)
    if config['save_frames']:
        image.save('output_v/loki_frame_t{:09d}.png'.format(t))
    extract_energy(agent_data, resources)
    replication(agent_data, config['map_size'])
    t += 1
    if t == 500:
        break

