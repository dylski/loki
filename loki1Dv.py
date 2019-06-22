#!/usr/bin/python3

import colorsys
from enum import IntEnum
from functools import reduce
from matplotlib import pyplot as plt
import numpy as np
import operator
import pickle
from PIL import Image
import pygame
from pygame import surfarray
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
    colour_start = 4
    red = 4
    green = 5
    blue = 6
    colour_end = 7
    _num = 7


config = dict(
        num_resources=2,
        map_size=(320,),
        # map_size=(64,48),
        num_1d_history = 240,
        display_size=(640,480),
        save_frames=False,
        # gui = 'console',
        # gui = 'headless',
        gui = 'pygame',
        )
config['num_agents'] = reduce(operator.mul, config['map_size'])
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
    state[:,State.colour_start:State.colour_end] = np.random.uniform(
            size=(state.shape[0],3))
    return agent_data


def extract_energy(agent_data, resources):
    # env is list of resources
    # import pdb; pdb.set_trace()
    dist_squared = np.square(agent_data['keys'][:,:,Key.mean] - resources)
    sigmas = agent_data['keys'][:,:,Key.sigma]
    agent_data['keys'][:,:,Key.energy] = (
            (np.exp(-dist_squared / (2 * sigmas * sigmas)))
            / (sigmas * sqrt_2_pi))
    agent_data['state'][:,State.energy] += agent_data[
            'keys'][:,:,Key.energy].sum(axis=1) * 0.01
    # print(agent_data['keys'][0,:,Key.energy], 
    #         agent_data['state'][0,State.energy])
    # print(agent_data['keys'][1,:,Key.energy], 
    #         agent_n_explorer_chromotedata['state'][1,State.energy])

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
        bitmap[:] = (render_data[:,:,0:3] 
                * (render_data[:,:,3] 
                    / render_data[:,:,3].max())[:, :, np.newaxis]
                * 255).astype(np.uint8)

def render_rgb(agent_data, map_size):
    rgb_data = agent_data['state'][:,State.colour_start:State.colour_end].copy()
    max_energy = agent_data['state'][:,State.energy].max()
    # max_energy = 6
    norm_energy = agent_data['state'][:,State.energy] / (max_energy + 0.01)
    factor = 0.0
    rgb_data[:,:] = factor + ((1 - factor) 
            * rgb_data[:,:] * norm_energy[:, np.newaxis])
    rgb_data = (rgb_data * 255).astype(np.uint8).reshape(map_size + (3,))
    return rgb_data


def bitmap_to_image(rgb_data, display_size):
    return Image.fromarray(rgb_data.swapaxes(0,1)).resize(display_size)


def rgb_to_image(rgb_data, display_size):
    return Image.fromarray(rgb_data.swapaxes(0,1)).resize(display_size)


def display_image(image, display):
    bitmap = np.array(image).swapaxes(0,1).astype(np.uint8)
    surfarray.blit_array(display, bitmap)
    pygame.display.flip()


pygame.init()
plt.ion()

if config['gui'] == 'pygame':
    display = pygame.display.set_mode(config['display_size'])

agent_data = init_agents(config)
print(agent_data['state'])

# if config['world_d'] == 1:
#     rgb_history = np.zeros((config['map_size'][0], config['num_1d_history'], 3),
#             dtype=np.uint8)

if config['world_d'] == 1:
    render_data = np.zeros((config['map_size'][0], config['num_1d_history'], 4))
    bitmap = np.zeros((config['map_size'][0], config['num_1d_history'], 3),
            dtype=np.uint8)
else:
    render_data = np.zeros((config['map_size'] + (4,)))
    bitmap = np.zeros((config['map_size'] + (4,)),
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
    image = bitmap_to_image(bitmap, config['display_size']) 


    # rgb = render_rgb(agent_data, config['map_size'])
    # if config['world_d'] == 1:
    #     rgb_history[:,t % config['num_1d_history']] = rgb
    #     rgb = rgb_history
    # image = rgb_to_image(rgb, config['display_size']) 

    if config['gui'] == 'pygame':
        display_image(image, display)
    if config['save_frames']:
        image.save('output_v/loki_frame_t{:09d}.png'.format(t))
    extract_energy(agent_data, resources)
    # print(t)
    t += 1

print(agent_data['state'])
