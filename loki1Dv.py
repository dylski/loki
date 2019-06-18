#!/usr/bin/python3

import colorsys
from enum import IntEnum
from matplotlib import pyplot as plt
import numpy as np
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
        map_size=(64,48),
        display_size=(640,480),
        # gui = 'console',
        # gui = 'headless',
        gui = 'pygame',
        )

config['num_agents'] = config['map_size'][0] * config['map_size'][1]


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

def render_rgb(agent_data, map_size, display_size):
    rgb_data = agent_data['state'][:,State.colour_start:State.colour_end]
    rgb_data = rgb_data.reshape(map_size + (3,))
    rgb_data = (rgb_data * 255).astype(np.uint8).reshape(map_size + (3,))
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

t = 0
image = render_rgb(agent_data, config['map_size'], config['display_size'])
display_image(image, display)
image.save('output/loki_frame_t{:09d}.png'.format(t))

print(agent_data['state'])
