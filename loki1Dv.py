#!/usr/bin/python3
import colorsys
from enum import IntEnum
import functools
import getopt
import io
from matplotlib import pyplot as plt
import numpy as np
import operator
import pickle
from PIL import Image
import pygame
from pygame import surfarray
import random
from scipy.stats import logistic
import sys


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


def memoize(obj):
  cache = obj.cache = {}
  @functools.wraps(obj)
  def memoizer(*args, **kwargs):
    key = str(args) + str(kwargs)
    if key not in cache:
      cache[key] = obj(*args, **kwargs)
    return cache[key]
  return memoizer


def get_config(argv):
  width = 32
  height = None
  history_len = 48
  render_method = 'flat'
  render_methods = ['flat', 
      'energy_up', 'rgb_energy_up', 'irgb_energy_up', 
      'energy_down', 'rgb_energy_down', 'irgb_energy_down']
  resource_mutation_level = 0.1
  try:
    opts, args = getopt.getopt(argv,'hx:y:p:r:q:',
            ['width=', 'height=', 'past=', 'mut_res='])
  except getopt.GetoptError:
    print('test.py -x <width> [-y <height> | -p <past_history>]')
    print('    -r {}'.format(str(render_methods)))
    print('    -q resource_mutation_level')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print('test.py -x <width> [-y <height> | -p <past_history>]')
      print('    -r {}'.format(str(render_methods)))
      sys.exit()
    elif opt in ('-p', '--past'):
      history_len = int(arg)
    elif opt in ('-q', '--res_mut'):
      resource_mutation_level = float(arg)
    elif opt in ('-x', '--width'):
      width = int(arg)
    elif opt in ('-y', '--height'):
      height = int(arg)
    elif opt in ('-r'):
      render_method = arg

  if height is not None:
    map_size = (width, height)
  else:
    map_size = (width,)

  config = dict(
      num_resources=2,
      resource_mutation_level=resource_mutation_level,
      # map_size=(640,),
      map_size=map_size,
      num_1d_history=history_len,

      # map_size=(1280,),
      # num_1d_history = 720,
      display_size=(650, 410),
      # display_size=(640, 480),
      # display_size=(1280,720),

      save_frames=False,
      # gui = 'console',
      # gui = 'headless',
      gui = 'pygame',
      render_methods=render_methods,
      )

  config['num_agents'] = functools.reduce(operator.mul, config['map_size'])
  config['world_d'] = len(config['map_size'])
  return config


class Loki():
  def __init__(self, config):
    self._time = 0
    if config['gui'] == 'pygame':
      self._display = pygame.display.set_mode(config['display_size'],pygame.FULLSCREEN)
      # self._display = pygame.display.set_mode(config['display_size'])

    self._agent_data = self._init_agents(config)

    if config['world_d'] == 1:
      self._render_data = np.zeros((config['map_size'][0], config['num_1d_history'], 4))
      self._bitmap = np.zeros((config['map_size'][0], config['num_1d_history'], 3),
        dtype=np.uint8)
    else:
      self._render_data = np.zeros((config['map_size'] + (4,)))
      self._bitmap = np.zeros((config['map_size'] + (3,)),
        dtype=np.uint8)
    self._resources = np.random.uniform(-5, 5, size=config['num_resources'])
    self._config = config

  def render(self, render_method):
    self._render_data = np.roll(self._render_data, 1, axis=1)
    self._agents_to_render_data(self._agent_data, self._render_data)
    self._render_data_to_bitmap(self._render_data, self._bitmap, method=render_method)
    if (self._config['gui'] == 'pygame' or self._config['save_frames'] or
        self._config['gui'] == 'yield_frame'):
    
      image = self._bitmap_to_image(self._bitmap, self._config['display_size']) 
  
      if self._config['gui'] == 'pygame':
        self._display_image(image, self._display)
      if self._config['save_frames']:
        image.save('output_v/loki_frame_t{:09d}.png'.format(t))
      if self._config['gui'] == 'yield_frame':
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        return imgByteArr.getvalue()

  def step_frame(self, render_method):
    self.step()
    return self.render(render_method)

  def step(self):
    self._change_resources(self._resources)
    self._extract_energy(self._agent_data, self._resources)
    self._replication(self._agent_data, self._config['map_size'])
    self._time += 1

  def _init_agents(self, config):
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
  
  def _extract_energy(self, agent_data, resources):
    # env is list of resources
    dist_squared = np.square(agent_data['keys'][:,:,Key.mean] - resources)
    sigmas = agent_data['keys'][:,:,Key.sigma]
    agent_data['keys'][:,:,Key.energy] = (
        (np.exp(-dist_squared / (2 * sigmas * sigmas)))
        / (sigmas * sqrt_2_pi))
    agent_data['state'][:,State.energy] += agent_data[
        'keys'][:,:,Key.energy].max(axis=1)  * 0.1
    # print('Max energy', agent_data['state'][:,State.energy].max())
  
  def _agents_to_render_data(self, agent_data, render_data, row=0):
    # Updata RGB matrix
    new_render_data = agent_data['state'][
        :,[State.red, State.green, State.blue, State.energy]]
    if new_render_data.shape[0] == render_data.shape[0]:
      # new_render_data is 1D, i.e. a row 
      render_data[:, row] = new_render_data
    else:
      # new_render_data is 2D, i.e. data for the whole render_data map.
      render_data[:] = new_render_data.reshape(render_data.shape)
  
  def _render_data_to_bitmap(self, render_data, bitmap, method='flat'):
    if method == 'flat':
      # Just RGB
      bitmap[:] = (render_data[:,:,0:3] * 255).astype(np.uint8)
    elif method == 'energy_up':
      # Scaled min->max
      energy = render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      bitmap[:] = (np.ones_like(render_data[:,:,0:3]) 
          * normed_energy[:, :, np.newaxis]
          * 255).astype(np.uint8)
    elif method == 'rgb_energy_up':
      # Scaled min->max
      energy = render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      bitmap[:] = (render_data[:,:,0:3] * normed_energy[:, :, np.newaxis] 
          * 255).astype(np.uint8)
    elif method == 'irgb_energy_up':
      # Scaled min->max
      energy = render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      bitmap[:] = ((1. - render_data[:,:,0:3] * normed_energy[:, :, np.newaxis])
          * 255).astype(np.uint8)
    elif method == 'energy_down':
      # Scaled min->max
      energy = render_data[:,:,3]
      normed_energy = 1. - (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      bitmap[:] = (np.ones_like(render_data[:,:,0:3]) 
          * normed_energy[:, :, np.newaxis]
          * 255).astype(np.uint8)
    elif method == 'rgb_energy_down':
      # Scaled min->max
      energy = render_data[:,:,3]
      normed_energy = 1. - (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      bitmap[:] = (render_data[:,:,0:3] * normed_energy[:, :, np.newaxis] 
          * 255).astype(np.uint8)
    elif method == 'irgb_energy_down':
      # Scaled 0->max
      energy = render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      bitmap[:] = ((1 - render_data[:,:,0:3] * normed_energy[:, :, np.newaxis])
          * 255).astype(np.uint8)
  
  def _bitmap_to_image(self, rgb_data, display_size):
    return Image.fromarray(rgb_data.swapaxes(0,1)).resize(display_size)

  def _display_image(self, image, display):
    bitmap = np.array(image).swapaxes(0,1).astype(np.uint8)
    surfarray.blit_array(display, bitmap)
    pygame.display.flip()

  def _replication(self, agent_data, map_size):
    agent_indices, agent_neighbours = init_indices(map_size)
    indices = list(range(len(agent_indices)))
    random.shuffle(indices)
    for index in indices:
      self._replicate_stochastic(agent_data, agent_indices[index], 
          agent_neighbours[index]) 
  
  def _replicate_stochastic(self, agent_data, agent_index, neighbours):
    energy = agent_data['state'][agent_index, State.energy]
    reproduction_threshold = agent_data['state'][
        agent_index, State.repo_threshold]
    if energy >= reproduction_threshold:  
      choose = random.sample(range(len(neighbours)), 1)[0]
      neighbour_idx = neighbours[choose]
      # if np.random.uniform() < logistic.cdf(energy - 6):
      if np.random.uniform() < energy / 10:
        self._make_offspring(agent_data, agent_index, neighbour_idx)
  
  def _make_offspring(self, agent_data, agent_index, target_index):
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
      target_index, State._colour_start:State._colour_end], 0.01,
      lower=0.0, higher=1.0, reflect=True)
  
  def _change_resources(self, resources):
    changed = False
    resource_mutability = np.ones(resources.shape) * self._config['resource_mutation_level']
    for i in range(len(resources)):
      if np.random.uniform() < resource_mutability[i]:
        resources[0] = np.random.uniform(-1,1) * 5
        # resources[i] += np.random.uniform(-1,1) * 10
        # resources[0] += np.random.normal() * 5
        # resources[0] += np.random.standard_cauchy() * 5
        changed = True
    if changed:
      print('Resources at {} = {} (mutabilitt {})'.format(
        self._time, resources, resource_mutability))

  
@memoize
def init_indices(map_size):
  indices = []
  neighbours = []
  width = map_size[0]
  if len(map_size) == 1:
    print('Calculating 1D indices.')
    indices.append(0)
    neighbours.append([1])
    for x in range(1, width - 1):
      indices.append(x)
      neighbours.append([x - 1, x + 1])
    indices.append(width - 1)
    neighbours.append([width - 2])
  elif len(map_size) == 2:
    print('Calculating 2D indices.')
    height = map_size[1]
    for y in range(height):
      for x in range(width):
        current = x * height + y
        indices.append(current)
        cell_neighbours = []
        if y > 0:
          cell_neighbours.append(current - 1)
        if y < height - 1:
          cell_neighbours.append(current + 1)
        if x > 0:
          cell_neighbours.append(current - height)
        if x < width - 1:
          cell_neighbours.append(current + height)
        neighbours.append(cell_neighbours)
  else:
    raise ValueError('map_size dim {} not supported'.format(len(map_size)))
  return indices, neighbours
  

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


def main(argv):
  config = get_config(argv)
  loki = Loki(config)
  render_methods = config['render_methods']
  render_method = render_methods[0]

  pygame.init()
  plt.ion()

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
          if event.key == pygame.K_r:
            render_method = render_methods[
                (render_methods.index(render_method) + 1) 
                % len(render_methods)]
    if stop:
      break
    
    loki.step()
    image = loki.render(render_method)


# class FrameGenerator():


if __name__ == "__main__":
  main(sys.argv[1:])
