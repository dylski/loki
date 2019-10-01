#!/usr/bin/python3
import colorsys
from enum import IntEnum
import functools
# import getopt
import math
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
import argparse


# TODO Negative resources don't work!!!
# raise ValueError("Negative resources don't work!")

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


render_methods = ['flat',
  'energy_up', 'rgb_energy_up', 'irgb_energy_up',
  'energy_down', 'rgb_energy_down', 'irgb_energy_down']

display_modes = ['pygame', 'console', 'headless', 'fullscreen', 'windowed',
    'yield_frame']


def memoize(obj):
  cache = obj.cache = {}
  @functools.wraps(obj)
  def memoizer(*args, **kwargs):
    key = str(args) + str(kwargs)
    if key not in cache:
      cache[key] = obj(*args, **kwargs)
    return cache[key]
  return memoizer


def get_config(width=128,
      height=None,
      num_1d_history=48,
      render_method='flat',
      extraction_method='mean',
      resource_mutation=0.00,
      display='windowed'):

  gui = display
  if (display == 'windowed' or display == 'fullscreen'):
    gui = 'pygame'
  elif display == 'headless':
    gui = 'yield_frame'

  config = dict(
      num_resources=2,
      resource_mutation_level=resource_mutation,
      extraction_method = extraction_method,

      map_size=(width,) if height is None else (width, height),
      # map_size=(640,),
      # map_size=(1280,),

      num_1d_history=num_1d_history,
      # num_1d_history = 720,

      display_size=(650, 410),
      # display_size=(640, 480),
      # display_size=(1280,720),

      gui=gui,
      display=display,

      save_frames=False,
      )

  config['num_agents'] = functools.reduce(operator.mul, config['map_size'])
  config['world_d'] = len(config['map_size'])
  return config


class Loki():
  def __init__(self, config):
    print(config)
    self._time = 0
    if config['gui'] == 'pygame':
      if config['display'] == 'fullscreen':
        self._display = pygame.display.set_mode(
          config['display_size'],pygame.FULLSCREEN)
      else:
        self._display = pygame.display.set_mode(config['display_size'])

    self._agent_data = self._init_agents(config)

    if config['world_d'] == 1:
      self._render_data = np.zeros((config['map_size'][0],
          config['num_1d_history'], 4))
      self._bitmap = np.zeros((config['map_size'][0],
          config['num_1d_history'], 3),
        dtype=np.uint8)
    else:
      self._render_data = np.zeros((config['map_size'] + (4,)))
      self._bitmap = np.zeros((config['map_size'] + (3,)),
        dtype=np.uint8)
    # self._resources = np.zeros(config['num_resources'])
    # HACK FOR TESTING
    self._resources = np.random.uniform(-3, 3,
            size=(config['num_agents'], config['num_resources']))
    window_len = config['map_size'][0]
    left_off = math.ceil((window_len - 1) / 2)
    right_off = math.ceil((window_len - 2) / 2)
    s = np.r_[self._resources[:,0][window_len-1:0:-1],
              self._resources[:,0],
              self._resources[:,0][-2:-window_len-1:-1]]
    w = np.ones(window_len,'d')
    # import pdb; pdb.set_trace()
    self._resources[:,0] = np.convolve(
            w / w.sum(), s, mode='valid')[left_off : -right_off]
    s = np.r_[self._resources[:,1][window_len-1:0:-1],
              self._resources[:,1],
              self._resources[:,1][-2:-window_len-1:-1]]
    w = np.ones(window_len,'d')
    self._resources[:,1] = np.convolve(
            w / w.sum(), s, mode='valid')[left_off : -right_off]

    self._config = config
    self._data = {}  # For plotting, etc.
    # mean min, max; sigma min, max
    self._resources_metrics = np.zeros((4, config['num_resources']))
    self._resources_metrics[0] = np.inf
    self._resources_metrics[1] = -np.inf
    self._resources_metrics[2] = np.inf
    self._resources_metrics[3] = -np.inf
    print('Resources: ', self._resources)

  def render(self, render_method):
    self._render_data = np.roll(self._render_data, 1, axis=1)
    self._agents_to_render_data()
    self._render_data_to_bitmap(method=render_method)
    if (self._config['gui'] == 'pygame' or self._config['save_frames'] or
        self._config['gui'] == 'yield_frame'):

      image = self._bitmap_to_image(self._config['display_size'])

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
    self._change_resources()
    self._extract_energy()
    self._replication(self._config['map_size'])
    self._gather_data()
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
    # keys[:,:,Key.sigma] = np.random.uniform(size=keys.shape[0:2]) * 4
    keys[:,:,Key.sigma] = np.ones(keys.shape[0:2])
    keys[:,:,Key.mean_mut] = np.random.uniform(size=keys.shape[0:2]) * 0.1
    keys[:,:,Key.sigma_mut] = np.random.uniform(size=keys.shape[0:2]) * 0.1
    state = agent_data['state']
    state[:,State.repo_threshold] = np.random.uniform(size=state.shape[0]) * 5
    state[:,State.repo_threshold_mut] = np.random.uniform(size=state.shape[0])
    state[:,State._colour_start:State._colour_end] = np.random.uniform(
        size=(state.shape[0],3))
    return agent_data

  def _gather_data(self):
    if 'energy_history' not in self._data:
      self._data['energy_history'] = []
    self._data['energy_history'].append(
        self._agent_data['state'][:, State.energy])
    if 'mean_history' not in self._data:
      self._data['mean_history'] = []
    self._data['mean_history'].append(
        self._agent_data['keys'][:, :, Key.mean].mean(axis=0))
    if 'sigma_history' not in self._data:
      self._data['sigma_history'] = []
    self._data['sigma_history'].append(
        self._agent_data['keys'][:, :, Key.sigma].mean(axis=0))

  def plot_data(self):
    plt.clf()
    # import pdb; pdb.set_trace()
    ax = plt.subplot(3,2,1)
    ax.plot(np.array(self._data['mean_history'])[:])
    ax.set_title('Mean history')
    ax = plt.subplot(3,2,2)
    ax.plot(np.array(self._data['sigma_history'])[:])
    ax.set_title('Sigma history')

    if self._config['num_resources'] == 1:
      plt.tight_layout()
      plt.draw()
      plt.pause(0.0001)
      return

    ax = plt.subplot(3,2,3)
    ax.hist(self._agent_data['state'][:, State.energy], 20)
    ax = plt.subplot(3,2,4)
    ax.plot(self._resources)
    # import pdb; pdb.set_trace()
    ax = plt.subplot(3,2,5)
    # Get min and max for resource means.
    m0_min = self._agent_data['keys'][:, :, Key.mean].min(axis=0)
    m0_max = self._agent_data['keys'][:, :, Key.mean].max(axis=0)
    smaller = m0_min < self._resources_metrics[0]
    self._resources_metrics[0][smaller] = m0_min[smaller]
    larger = m0_max > self._resources_metrics[1]
    self._resources_metrics[1][larger] = m0_max[larger]
    ax.scatter(self._agent_data['keys'][:, :, Key.mean][:, 0],
        self._agent_data['keys'][:, :, Key.mean][:, 1])
    ax.set_xlim(self._resources_metrics[0][0], self._resources_metrics[1][0])
    ax.set_ylim(self._resources_metrics[0][1], self._resources_metrics[1][1])
    ax.set_title('Mean')

    ax = plt.subplot(3,2,6)
    # Get min and max for resource sigmas.
    m0_min = self._agent_data['keys'][:, :, Key.sigma].min(axis=0)
    m0_max = self._agent_data['keys'][:, :, Key.sigma].max(axis=0)
    smaller = m0_min < self._resources_metrics[2]
    self._resources_metrics[2][smaller] = m0_min[smaller]
    larger = m0_max > self._resources_metrics[3]
    self._resources_metrics[3][larger] = m0_max[larger]
    ax.scatter(self._agent_data['keys'][:, :, Key.sigma][:, 0],
        self._agent_data['keys'][:, :, Key.sigma][:, 1])
    ax.set_xlim(self._resources_metrics[2][0], self._resources_metrics[3][0])
    ax.set_ylim(self._resources_metrics[2][1], self._resources_metrics[3][1])
    ax.set_title('Sigma')
    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)
    self._resources_metrics *= 0.9

  def _extract_energy(self):
    # env is list of resources
    # import pdb; pdb.set_trace()
    dist_squared = np.square(self._agent_data['keys'][:,:,Key.mean]
        - self._resources)
    sigmas = self._agent_data['keys'][:,:,Key.sigma]
    self._agent_data['keys'][:,:,Key.energy] = (
        np.exp(-dist_squared / (2 * sigmas * sigmas))
        / (sigmas * sqrt_2_pi)
        )
    if self._config['extraction_method'] == 'max':
        self._agent_data['state'][:,State.energy] += self._agent_data[
            'keys'][:,:,Key.energy].max(axis=1)  * 0.1
    elif self._config['extraction_method'] == 'mean':
        self._agent_data['state'][:,State.energy] += self._agent_data[
            'keys'][:,:,Key.energy].mean(axis=1)  * 0.1
    # print('Max energy', self._agent_data['state'][:,State.energy].max())

  def _agents_to_render_data(self, row=0):
    # Updata RGB matrix
    new_render_data = self._agent_data['state'][
        :,[State.red, State.green, State.blue, State.energy]]
    if new_render_data.shape[0] == self._render_data.shape[0]:
      # new_self._render_data is 1D, i.e. a row
      self._render_data[:, row] = new_render_data
    else:
      # new_render_data is 2D, i.e. data for the whole render_data map.
      self._render_data[:] = new_render_data.reshape(self._render_data.shape)

  def _render_data_to_bitmap(self, method='flat'):
    if method == 'flat':
      # Just RGB
      self._bitmap[:] = (self._render_data[:,:,0:3] * 255).astype(np.uint8)
    elif method == 'energy_up':
      # Scaled min->max
      energy = self._render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      self._bitmap[:] = (np.ones_like(self._render_data[:,:,0:3])
          * normed_energy[:, :, np.newaxis]
          * 255).astype(np.uint8)
    elif method == 'rgb_energy_up':
      # Scaled min->max
      energy = self._render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      self._bitmap[:] = (self._render_data[:,:,0:3]
              * normed_energy[:, :, np.newaxis]
              * 255).astype(np.uint8)
    elif method == 'irgb_energy_up':
      # Scaled min->max
      energy = self._render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      self._bitmap[:] = ((1. - self._render_data[:,:,0:3]
          * normed_energy[:, :, np.newaxis])
          * 255).astype(np.uint8)
    elif method == 'energy_down':
      # Scaled min->max
      energy = self._render_data[:,:,3]
      normed_energy = 1. - (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      self._bitmap[:] = (np.ones_like(self._render_data[:,:,0:3])
          * normed_energy[:, :, np.newaxis]
          * 255).astype(np.uint8)
    elif method == 'rgb_energy_down':
      # Scaled min->max
      energy = self._render_data[:,:,3]
      normed_energy = 1. - (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      self._bitmap[:] = (self._render_data[:,:,0:3]
              * normed_energy[:, :, np.newaxis]
              * 255).astype(np.uint8)
    elif method == 'irgb_energy_down':
      # Scaled 0->max
      energy = self._render_data[:,:,3]
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
      self._bitmap[:] = ((1 - self._render_data[:,:,0:3]
          * normed_energy[:, :, np.newaxis])
          * 255).astype(np.uint8)

  def _bitmap_to_image(self, display_size):
    return Image.fromarray(self._bitmap.swapaxes(0,1)).resize(display_size)

  def _display_image(self, image, display):
    bitmap = np.array(image).swapaxes(0,1).astype(np.uint8)
    surfarray.blit_array(display, bitmap)
    pygame.display.flip()

  def _replication(self, map_size):
    agent_indices, agent_neighbours = init_indices(map_size)
    indices = list(range(len(agent_indices)))
    random.shuffle(indices)
    for index in indices:
      self._replicate_stochastic(agent_indices[index],
          agent_neighbours[index])

  def _replicate_stochastic(self, agent_index, neighbours):
    energy = self._agent_data['state'][agent_index, State.energy]
    reproduction_threshold = self._agent_data['state'][
        agent_index, State.repo_threshold]
    if energy >= reproduction_threshold:
      choose = random.sample(range(len(neighbours)), 1)[0]
      neighbour_idx = neighbours[choose]

      # No regard for neighour's energy
      # if np.random.uniform() < energy / 10:
      #   self._make_offspring(self._agent_data, agent_index, neighbour_idx)

      # No chance if zero energy, 50:50 if matched, and 100% if double.
      opponent_energy = self._agent_data['state'][neighbour_idx, State.energy]
      chance =  energy / (opponent_energy + 0.00001)
      if chance > 2:
        chance = 2
      if np.random.uniform() < chance / 2:
        self._make_offspring(agent_index, neighbour_idx)

  def _make_offspring(self, agent_index, target_index):
    self._agent_data['state'][agent_index, State.energy] /= 2
    self._agent_data['keys'][target_index, :] = self._agent_data[
            'keys'][agent_index, :]
    self._agent_data['state'][target_index, :] = self._agent_data[
            'state'][agent_index, :]
    self._mutate_agent(target_index)

  def _mutate_agent(self, target_index):
    mutate_array(self._agent_data['keys'][target_index, :, Key.mean],
        self._agent_data['keys'][target_index, :, Key.mean_mut])
    #mutate_array(self._agent_data['keys'][target_index, :, Key.mean_mut],
    #    0.01, lower=0.0, higher=1.0, reflect=True)
    mutate_array(self._agent_data['keys'][target_index, :, Key.sigma],
        self._agent_data['keys'][target_index, :, Key.sigma_mut], lower=1.0)
    #mutate_array(self._agent_data['keys'][target_index, :, Key.sigma_mut],
    #    0.01, lower=0.0, higher=1.0, reflect=True)
    self._agent_data['state'][target_index, State.repo_threshold] = mutate_value(
        self._agent_data['state'][target_index, State.repo_threshold],
        self._agent_data['state'][target_index, State.repo_threshold_mut],
        lower=0.0)
    mutate_array(self._agent_data['state'][
      target_index, State._colour_start:State._colour_end], 0.001,
      lower=0.0, higher=1.0, reflect=True)

  def _change_resources(self):
    changed = False
    resource_mutability = np.ones(self._resources.shape[1]) * self._config[
            'resource_mutation_level']
    for i in range(self._resources.shape[1]):
      if np.random.uniform() < resource_mutability[i]:
        # self._resources[0] = np.random.uniform(-1,1) * 5
        self._resources[i] += np.random.uniform(-1,1)
        # self._resources[0] += np.random.normal() * 5
        # self._resources[0] += np.random.standard_cauchy() * 5
        changed = True
    if changed:
      print('Resources at {} = {} (mutabilitt {})'.format(
        self._time, self._resources, resource_mutability))



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

def test_mutate(config):
  loki = Loki(config)
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
          if event.key == pygame.K_p:
            loki.plot_data()
    if stop:
      break

    for i in range(loki._config['num_agents']):
      loki._mutate_agent(i)
    image = loki.render(render_method)



def main(config):
  loki = Loki(config)
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
            print('Render method {}'.format(render_method))
          if event.key == pygame.K_p:
            loki.plot_data()
    if stop:
      break

    loki.step()
    image = loki.render(render_method)


if __name__ == '__main__':
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("-x", "--width", help="Cells wide", default=128)
  ap.add_argument("-y", "--height", help="Cells high (only 2D)", default=None)
  ap.add_argument("-g", "--gen_history", help="Size of history (for 1D)",
          default=48)
  ap.add_argument("-r", "--render_method", help="Render methods [{}]".format(
    render_methods), default=render_methods[0])
  ap.add_argument("-e", "--extraction", help="Extraction method [mean|max]",
    default='max')
  ap.add_argument("-n", "--resource_mutation", help="Resrouce mutation level",
    default=0.)
  ap.add_argument("-d", "--display",
      help="Display mode [{}]".format(display_modes), default=display_modes[0])
  args = vars(ap.parse_args())
  # ap.add_argument("-t", "--testing", help="test_mutateex",
  #     action='store_true')


  width = int(args.get("width"))
  height = None if args.get("height") is None else int(args.get("height"))
  gen_history = int(args.get("gen_history"))
  render_method = args.get("render_method")
  extraction = args.get("extraction")
  resource_mutation = float(args.get("resource_mutation"))
  display = args.get("display")
  testing = False
  # testing = args.get("testing")

  config = get_config(width=width,
      height=height,
      num_1d_history=gen_history,
      render_method=render_method,
      extraction_method=extraction,
      resource_mutation=resource_mutation,
      display=display)

  print(config)
  # ap = argparse.ArgumentParser()
  # ap.add_argument("-m", "--test_mutate", help="test_mutateex",
  #     action='store_true')
  # args = vars(ap.parse_args())
  if testing:
    test_mutate(config)
  else:
    main(config)
