#!/usr/bin/python3
import argparse
import colorsys
from enum import IntEnum
import functools
import io
import math
from matplotlib import pyplot as plt
import numpy as np
import operator
import pygame
import pickle
from PIL import Image
from pygame import surfarray
import random
from scipy.stats import logistic
import sys

BUTTON_SHIM = True
if BUTTON_SHIM:
  import buttons

sqrt_2_pi = np.sqrt(2 * np.pi)


class Key(IntEnum):
  mean = 0  # key per resource
  sigma = 1
  mean_mut = 2
  sigma_mut = 3
  energy = 4  # energy per key
  _num = 5


class State(IntEnum):
  energy = 0  # agent's energy
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

render_colouring = ['rgb', 'irgb', 'keys','none']
render_texturing = ['flat', 'energy_up', 'energy_down']

display_modes = ['pygame', 'console', 'headless', 'fullscreen',
        'ssh_fullscreen', 'windowed', 'yield_frame']

extraction_methods = ['max', 'mean']

extraction_rates = [0.002, 0.01, 0.1, 0.5]

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
      render_colour='rgb',
      render_texture='flat',
      extraction_method='mean',
      resource_mutation=0.0001,
      show_resource=False,
      display='windowed',
      extraction_rate=0.1):

  gui = display
  if (display == 'windowed' or display == 'fullscreen'
          or display == 'ssh_fullscreen'):
    gui = 'pygame'
  elif display == 'headless':
    gui = 'yield_frame'

  num_resources = 3

  config = dict(
      num_resources=num_resources,

      snooth_resources=True,
      resource_mutation_level=resource_mutation,
      extraction_method = extraction_method,
      extraction_rate = extraction_rate,


      map_size=(width,) if height is None else (width, height),
      # map_size=(640,),
      # map_size=(1280,),

      num_1d_history=num_1d_history,
      # num_1d_history = 720,

      display_size=(656, 416),
      # display_size=(640, 480),
      # display_size=(1280,720),

      gui=gui,
      display=display,
      render_colour=render_colour,
      render_texture=render_texture,
      show_resource=show_resource,

      save_frames=False,
      )

  config['num_agents'] = functools.reduce(operator.mul, config['map_size'])
  config['world_d'] = len(config['map_size'])
  return config


class Loki():
  def __init__(self, config):
    self._time = 0
    if config['gui'] == 'pygame':
      if config['display'] == 'fullscreen':
        self._display = pygame.display.set_mode(
          config['display_size'],pygame.FULLSCREEN)
      else:
        self._display = pygame.display.set_mode(config['display_size'])

    # Mutation rates that probably ought to be zero.
    self._mutation_of_mean_mutation_rate = 0.000
    self._mutation_of_sigma_mutation_rate = 0.000
    self._reproduction_mutation_rate = 0.000

    self._agent_data = self._init_agents(config)

    # render_data stores RGB, Energy and first 3 Keys for rendering.
    if config['world_d'] == 1:
      self._render_data = np.zeros((config['map_size'][0],
          config['num_1d_history'], 7))
      self._bitmap = np.zeros((config['map_size'][0],
          config['num_1d_history'], 3),
        dtype=np.uint8)
    else:
      self._render_data = np.zeros((config['map_size'] + (7,)))
      self._bitmap = np.zeros((config['map_size'] + (3,)),
        dtype=np.uint8)

    # self._resources = np.random.uniform(0, 1,
    #         size=(config['num_agents'], config['num_resources']))

    if 'smooth_resources' in config and config['smooth_resources']:
      window_len = int(config['map_size'][0]/4)
      left_off = math.ceil((window_len - 1) / 2)
      right_off = math.ceil((window_len - 2) / 2)
      w = np.ones(window_len,'d')
      for i in range(config['num_resources']):
          s = np.r_[self._resources[:,i][window_len-1:0:-1],
                    self._resources[:,i],
                    self._resources[:,i][-2:-window_len-1:-1]]
          self._resources[:,i] = np.convolve(
                  w / w.sum(), s, mode='valid')[left_off : -right_off]

    # --- TESTING STUFF
    self._resources = np.ones((config['num_agents'], config['num_resources']))
    self._resources[:,0] = np.linspace(0., 1., self._resources.shape[0])
    self._resources[:,1] = np.linspace(0., 1., self._resources.shape[0])
    self._resources[:,2] = np.linspace(0., 1., self._resources.shape[0])

    # half = int(self._resources.shape[0] / 2)
    # self._resources[0:half,:] = 0
    # self._resources[half:,:] = 1
    # --- TESTING STUFF

    self._config = config
    self._data = {}  # For plotting, etc.
    self._data_history_len = self._render_data.shape[1];
    self._repo_energy_stats = []
    # mean min, max; sigma min, max
    self._resources_metrics = np.zeros((4, config['num_resources']))
    self._resources_metrics[0] = np.inf
    self._resources_metrics[1] = -np.inf
    self._resources_metrics[2] = np.inf
    self._resources_metrics[3] = -np.inf
    # print('Resources: ', self._resources)

  def _init_agents(self, config):
    """Creates dict with all agents' key and state data."""

    agent_data = dict(
      keys = np.zeros((config['num_agents'], config['num_resources'],
        int(Key._num))),
      state = np.zeros((config['num_agents'], int(State._num)))
      )
    keys = agent_data['keys']
    # #agents, #resources
    # import pdb; pdb.set_trace()
    keys[:,:,Key.mean] = np.random.uniform(size=keys.shape[0:2])
    keys[:,:,Key.sigma] = np.ones(keys.shape[0:2]) * 0.1

    # Same mutation rate for all
    keys[:,:,Key.mean_mut] = 0.01
    keys[:,:,Key.sigma_mut] = 0.000
    # Although could initialise each with unique mutation rates
    # keys[:,:,Key.mean_mut] = np.random.uniform(size=keys.shape[0:2]) * 0.02
    # keys[:,:,Key.sigma_mut] = np.random.uniform(size=keys.shape[0:2]) * 0.02

    state = agent_data['state']
    state[:,State.repo_threshold] = 5.

    state[:,State.repo_threshold_mut] = np.random.uniform(
            size=state.shape[0]) * self._reproduction_mutation_rate
    state[:,State._colour_start:State._colour_end] = np.random.uniform(
        size=(state.shape[0],3))
    return agent_data

  def set_config(self, key, value):
    if key not in self._config:
      raise ValueError('{} not in config'.format(key))
    self._config[key] = value

  def set_render_resource(self, render_resource):
    self._config['show_resource'] = render_resource

  def set_render_colour(self, render_colour):
    self._config['render_colour'] = render_colour

  def set_render_texture(self, render_texture):
    self._config['render_texture'] = render_texture

  def render(self):
    self._render_data = np.roll(self._render_data, 1, axis=1)

    row_offset  = 0
    if self._config['world_d'] == 1:  #  and show_resource:
        row_offset = math.ceil(self._config['num_1d_history'] * 0.02)

    self._agents_to_render_data(row=row_offset)
    self._render_data_to_bitmap(
        render_colour=self._config['render_colour'],
        render_texture=self._config['render_texture'])

    if self._config['world_d'] == 1:
      if self._config['show_resource']:
        self._bitmap[:,0:row_offset,:] = np.expand_dims(
                (self._resources[:,0:3] * 255).astype(np.uint8), axis=1)
      else:
        self._bitmap[:,0:row_offset,:] = 0

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

  def step_frame(self):
    self.step()
    return self.render()

  def step(self):
    self._change_resources()
    self._extract_energy()
    self._replication(self._config['map_size'])
    self._gather_data()
    self._time += 1

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
    if 'repo_energy_stats' not in self._data:
      self._data['repo_energy_stats'] = []
    self._data['repo_energy_stats'].append(np.array(
        self._repo_energy_stats).mean())
    self._repo_energy_stats = []
    for _, data_list in self._data.items():
      if len(data_list) > self._data_history_len:
        data_list.pop(0)

  def plot_data(self):
    plt.clf()
    plot_h = 4
    plot_w = 2
    plot_i = 1
    ax = plt.subplot(plot_h, plot_w, plot_i)
    ax.plot(np.array(self._data['mean_history'])[:])
    ax.set_title('Mean history')
    plot_i += 1
    ax = plt.subplot(plot_h, plot_w, plot_i)
    ax.plot(np.array(self._data['sigma_history'])[:])
    ax.set_title('Sigma history')

    if self._config['num_resources'] == 1:
      plt.tight_layout()
      plt.draw()
      plt.pause(0.0001)
      return

    plot_i += 1
    ax = plt.subplot(plot_h,plot_w,plot_i)
    ax.plot(self._resources)
    ax.set_title('Resource values')

    plot_i += 1
    ax = plt.subplot(plot_h, plot_w, plot_i)
    ax.plot(self._extracted_energy)
    ax.set_title('Extracted energy')

    plot_i += 1
    ax = plt.subplot(plot_h,plot_w,plot_i)
    ax.hist(self._agent_data['state'][:, State.energy], 20)
    ax.set_title('Histogram of agent energies')

    plot_i += 1
    ax = plt.subplot(plot_h,plot_w,plot_i)
    ax.plot(self._agent_data['keys'][:, :, Key.mean_mut])
    ax.set_title('Key mutability')

    plot_i += 1
    ax = plt.subplot(plot_h,plot_w,plot_i)
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

    plot_i += 1
    ax = plt.subplot(plot_h,plot_w,plot_i)
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
    dist_squared = np.square(self._agent_data['keys'][:,:,Key.mean]
        - self._resources)
    sigmas = self._agent_data['keys'][:,:,Key.sigma]
    self._agent_data['keys'][:,:,Key.energy] = (
        np.exp(-dist_squared / (2 * sigmas * sigmas))
        / (sigmas * sqrt_2_pi)
        )
    if self._config['extraction_method'] == 'max':
        self._extracted_energy = self._agent_data['keys'][:,:,Key.energy
                ].max(axis=1) * self._config['extraction_rate']
    elif self._config['extraction_method'] == 'mean':
        self._extracted_energy = self._agent_data['keys'][:,:,Key.energy
                ].mean(axis=1) * self._config['extraction_rate']
    self._agent_data['state'][:,State.energy] += self._extracted_energy
    # print('Max energy', self._agent_data['state'][:,State.energy].max())

  def _agents_to_render_data(self, row=0):
    # Updata RGB matrix
    new_render_data = np.concatenate((self._agent_data['state'][
        :,[State.red, State.green, State.blue, State.energy]],
        self._agent_data['keys'][:, 0:3, Key.mean]), axis=1)
    if new_render_data.shape[0] == self._render_data.shape[0]:
      # new_self._render_data is 1D, i.e. a row
      self._render_data[:, row] = new_render_data
    else:
      # new_render_data is 2D, i.e. data for the whole render_data map.
      self._render_data[:] = new_render_data.reshape(self._render_data.shape)

  def _render_data_to_bitmap(self, render_colour='rgb', render_texture='flat'):
    energy = self._render_data[:,:,3]
    if render_texture == 'energy_up':
      normed_energy = (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
    elif render_texture == 'energy_down':
      normed_energy = 1. - (energy - np.min(energy)) / (np.ptp(energy) + 0.0001)
    else:
      normed_energy = np.ones_like(energy)
    if (normed_energy > 1.0).any():
      print('WARNING normed_energy max {}'.format(normed_energy.max()))
    if (normed_energy < 0.0).any():
      print('WARNING normed_energy min {}'.format(normed_energy.min()))

    # RGB
    colour = self._render_data[:,:,0:3]
    if render_colour == 'irgb':
      colour = (1 - colour)
    elif render_colour == 'none':
      colour = np.ones_like(colour)
    elif render_colour == 'keys':
      colour = self._render_data[:,:,4:7]
    if (colour > 1.0).any():
      print('WARNING colour max {}'.format(colour.max()))
    if (colour < 0.0).any():
      print('WARNING colour min {}'.format(colour.min()))

    self._bitmap[:] = (colour * normed_energy[:, :, np.newaxis] * 255
        ).astype(np.uint8)

  def _bitmap_to_image(self, display_size):
    return np.array(Image.fromarray(self._bitmap).resize((display_size[1],
      display_size[0])))

  def _display_image(self, image, display):
    surfarray.blit_array(display, image)
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
        self._repo_energy_stats.append(energy)
        self._make_offspring(agent_index, neighbour_idx)

  def _make_offspring(self, agent_index, target_index):
    self._agent_data['state'][agent_index, State.energy] /= 2
    self._agent_data['keys'][target_index, :] = self._agent_data[
            'keys'][agent_index, :]
    self._agent_data['state'][target_index, :] = self._agent_data[
            'state'][agent_index, :]
    self._mutate_agent(target_index)

  def _mutate_agent(self, target_index):
    # Now resources are expected to be 0 <= x <= 1
    lower = 0.0
    higher = 1.0

    check_array(self._agent_data['keys'][target_index, :, Key.mean])
    mutate_array(self._agent_data['keys'][target_index, :, Key.mean],
        self._agent_data['keys'][target_index, :, Key.mean_mut],
        lower=lower, higher=higher, reflect=True, dist='cauchy')
    check_array(self._agent_data['keys'][target_index, :, Key.mean])

    mutate_array(self._agent_data['keys'][target_index, :, Key.sigma],
        self._agent_data['keys'][target_index, :, Key.sigma_mut], lower=0.1,
        dist='cauchy')
    # Turn off mutation of mutation rates
    mutate_array(self._agent_data['keys'][target_index, :, Key.mean_mut],
        self._mutation_of_mean_mutation_rate,
        lower=0.0, higher=1.0, reflect=True, dist='cauchy')
    mutate_array(self._agent_data['keys'][target_index, :, Key.sigma_mut],
        self._mutation_of_sigma_mutation_rate,
        lower=0.0, higher=1.0, reflect=True, dist='cauchy')
    self._agent_data['state'][
            target_index, State.repo_threshold] = mutate_value(
                    self._agent_data['state'][target_index,
                        State.repo_threshold],
                    self._agent_data['state'][target_index,
                        State.repo_threshold_mut],
                    lower=0.0, dist='cauchy')
    mutate_array(self._agent_data['state'][
      target_index, State._colour_start:State._colour_end], 0.01,
      lower=0.0, higher=1.0, reflect=True)

  def _change_resources(self, force=False):
    changed = False
    resource_mutability = np.ones(self._resources.shape[1]) * self._config[
            'resource_mutation_level']

    intensity = 0.1
    roughness = 4
    if force:
      intesity = 1.0
      roughness = 4
    window_len = int(self._config['map_size'][0]/roughness)
    left_off = math.ceil((window_len - 1) / 2)
    right_off = math.ceil((window_len - 2) / 2)
    w = np.ones(window_len,'d')

    for i in range(self._resources.shape[1]):
      if np.random.uniform() < resource_mutability[i] or force:
        # self._resources[0] = np.random.uniform(-1,1) * 5
        # self._resources[:,i] += np.random.uniform(-0.1, 0.1,
        #         size=(self._resources.shape[0],))

        # Slowly evolving resource with -n < 0.1
        change = np.random.normal(
                size=(self._resources.shape[0],)) * intensity
        s = np.r_[change[window_len-1:0:-1],
                  change, change[-2:-window_len-1:-1]]
        change = np.convolve(
                w / w.sum(), s, mode='valid')[left_off : -right_off]
        self._resources[:,i] += change
        # self._resources[0] += np.random.standard_cauchy() * 5

        self._resources[:,i] = np.clip(self._resources[:,i], 0., 1.)
        changed = True
    # if changed:
    #   print('Resources at {} = {} (mutability {})'.format(
    #     self._time, self._resources, resource_mutability))



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


def mutate_array(arr, level, lower=None, higher=None, reflect=False,
        dist='normal'):
  if dist == 'cauchy':
    arr += (np.random.standard_cauchy(size=arr.shape) * level)
  elif dist == 'normal':
    arr += (np.random.normal(size=arr.shape) * level)

  if lower is not None:
   if reflect:
     arr[arr < lower] = 2 * lower - arr[arr < lower]
  if higher is not None:
   if reflect:
     arr[arr > higher] = 2 * higher - arr[arr > higher]

  if lower is not None:
    arr[arr < lower] = lower
  if higher is not None:
    arr[arr > higher] = higher

def mutate_value(val, level, lower=None, higher=None, dist='normal'):
  if dist == 'cauchy':
    val += np.random.standard_cauchy() * level
  elif dist == 'normal':
    val += np.random.normal() * level
  if lower is not None and val < lower:
   val = lower
  if higher is not None and val > higher:
   val = higher
  return val

def test_mutate(config):
  loki = Loki(config)
  render_colour = render_colouring[0]
  render_texture = render_texturing[0]
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
          if event.key == pygame.K_p:
            loki.plot_data()
    if stop:
      break

    for i in range(loki._config['num_agents']):
      loki._mutate_agent(i)
    image = loki.render()

def check_array(arr, lower=0., higher=1.):
  if (arr > 1.0).any():
    print('WARNING max {}'.format(arr.max()))
  if (arr < 0.0).any():
    print('WARNING min {}'.format(arr.min()))



def main(config):
  loki = Loki(config)
  render_colour = config['render_colour']
  render_texture = config['render_texture']

  pygame.init()
  plt.ion()
  show_resource = config['show_resource']

  button_todo = ['render_colour', 'render_texture',
      'extraction_rate',
      'extraction_method',
      'show_resource',
      'no_press']

  while True:
    todo = ''
    if BUTTON_SHIM:
      todo = button_todo[buttons.last_button_release()]
    if config['gui'] == 'pygame':
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          todo = 'stop'
        elif event.type == pygame.KEYDOWN:
          if event.key == pygame.K_ESCAPE:
            todo = 'stop'
          if event.key == pygame.K_p:
            loki.plot_data()
          if event.key == pygame.K_r:
            todo = 'extraction_rate'
          if event.key == pygame.K_e:
            todo = 'extraction_method'
          if event.key == pygame.K_s:
            todo = 'show_resource'
          if event.key == pygame.K_x:
            todo = 'change_resources'
          if event.key == pygame.K_c:
            todo = 'render_colour'
          if event.key == pygame.K_t:
            todo = 'render_texture'

    if todo == 'stop':
      break

    if todo == 'extraction_rate':
      loki.set_config('extraction_rate', extraction_rates[
        (extraction_rates.index(
          config['extraction_rate']) + 1) % len(extraction_rates)])
      print('extraction_rate: {}'.format(config['extraction_rate']))
    if todo == 'extraction_method':
      loki.set_config('extraction_method', extraction_methods[
        (extraction_methods.index(
          config['extraction_method']) + 1) % len(extraction_methods)])
      print('Extraction method: {}'.format(config['extraction_method']))
    if todo == 'show_resource':
      show_resource = not show_resource
      loki.set_render_resource(show_resource)
      print('Show keys: {}'.format(show_resource))
    if todo == 'change_resources':
      loki._change_resources(force=True)
    if todo == 'render_colour':
      render_colour = render_colouring[
          (render_colouring.index(render_colour) + 1)
          % len(render_colouring)]
      loki.set_render_colour(render_colour)
      print('Render method {} {}'.format(render_colour, render_texture))
    if todo == 'render_texture':
      render_texture = render_texturing[
          (render_texturing.index(render_texture) + 1)
          % len(render_texturing)]
      loki.set_render_texture(render_texture)
      print('Render method {} {}'.format(render_colour, render_texture))

    loki.step()
    image = loki.render()


if __name__ == '__main__':
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument('-x', '--width', help='Cells wide', default=128)
  ap.add_argument('-y', '--height', help='Cells high (only 2D)', default=None)
  ap.add_argument('-g', '--gen_history', help='Size of history (for 1D)',
          default=480)
  ap.add_argument('-c', '--render_colour', help='Render colouring [{}]'.format(
    render_colouring), default=render_colouring[0])
  ap.add_argument('-t', '--render_texture', help='Render texture [{}]'.format(
    render_texturing), default=render_texturing[0])
  ap.add_argument('-e', '--extraction', help='Extraction method [mean|max]',
    default='mean')
  ap.add_argument('-r', '--extraction_rate',
      help='Energy extraction rate ({})'.format(extraction_rates),
      default=extraction_rates[2])
  ap.add_argument('-n', '--resource_mutation', help='Resrouce mutation level',
    default=0.0001)
  ap.add_argument('-d', '--display',
      help='Display mode [{}]'.format(display_modes), default=display_modes[0])
  ap.add_argument('-s', '--show_res', action='store_true', help='Show resource')
  args = vars(ap.parse_args())
  # ap.add_argument('-t', '--testing', help='test_mutateex',
  #     action='store_true')


  width = int(args.get('width'))
  height = None if args.get('height') is None else int(args.get('height'))
  gen_history = int(args.get('gen_history'))
  render_colour = args.get('render_colour')
  render_texture = args.get('render_texture')
  extraction = args.get('extraction')
  resource_mutation = float(args.get('resource_mutation'))
  extraction_rate = float(args.get('extraction_rate'))
  display = args.get('display')
  show_resource = args.get('show_res', False)

  testing = False
  # testing = args.get('testing')

  config = get_config(width=width,
      height=height,
      num_1d_history=gen_history,
      render_colour=render_colour,
      render_texture=render_texture,
      show_resource=show_resource,
      extraction_method=extraction,
      resource_mutation=resource_mutation,
      display=display,
      extraction_rate=extraction_rate)

  print(config)
  # ap = argparse.ArgumentParser()
  # ap.add_argument('-m', '--test_mutate', help='test_mutateex',
  #     action='store_true')
  # args = vars(ap.parse_args())
  if testing:
    test_mutate(config)
  else:
    main(config)
