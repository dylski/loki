#!/usr/bin/python3
"""
Loki - Lock and key-based artificial-life simulation generating pretty patterns.
Copyright (C) 2019 Dylan Banarse

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import argparse
from data_logger import DataLogger
# from loki.data_logger import DataLogger  # when a package
from enum import IntEnum
import functools
import io
import math
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pygame
from pygame import surfarray
import random

sqrt_2_pi = np.sqrt(2 * np.pi)


class Key(IntEnum):
  mean = 0  # key per lock
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


def memoize(obj):
  cache = obj.cache = {}
  @functools.wraps(obj)
  def memoizer(*args, **kwargs):
    key = str(args) + str(kwargs)
    if key not in cache:
      cache[key] = obj(*args, **kwargs)
    return cache[key]
  return memoizer


class Loki():
  def __init__(self, config):
    self._config = config
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

    self._init_agents()
    self._init_landscape_locks()

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

    self._data = {}  # For plotting, etc.
    self._data_history_len = self._render_data.shape[1];
    self._repo_energy_stats = []
    # Track mean min, max; sigma min, max
    self._locks_metrics = np.zeros((4, config['num_locks']))
    self._locks_metrics[0] = np.inf
    self._locks_metrics[1] = -np.inf
    self._locks_metrics[2] = np.inf
    self._locks_metrics[3] = -np.inf

    self._data_logger = None
    if config['log_data'] is not None:
      self._data_logger = DataLogger(['sigma'], config['log_data'], 1000)

  def _init_agents(self):
    """Creates dict with all agents' key and state data."""

    self._agent_data = dict(
      keys = np.zeros((self._config['num_agents'],
        self._config['num_locks'], int(Key._num))),
      state = np.zeros((self._config['num_agents'], int(State._num)))
      )
    keys = self._agent_data['keys']
    keys[:,:,Key.mean] = np.random.uniform(size=keys.shape[0:2])
    keys[:,:,Key.sigma] = np.ones(keys.shape[0:2]) * 0.1

    # Same mutation rate for all (although could try random intial mutation
    # rates, e.g. np.random.uniform(size=keys.shape[0:2]) * 0.02

    keys[:,:,Key.mean_mut] = self._config['key_mean_mutation']
    keys[:,:,Key.sigma_mut] = self._config['key_sigma_mutation']

    state = self._agent_data['state']
    state[:,State.repo_threshold] = 5.

    state[:,State.repo_threshold_mut] = np.random.uniform(
            size=state.shape[0]) * self._reproduction_mutation_rate
    state[:,State._colour_start:State._colour_end] = np.random.uniform(
        size=(state.shape[0],3))

  def _init_landscape_locks(self):
    # --- A few different ways of setting up the initial locks
    # Randomised
    if (self._config['landscape'] == 'wobbly'
        or self._config['landscape'] == 'rough'):
      self._locks = np.random.uniform(0, 1,
          size=(self._config['num_agents'], self._config['num_locks']))
      if self._config['landscape'] == 'wobbly':
        window_len = int(self._config['map_size'][0]/8)
        left_off = math.ceil((window_len - 1) / 2)
        right_off = math.ceil((window_len - 2) / 2)
        w = np.ones(window_len,'d')
        for i in range(self._config['num_locks']):
            s = np.r_[self._locks[:,i][window_len-1:0:-1],
                      self._locks[:,i],
                      self._locks[:,i][-2:-window_len-1:-1]]
            self._locks[:,i] = np.convolve(
                    w / w.sum(), s, mode='valid')[left_off : -right_off]

    elif self._config['landscape'] == 'gradient':
      self._locks = np.ones(
          (self._config['num_agents'], self._config['num_locks']))
      self._locks[:,0] = np.linspace(0., 1., self._locks.shape[0])
      self._locks[:,1] = np.linspace(0., 1., self._locks.shape[0])
      self._locks[:,2] = np.linspace(0., 1., self._locks.shape[0])

    elif (self._config['landscape'] == 'level'
        or self._config['landscape'] == 'variable'):
      self._locks = np.ones(
          (self._config['num_agents'], self._config['num_locks']))
      self._locks *= 0.5

      if self._config['landscape'] == 'variable':
        self._locks[:int(self._locks.shape[0]/2),0] = np.linspace(0., 1.,
            int(self._locks.shape[0]/2))
        self._locks[:int(self._locks.shape[0]/2),1] = np.linspace(0., 1.,
            int(self._locks.shape[0]/2))
        self._locks[:int(self._locks.shape[0]/2),2] = np.linspace(0., 1.,
            int(self._locks.shape[0]/2))

    elif self._config['landscape'] == 'black-white':
      self._locks = np.ones(
          (self._config['num_agents'], self._config['num_locks']))
      half = int(self._locks.shape[0] / 2)
      self._locks[0:half,:] = 0
      self._locks[half:,:] = 1
    else:
      raise ValueError('Unkown landscape config {}'.format(
        self._config['landscape']))

  def set_config(self, key, value):
    if key not in self._config:
      raise ValueError('{} not in config'.format(key))
    self._config[key] = value

  def set_render_lock(self, render_lock):
    self._config['show_lock'] = render_lock

  def set_render_colour(self, render_colour):
    self._config['render_colour'] = render_colour

  def set_render_texture(self, render_texture):
    self._config['render_texture'] = render_texture

  def render(self):
    self._render_data = np.roll(self._render_data, 1, axis=1)

    row_offset  = 0
    if self._config['world_d'] == 1 and self._config['show_lock']:
        row_offset = math.ceil(self._config['num_1d_history'] * 0.02)

    self._agents_to_render_data(row=row_offset)
    self._render_data_to_bitmap(
        render_colour=self._config['render_colour'],
        render_texture=self._config['render_texture'])

    if self._config['world_d'] == 1:
      if self._config['show_lock']:
        self._bitmap[:,0:row_offset,:] = np.expand_dims(
                (self._locks[:,0:3] * 255).astype(np.uint8), axis=1)
      else:
        self._bitmap[:,0:row_offset,:] = 0

    if (self._config['gui'] == 'pygame' or self._config['save_frames'] or
        self._config['gui'] == 'yield_frame'):

      image = self._bitmap_to_image(self._config['display_size'])

      if self._config['gui'] == 'pygame':
        self._display_image(np.array(image), self._display)
      if self._config['save_frames']:
        Image.fromarray(self._bitmap.swapaxes(0,1)).resize(
            self._config['display_size']).save(
                'output_v/loki_frame_t{:09d}.png'.format(self._time))
      if self._config['gui'] == 'yield_frame':
        # import pdb; pdb.set_trace()
        image = image.rotate(-90)
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='PNG')
        return imgByteArr.getvalue()

  def step_frame(self):
    self.step()
    return self.render()

  def step(self):
    self._change_locks()
    self._extract_energy()
    self._replication(self._config['map_size'])
    # self._gather_data()
    if self._data_logger:
      self._data_logger.add_data([self._data['sigma_history'][-1].mean()])
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
    # Empty stats causes problems.
    # import pdb; pdb.set_trace()
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

    if self._config['num_locks'] == 1:
      plt.tight_layout()
      plt.draw()
      plt.pause(0.0001)
      return

    plot_i += 1
    ax = plt.subplot(plot_h,plot_w,plot_i)
    ax.plot(self._locks)
    ax.set_title('Lock values')

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
    # Get min and max for lock means.
    m0_min = self._agent_data['keys'][:, :, Key.mean].min(axis=0)
    m0_max = self._agent_data['keys'][:, :, Key.mean].max(axis=0)
    smaller = m0_min < self._locks_metrics[0]
    self._locks_metrics[0][smaller] = m0_min[smaller]
    larger = m0_max > self._locks_metrics[1]
    self._locks_metrics[1][larger] = m0_max[larger]
    ax.scatter(self._agent_data['keys'][:, :, Key.mean][:, 0],
        self._agent_data['keys'][:, :, Key.mean][:, 1])
    ax.set_xlim(self._locks_metrics[0][0], self._locks_metrics[1][0])
    ax.set_ylim(self._locks_metrics[0][1], self._locks_metrics[1][1])
    ax.set_title('Mean')

    plot_i += 1
    ax = plt.subplot(plot_h,plot_w,plot_i)
    # Get min and max for lock sigmas.
    m0_min = self._agent_data['keys'][:, :, Key.sigma].min(axis=0)
    m0_max = self._agent_data['keys'][:, :, Key.sigma].max(axis=0)
    smaller = m0_min < self._locks_metrics[2]
    self._locks_metrics[2][smaller] = m0_min[smaller]
    larger = m0_max > self._locks_metrics[3]
    self._locks_metrics[3][larger] = m0_max[larger]
    ax.scatter(self._agent_data['keys'][:, :, Key.sigma][:, 0],
        self._agent_data['keys'][:, :, Key.sigma][:, 1])
    ax.set_xlim(self._locks_metrics[2][0], self._locks_metrics[3][0])
    ax.set_ylim(self._locks_metrics[2][1], self._locks_metrics[3][1])
    ax.set_title('Sigma')

    plt.tight_layout()
    plt.draw()
    plt.pause(0.0001)
    self._locks_metrics *= 0.9

  def _extract_energy(self):
    # env is list of locks
    dist_squared = np.square(self._agent_data['keys'][:,:,Key.mean]
        - self._locks)
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
    return Image.fromarray(self._bitmap).resize((display_size[1],
      display_size[0]))

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
    # Now locks are expected to be 0 <= x <= 1
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
      # target_index, State._colour_start:State._colour_end], 0.01,
      # lower=0.0, higher=1.0, reflect=True)
      target_index, State._colour_start:State._colour_end], 0.002,
      lower=0.0, higher=1.0, reflect=True, dist='cauchy')

  def _change_locks(self, force=False):
    changed = False
    lock_mutability = np.ones(self._locks.shape[1]) * self._config[
            'lock_mutation_level']

    intensity = 0.1
    roughness = self._config['landscape_roughness']
    if force:
      intesity = 1.0
      roughness = 4
    window_len = int(self._config['map_size'][0]/roughness)
    left_off = math.ceil((window_len - 1) / 2)
    right_off = math.ceil((window_len - 2) / 2)
    w = np.ones(window_len,'d')

    for i in range(self._locks.shape[1]):
      if np.random.uniform() < lock_mutability[i] or force:
        # self._locks[0] = np.random.uniform(-1,1) * 5
        # self._locks[:,i] += np.random.uniform(-0.1, 0.1,
        #         size=(self._locks.shape[0],))

        # Slowly evolving lock with -n < 0.1
        change = np.random.normal(
                size=(self._locks.shape[0],)) * intensity
        s = np.r_[change[window_len-1:0:-1],
                  change, change[-2:-window_len-1:-1]]
        change = np.convolve(
                w / w.sum(), s, mode='valid')[left_off : -right_off]
        self._locks[:,i] += change
        # self._locks[0] += np.random.standard_cauchy() * 5

        self._locks[:,i] = np.clip(self._locks[:,i], 0., 1.)
        changed = True
    # if changed:
    #   print('Lock at {} = {} (mutability {})'.format(
    #     self._time, self._locks, lock_mutability))


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

