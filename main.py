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
import functools
from loki1Dv import Loki
# from loki.loki1Dv import Loki  # use this when loki is a package
from matplotlib import pyplot as plt
import operator
import pygame

BUTTON_SHIM = True
try:
  import buttons
except ModuleNotFoundError:
  BUTTON_SHIM = False

render_colouring = ['rgb', 'irgb', 'keys','none']
render_texturing = ['flat', 'energy_up', 'energy_down']
display_modes = ['pygame', 'console', 'headless', 'fullscreen',
    'ssh_fullscreen', 'windowed', 'yield_frame']
extraction_methods = ['max', 'mean']
landscape = ['level', 'gradient', 'wobbly', 'rough', 'variable', 'black-white']
extraction_rates = [1./(2**x) for x in range(1, 9)]

def get_config(width=128,
      height=None,
      num_1d_history=48,
      render_colour='rgb',
      render_texture='flat',
      extraction_method='mean',
      lock_mutation=0.0001,
      key_mean_mutation=0.01,
      key_sigma_mutation=0.0001,
      show_lock=False,
      display='windowed',
      extraction_rate=0.1,
      save_frames=False,
      log_data=None,
      landscape='gradient',
      landscape_roughness=4):

  gui = display
  if (display == 'windowed' or display == 'fullscreen'
          or display == 'ssh_fullscreen'):
    gui = 'pygame'
  elif display == 'headless':
    gui = 'yield_frame'

  num_locks = 3

  config = dict(
      num_locks=num_locks,

      smooth_locks=False,
      lock_mutation_level=lock_mutation,
      key_mean_mutation=key_mean_mutation,
      key_sigma_mutation=key_sigma_mutation,
      extraction_method = extraction_method,
      extraction_rate = extraction_rate,
      landscape=landscape,
      landscape_roughness=landscape_roughness,

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
      show_lock=show_lock,

      save_frames=save_frames,
      log_data=log_data,
      )

  config['num_agents'] = functools.reduce(operator.mul, config['map_size'])
  config['world_d'] = len(config['map_size'])
  return config

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


def main(config):
  loki = Loki(config)
  render_colour = config['render_colour']
  render_texture = config['render_texture']

  pygame.init()
  plt.ion()
  show_lock = config['show_lock']

  button_todo = ['render_colour', 'render_texture',
      'extraction_rate',
      'extraction_method',
      'show_lock',
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
            todo = 'show_lock'
          if event.key == pygame.K_x:
            todo = 'change_locks'
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
    if todo == 'show_lock':
      show_lock = not show_lock
      loki.set_render_lock(show_lock)
      print('Show keys: {}'.format(show_lock))
    if todo == 'change_locks':
      loki._change_locks(force=True)
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
  print("""
Loki - Lock and key-based artificial-life simulation generating pretty patterns.
Copyright (C) 2019 Dylan Banarse

This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under certain conditions.
""")

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
  ap.add_argument('-p', '--landscape', help='Landscape peakiness [{}]'.format(
    landscape), default=landscape[1])
  ap.add_argument('-e', '--extraction', help='Extraction method [mean|max]',
    default='mean')
  ap.add_argument('-r', '--extraction_rate',
      help='Energy extraction rate ({})'.format(extraction_rates),
      default=extraction_rates[2])
  ap.add_argument('-n', '--lock_mutation', help='Lock mutation level',
    default=0.0001)
  ap.add_argument('-d', '--display',
      help='Display mode [{}]'.format(display_modes), default=display_modes[0])
  ap.add_argument('-s', '--show_locks', action='store_true',
      help='Show lock landscape')
  ap.add_argument('-f', '--save_frames', action='store_true',
      help='Save frames to outout_v directory')
  ap.add_argument('-l', '--log_data', help='Log data to file', default=None)
  args = vars(ap.parse_args())
  # ap.add_argument('-t', '--testing', help='test_mutateex',
  #     action='store_true')


  width = int(args.get('width'))
  height = None if args.get('height') is None else int(args.get('height'))
  gen_history = int(args.get('gen_history'))
  render_colour = args.get('render_colour')
  render_texture = args.get('render_texture')
  landscape = args.get('landscape')
  extraction = args.get('extraction')
  lock_mutation = float(args.get('lock_mutation'))
  extraction_rate = float(args.get('extraction_rate'))
  display = args.get('display')
  show_lock = args.get('show_res', False)
  save_frames = args.get('save_frames', False)
  log_data = args.get('log_data')

  testing = False
  # testing = args.get('testing')

  config = get_config(width=width,
      height=height,
      num_1d_history=gen_history,
      render_colour=render_colour,
      render_texture=render_texture,
      show_lock=show_lock,
      extraction_method=extraction,
      lock_mutation=lock_mutation,
      display=display,
      extraction_rate=extraction_rate,
      save_frames=save_frames,
      log_data=log_data,
      landscape=landscape)

  print(config)
  # ap = argparse.ArgumentParser()
  # ap.add_argument('-m', '--test_mutate', help='test_mutateex',
  #     action='store_true')
  # args = vars(ap.parse_args())
  if testing:
    test_mutate(config)
  else:
    main(config)
