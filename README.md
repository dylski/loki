# Loki - [A-Life On Pi](http://www.alifeonpi.com) project

![cover_gif](https://github.com/dylski/loki/blob/master/rgb_energydown.gif)
A 128-cell evolving 1D world.

## Description
Loki is a 1D or 2D grid world holding cells that gain energy based on how well adapted they are to their local environment. When they have sufficient energy they spawn an offspring that occuppies a neighbouring cells. The offspring inherits colour and genetic keyse from its parent with mutation.

A cell's compatibilty with the environment is though a simple Lock-and-Key
mechanism. Cells have three *keys* (simplyfloats) and each grid location has
three *locks* (also floats). The degree to which the cell's three keys match
their environment's three locks determines how much energy they extract at each
time step. The environmental lock values can slow evolve over time.

Once a cell's energy reaches the repoduction threshold they split into two -
spawning a mutated child in a neighbouring grid location.

## Installing
Runs nicely on a Raspberry Pi, using Python3 and PyGame.

I think you just need to run install numpy and pygame, i.e. run the following:

    pip3 install numpy
    pip3 install pygame

## Quick Start
    $ python loki1Dv.py

Once running, keyboard controls are
* C - cycle **C**olour mode
* T - cycle **T**exture mode
* R - cycle **R**esource extraction **R**ate
* E - toggle energy **E**traction mode mean/max
* S - toggle **S**how resource keys on/off
Button SHIM buttons are the same.

## Operation
Can display in a variety of ways:
* In a window on the Desktop
* Fullscreen from the Desktop
* Fullscreen from command line (X server) under sudo
* As streaming web server
* Frame mode for creating videos
* Fullscreen on boot-up

Some settings can be changed on the fly through a keyboard or the **Button SHIM** for keyboard-free control):
* Colour rendering mode: RGB, inverse-RGB, genetic-key values as RGB
* Texture rendering mode: Just colour, modulated by energy, modulated by inverse
  energy
* Energy extraction rate: Cycles from slow to fast
* Energy extration method: Based on best key performance or mean of all keys
* Render on top line the environment lock values as RGB

Recommended operation for simple and cheap install:
* Cheap low-res projector (e.g. 640x480)
* Secondhand picture frame containing white poster against black background
* Raspberry Pi with Button SHIM
* Startup on boot in fullscreen
* Press the Pi's buttons to change modes

## More Details
At the moment your options are to either run

    $ python lock1Dv.py -h

and take a look at all command line arguments. For example, providing a height defines a 2D world.

Alternatively take a look through the code, e.g. at *get_config* and *__init__* as that's where a lot of other stuff is defined that is not accessible through the command line args.

## Start on boot
Edit crontab:

    $ sudo crontab -e

and append the line

    @reboot sh <your_path_to>/launch_loki.sh > /tmp/loki.log 2>&1

Edit launch_loki.sh to use your favourite command line args.

## Web server
    $ python loki_server.py

