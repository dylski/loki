# Loki - [A-Life On Pi](http://www.alifeonpi.com) project

![cover_gif](https://github.com/dylski/loki/blob/master/rgb_energydown.gif)
A 128-cell evolving 1D world.

## Description
Loki is a 1D (or 2D) grid world holding *agents* that gain
energy based on how well adapted they are to their local
environment. When they have sufficient energy they spawn
an offspring that occupies a neighbouring cell. The
offspring inherits, with mutation, their parent's colour
and genetic *keys*. Their *keys* are used
to unlock energy from the envorinment.

Agents unlock energy through a simple *Lock and Key* mechanism. Each agent has
three unique *key* values and each grid
location has three *lock* values.
The compaibility between an agent's keys and their location's locks determines
the energy they extract at each time step.
The environment's locks can slow evolve over time.

An agent spwans a mutated child into a neighbouring location once its energy
reaches a repoduction threshold. The parent's energy is shared equally between
the two. Colour mutation is minimal so related cells can be identified by similarity of colour.

Go to [Loki](http://www.alifeonpi.com/loki.html) for more details.

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
* R - cycle Energy extraction **R**ate
* E - toggle energy **E**traction mode mean/max
* S - toggle **S**how lock landscape on/off
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
    $ pip3 install flask
    $ python loki_server.py

