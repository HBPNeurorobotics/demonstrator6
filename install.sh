#!/bin/bash

# If you are using the NRP in a docker container,
# this script should be run on every reset/update of the backend

sudo apt update
sudo apt install ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-controller-manager ros-melodic-robot-state-publisher

cp -r ./Models/* $NRP_MODELS_DIRECTORY
cd $NRP_MODELS_DIRECTORY
./create-symlinks.sh
cd -

source ~/.opt/platform_venv/bin/activate
pip install torch
