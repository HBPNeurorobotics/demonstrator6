#!/bin/bash

# If you are using the NRP in a docker container,
# this script should be run on every reset/update of the backend

sudo apt update
sudo apt install ros-melodic-ros-control ros-melodic-ros-controllers ros-melodic-controller-manager ros-melodic-robot-state-publisher

cp -r kuka_lbr_iiwa_generic_d6/ $NRP_MODELS_DIRECTORY
cd $NRP_MODELS_DIRECTORY
./create-symlinks.sh
cd -

cd $NRP_SOURCE_DIR/gzweb
./deploy.sh -m local
cd -
# source $NRP_SOURCE_DIR/user-scripts/nrp_functions
# build_gzweb

source ~/.opt/platform_venv/bin/activate
pip install torch
