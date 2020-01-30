Demonstrator 6: Visuomotor reaching and grasping
================

This repo contains the files related to the Demonstrator 6 - visuomotor reaching and grasping experiment.

Install
--------

To run this experiment you need the recent version of the [Neurorobotics Platform](https://neurorobotics.net/), a [docker installation](https://neurorobotics.net/local_install.html) also works.

1. Clone this repo into your `~/.opt/.nrpStorage` folder
2. Copy the content of the [Models folder](Models/) into your `Models` and create a symlink:
```bash
cp -r ./Models/* ${NRP_MODELS_DIRECTORY}
cd ${NRP_MODELS_DIRECTORY}
./create-symlinks.sh
```
3. Register this new experiment with the [web frontend](http://localhost:9000/#/esv-private) (click on `scan` or import a zip/folder)


Dependencies
--------

If you are using the NRP docker image, you will need to run these steps everytime you **reset** or **update** the image.

1. Checkout and build the `demonstrator6` branch of `GazeboRosPackages`:
```bash
cd ${HBP}/GazeboRosPackages
git remote add github https://github.com/HBPNeurorobotics/GazeboRosPackages.git
git fetch github
git checkout demonstrator6
catkin_make
```
2. Install the following packages:
```bash
sudo apt install ros-melodic-ros-controllers ros-melodic-controller-manager ros-melodic-robot-state-publisher
```
3. As well as the following python packages available in the NRP experiment:
```bash
source ~/.opt/platform_venv/bin/activate
pip install torch
deactivate
```
