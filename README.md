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
