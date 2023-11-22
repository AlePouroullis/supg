Coupled Single Unit Pattern Generators for Gait Evolution and Damage Control in Hexapod Robots

This project evolves coupled SUPG controllers with the NEAT algorithm, whereafter they are used
for robotic locomotion and damage control scenarios.

## Required Python packages:
* numpy
* pybullet
* matplotlib
* sklearn
* mpi4py
* GPy
* scipy

## Simulator:
This project makes of the simulator developed by Mailer et al. in the paper: [Evolving Gaits for Damage Control in a Hexapod Robot](https://dl.acm.org/doi/abs/10.1145/3449639.3459271)
The corresponding code repo for that paper is used within this project in the folder 'mailer'
The simulator class (JSUPG\mailer\hexapod\simulator.py) is used extensively within this project,
and is interacted with by the SUPGController class

## NEAT python:
The controllers evolved in this project were done with the standard NEAT packages and framework, available:  https://github.com/CodeReclaimers/neat-python.git 

## Evolving Controllers:
The coupled SUPG controller and its supporting classes are spread between the following classes -->
* SUPGController
* sNeuron
* sNeuronList

These classes interact with the NEAT framework and are evolved in the following class -->
* example_Run

The NEAT config file is found in -->
* config_SUPG

## Damage Trials and Test Runs:
The experiments were conducted in the following class -->
* Damage_Runs

This class runs the simulator for each permutation of leg damage and stores the values in panda frames, whereafter they are exported to excel files in the following class -->
* toEx

## Converting from the Coupled Architecture to the Standard Architecture:
To change to the standard architecture, one must change the NEAT parameters and reduce the number of inputs from 12 to 3

Additionally, in the SUPGController class, comment out the code in the activation and timer methods to reduce how much input is fed into the CPPN
