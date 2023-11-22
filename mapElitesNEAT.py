import sys
from hexapod.controllers.NEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator
import pymap_elites.map_elites.cvt as cvt_map_elites
import numpy as np
import neat
import pymap_elites.map_elites.common as cm
import pickle
import os
from neat.reporting import ReporterSet

"""
A script to produce the NEAT maps

The script takes two command line arguments:
1) The size of the map to be tested
2) The run/map number
"""

# Config file used for the NEAT maps
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')

# Fitness function
def evaluate_gait(x, duration=5):
    net = neat.nn.FeedForwardNetwork.create(x, config)
    # Reset net

    leg_params = np.array(stationary).reshape(6, 5)
    # Set up controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                ann=net)
    except:
        return 0, np.zeros(6)
    # Initialise Simulator
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    # Step in simulator
    contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
        contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0,
                               posinf=0.0, neginf=0.0)
    # Terminate Simulator
    simulator.terminate()
    # print(difference)
    # fitness = difference
    # Assign fitness to genome
    x.fitness = fitness
    return fitness, descriptor

# Load in initial high performing genomes

def load_genomes(num=200):
    reporters = ReporterSet()
    stagnation = config.stagnation_type(config.stagnation_config, reporters)
    reproduction = config.reproduction_type(config.reproduction_config, reporters, stagnation)

    genomes = reproduction.create_new(config.genome_type, config.genome_config, num)
    print(type(genomes))
    print(type(list(genomes.values())))
    return list(genomes.values())

if __name__ == '__main__':
    mapSize = int(sys.argv[1])
    runNum = (sys.argv[2])
    # Map Elites Parameters
    params = \
        {
            # more of this -> higher-quality CVT
            "cvt_samples": 1000000,
            # we evaluate in batches to parallelise
            "batch_size": 2390,
            # proportion of niches to be filled before starting (400)
            "random_init": 0.01,
            # batch for random initialization
            "random_init_batch": 2390,
            # when to write results (one generation = one batch)
            "dump_period": 1e6,   # Change that
            # do we use several cores?
            "parallel": True,
            # do we cache the result of CVT and reuse?
            "cvt_use_cache": True,
            # min/max of parameters
            "min": 0,
            "max": 1,
        }

    # Load in the high performing genomes
    genomes = load_genomes(int (mapSize * 0.1))
    if not os.path.exists("mapElitesOutput/NEAT/" + runNum + "_" + str(mapSize)):
        os.mkdir("mapElitesOutput/NEAT/" + runNum + "_" + str(mapSize))
    if not os.path.exists("mapElitesOutput/NEAT/" + runNum + "_" + str(mapSize) + "archive"):
        os.mkdir("mapElitesOutput/NEAT/" + runNum + "_" + str(mapSize) + "archive")

    # Run Map Elites
    archive = cvt_map_elites.compute(6, genomes, evaluate_gait, n_niches=mapSize, max_evals=10e6,
                                     log_file=open('mapElitesOutput/NEAT/' + runNum + "_" + str(mapSize) + '/log.dat', 'w'), archive_file='mapElitesOutput/NEAT/' + runNum + "_" + str(mapSize) + "archive" + '/archive', params=params,
                                     variation_operator=cm.neatMutation)