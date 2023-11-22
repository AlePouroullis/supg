from mailer.hexapod.simulator import Simulator
from SUPGController import SUPGController
from BSUPGController import BSUPGController

import neat
import neat.nn
import numpy as np
import pickle
import multiprocessing
import visualize as vz
import os
import sys
import math

#configure neat for the SUPG CPPN\
#chnage to config_BSUPG or coupled config_SUPG 
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'config_BSUPG')  

# radius, offset, step_height, phase, duty_factor
tripod_gait = [	0.15, 0, 0.05, 0.5, 0.5, # leg 1
				0.15, 0, 0.05, 0.0, 0.5, # leg 2
				0.15, 0, 0.05, 0.5, 0.5, # leg 3
				0.15, 0, 0.05, 0.0, 0.5, # leg 4
				0.15, 0, 0.05, 0.5, 0.5, # leg 5
				0.15, 0, 0.05, 0.0, 0.5] # leg 6

#parallel implementation
def evaluate_gaitP(genome, config):
     # Create CPPN from Genome and configuration file
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)     
        
        leg_params = np.array(tripod_gait).reshape(6, 5)

        # Set up controller
        #BSUPG or coupled SUPG
        try:
           controller = BSUPGController(cppn, [] )
        except:
            
            return 0#, np.zeros(6)
            
        # Initialise Simulator
        simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
        # Step in simulator
        for t in np.arange(0, 5, step=simulator.dt):
            try:
                simulator.step()
            except RuntimeError as collision:

                return 0#, np.zeros(6)

        fitness = simulator.base_pos()[0]   # distance travelled along x axis
        # Terminate Simulator
        simulator.terminate()

        return fitness

#linear implementation
def evaluate_gait(genomes, config, duration=5):
    for genome_id, genome in genomes:
        # Create CPPN from Genome and configuration file
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
               
        leg_params = np.array(tripod_gait).reshape(6, 5)

        # Set up controller
        try:
           controller = BSUPGController(cppn)
        except:
            return 0, np.zeros(6)
            
        # Initialise Simulator
        simulator = Simulator(controller=controller, visualiser=False, collision_fatal=False)
        # Step in simulator
        for t in np.arange(0, duration, step=simulator.dt):
            try:
                simulator.step()
            except RuntimeError as collision:
                fitness = 0, np.zeros(6)
        fitness = simulator.base_pos()[0]  # distance travelled along x axis
        # Terminate Simulator
        simulator.terminate()
        # Assign fitness to genome
        genome.fitness = fitness

def run(gens, file):
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    #checkpoint population:
    tempFilename = 'Checkpoints/neat-checkpoint-' + file
    p = neat.Checkpointer.restore_checkpoint(tempFilename) #C:\Users\Dell\Documents\University\Unversity2022\Thesis\Hexapod Code\Hexapod---SUPG\JSUPG\Checkpoints\neat-checkpoint-TestEBSUPG17
    stats = neat.statistics.StatisticsReporter()
    p.add_reporter(stats)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))
    
    #tempFilename = 'Checkpoints/neat-checkpoint-' + file
    #p.add_reporter(neat.Checkpointer(generation_interval=5000, time_interval_seconds=None, filename_prefix=tempFilename))

    # running in parallel
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_gaitP)

    # Run until a solution is found
    #winner = p.run(evaluate_gait, gens)

    winner = p.run(pe.evaluate, gens)
    return winner, stats


if __name__ == "__main__":

    if not os.path.exists("Output"):
        os.mkdir("Output")
        if not os.path.exists("Output/genomeFitness"):
            os.mkdir("Output/genomeFitness")
        if not os.path.exists("Output/graphs"):
            os.mkdir("Output/graphs")
        if not os.path.exists("Output/bestGenomes"):
            os.mkdir("Output/bestGenomes")
        if not os.path.exists("Output/stats"):
            os.mkdir("Output/stats")
        if not os.path.exists("Output/CPPNS"):
            os.mkdir("Output/CPPNS")
    if not os.path.exists("Pickles"):
        os.mkdir("Pickles")

    numRuns = int(sys.argv[1])
    fileNumber = (sys.argv[2])
    checkpointFile = (sys.argv[3])
    winner, stats = run(numRuns, checkpointFile)

    stats.save_genome_fitness(delimiter=',', filename='Output/genomeFitness/FitnessHistory' + fileNumber + '.csv')
    vz.plot_stats(stats, ylog=False, view=True, filename='Output/graphs/AverageFitness' + fileNumber + '.svg')
    vz.plot_species(stats, view=True, filename='Output/graphs/Speciation' + fileNumber + '.svg')
    
    #create network with winning genome
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    leg_params = np.array(tripod_gait).reshape(6, 5)

    with open('Pickles/SUPG_xor_cppn_test' + fileNumber + '.pkl', 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)
        
    vz.draw_net(config, winner, filename="Output/graphs/NEATWINNER" + fileNumber)