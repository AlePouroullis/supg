#SUPG Controller Class
from mailer.hexapod.simulator import Simulator
from SUPGController import SUPGController
from BSUPGController import BSUPGController 
from SUPGController_Evolution import SUPGController1
from TripodController.TripodController import TripodController
from mailer.hexapod.controllers.simple_controller import SimpleController
import neat
import neat.nn
import numpy as np
import pickle
import multiprocessing

#create NEAT configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'config_SUPG')  


# radius, offset, step_height, phase, duty_factor
tripod_gait = [	0.15, 0, 0.05, 0.5, 0.5, # leg 1
				0.15, 0, 0.05, 0.0, 0.5, # leg 2
				0.15, 0, 0.05, 0.5, 0.5, # leg 3
				0.15, 0, 0.05, 0.0, 0.5, # leg 4
				0.15, 0, 0.05, 0.5, 0.5, # leg 5
				0.15, 0, 0.05, 0.0, 0.5] # leg 6

leg_params = np.array(tripod_gait).reshape(6, 5)


if __name__ == '__main__':
    
#read in pickl
    with open(r"Pickles/SUPG_xor_cppn_testECSUPG11.pkl", 'rb') as f:
        CPPN = pickle.load(f)
        simulator = Simulator(SUPGController(CPPN, []), follow=True, visualiser=True, collision_fatal=False, failed_legs=[])
        while True:
            simulator.step()
            # pause 
            input("Press Enter to continue...")
            


