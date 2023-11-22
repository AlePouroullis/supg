import copy
import math
from mailer.hexapod.simulator import Simulator
from BSUPGController import BSUPGController
import neat
import neat.nn
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

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

leg_params = np.array(tripod_gait).reshape(6, 5)

def runTrial(pickleNo, dmgLegs):

    with open(r"C:\Users\Dell\Documents\University\Unversity2022\Thesis\Hexapod Code\Hexapod---SUPG\JSUPG\Pickles_2\SUPG_xor_cppn_testEBSUPG" + str(pickleNo +1) + ".pkl", 'rb') as f:
        CPPN = pickle.load(f)
    
    cdl = copy.copy(dmgLegs)
    sdl = copy.copy(dmgLegs)

    controller = BSUPGController(CPPN, cdl)
    simulator = Simulator(controller, follow=True, visualiser=False, collision_fatal=False, failed_legs=sdl)
    
    for t in np.arange(0, 5, step=simulator.dt):
        simulator.step()

    #fitness is displacement from starting position
    ##fitness = math.sqrt(pow((simulator.base_pos()[0]),2) + pow((simulator.base_pos()[1]),2))   # distance travelled along x axis
    fitness = simulator.base_pos()[0]    
        # Terminate Simulator
    simulator.terminate()

    return fitness

def run():
    #scenarios
    S0 = [[]]
    S1 = [[1],[2],[3],[4],[5],[6]]
    S2 = [[1,4],[2,5],[3,6]]
    S3 = [[1,3],[2,4],[3,5],[4,6],[5,1],[6,2]]
    S4 = [[1,2],[2,3],[3,4],[4,5],[5,6],[6,1]]
    arrS = [S0, S1, S2, S3, S4]
    r0 =[]
    r1 =[]
    r2 = []
    r3 = []
    r4 = []
    Results = [r0,r1,r2,r3,r4]
    #load pickle
    for i in range(20):
        #load scenario
        for s in range(5):
            scenario = arrS[s]
            #run each permutation of leg damage and add to array
            #fitnessArr = []
            for y in range(len(scenario)):
                scenarioPerm = scenario[y]
                #fitnessArr.append(runTrial(i, scenarioPerm))
                Results[s].append(runTrial(i, scenarioPerm))
            #get avg of perms within the trial
            #average = np.mean(fitnessArr)
            #Results[s].append(average)   
    return Results


#set up data frame
def setupFrame(filename, Scenarios):
    data = {
        "Trial 1" : Scenarios[0]
        }
    d1 = {
        "Trial 2" : Scenarios[1]
    } 
    d2 = {
        "Trial 3" : Scenarios[2]
    }
    d3 = {
        "Trial 4" : Scenarios[3]
    }
    d4 ={
        "Trial 5" : Scenarios[4]
    }
        


    df = pd.DataFrame(data)
    df1 = pd.DataFrame(d1)
    df2 = pd.DataFrame(d2)
    df3 = pd.DataFrame(d3)
    df4 = pd.DataFrame(d4)

    df = pd.concat([df, df1], axis=1)
    df = pd.concat([df, df2], axis=1)
    df = pd.concat([df, df3], axis=1)
    df = pd.concat([df, df4], axis=1)
    df.to_csv(filename)


if __name__ == "__main__":
    results = run()
    
    filename = r"C:\Users\Dell\Documents\University\Unversity2022\Thesis\Hexapod Code\Hexapod---SUPG\JSUPG\SUPG_experiments\EBSUPGBOX.csv"

    setupFrame(filename, results)