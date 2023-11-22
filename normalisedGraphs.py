import csv
import matplotlib.pyplot as plt

##change to range, 10 through 20 once other half are done
for i in range(10,20):
    file = open(r'C:\Users\Dell\Documents\University\Unversity2022\Thesis\Hexapod Code\Hexapod---SUPG\JSUPG\OutputNormalised\genomeFitness\FitnessHistoryEBSUPG' + str(i+1) + '.csv')

    Fitness = []
    Generation = []
    Std = []
    lines = csv.reader(file, delimiter = ',')
    counter = 0
    for row in lines:
        Fitness.append(float(row[0]))
        Std.append(float(row[1]))
        Generation.append(counter)
        counter += 1

    plt.plot(Generation, Fitness, color = 'r', linestyle = 'solid',
         marker = '.',label = "best")


    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.axis([0, 10000, 0, 1])
    plt.title('BSUPG Population ' + str(i+1) +'\'s Best Fitness')
    plt.grid()
    plt.legend()
    plt.savefig(r'C:\Users\Dell\Documents\University\Unversity2022\Thesis\Hexapod Code\Hexapod---SUPG\JSUPG\OutputNormalised\graphs\AverageFitnessEBSUPG' + str(i+1) + '.png')
    plt.clf()
