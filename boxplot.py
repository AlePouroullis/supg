import csv
import matplotlib.pyplot as plt


##open CSUPG box and create trial arrays
fileC = open(r'SUPG_experiments\ECSUPGBOXNormalisedFin.csv')

CTrial1 = []
CTrial2 = []
CTrial3 = []
CTrial4 = []
CTrial5 = []

Clines = csv.reader(fileC, delimiter = ',')
counter = 0

next(Clines)
for row in Clines:
    if row[1]:
        CTrial1.append(float(row[1]))
    if row[2]:
        CTrial2.append(float(row[2]))
    if row[3]:
        CTrial3.append(float(row[3]))
    if row[4]:
        CTrial4.append(float(row[4]))
    if row[5]:
        CTrial5.append(float(row[5]))


#map elites - 40k CPG
fileM= open(r'SUPG_experiments\map-boxNormalisedFin.csv')

MTrial1 = []
MTrial2 = []
MTrial3 = []
MTrial4 = []
MTrial5 = []

Mlines = csv.reader(fileM, delimiter = ',')

next(Mlines)
for row in Mlines:
    if row[1]:
        MTrial1.append(float(row[1]))
    if row[2]:
        MTrial2.append(float(row[2]))
    if row[3]:
        MTrial3.append(float(row[3]))
    if row[4]:
        MTrial4.append(float(row[4]))
    if row[5]:
        MTrial5.append(float(row[5]))

##map elites - 10k neat
fileN= open(r'SUPG_experiments\NEAT10perfsNormalised.csv')

NTrial1 = []
NTrial2 = []
NTrial3 = []
NTrial4 = []
NTrial5 = []

Nlines = csv.reader(fileN, delimiter = ',')

next(Nlines)
for row in Nlines:
    if row[1]:
        NTrial1.append(float(row[1]))
    if row[2]:
        NTrial2.append(float(row[2]))
    if row[3]:
        NTrial3.append(float(row[3]))
    if row[4]:
        NTrial4.append(float(row[4]))
    if row[5]:
        NTrial5.append(float(row[5]))

##map elites - 5k Hyp neat
fileH= open(r'HyperNEAT5perfsNormalised.csv')

HTrial1 = []
HTrial2 = []
HTrial3 = []
HTrial4 = []
HTrial5 = []

Hlines = csv.reader(fileH, delimiter = ',')

next(Hlines)
for row in Hlines:
    if row[1]:
        HTrial1.append(float(row[1]))
    if row[2]:
        HTrial2.append(float(row[2]))
    if row[3]:
        HTrial3.append(float(row[3]))
    if row[4]:
        HTrial4.append(float(row[4]))
    if row[5]:
        HTrial5.append(float(row[5]))

##mrefer
fileR= open(r'ref-boxNormalisedFin.csv')

RTrial1 = []
RTrial2 = []
RTrial3 = []
RTrial4 = []
RTrial5 = []

Rlines = csv.reader(fileR, delimiter = ',')

next(Rlines)
for row in Rlines:
    if row[1]:
        RTrial1.append(float(row[1]))
    if row[2]:
        RTrial2.append(float(row[2]))
    if row[3]:
        RTrial3.append(float(row[3]))
    if row[4]:
        RTrial4.append(float(row[4]))
    if row[5]:
        RTrial5.append(float(row[5]))


dataTest = [CTrial1, CTrial2, CTrial3, CTrial4, CTrial5]
#dataB = [BTrial1, BTrial2, BTrial3, BTrial4, BTrial5]

#data = [CTrial1, BTrial1, CTrial2, BTrial2, CTrial3, BTrial3, CTrial4, BTrial4, CTrial5, BTrial5]

dataM = [CTrial1, MTrial1, NTrial1, HTrial1, RTrial1,
 CTrial2, MTrial2, NTrial2, HTrial2, RTrial2,
 CTrial3, MTrial3, NTrial3, HTrial3, RTrial3,
 CTrial4, MTrial4, NTrial4, HTrial4, RTrial4,
 CTrial5, MTrial5, NTrial5, HTrial5, RTrial5]

fig = plt.figure(figsize =(10, 7))

ax = fig.add_subplot(111)



bp = ax.boxplot(dataM, patch_artist = True,
                notch ='True')

# bp1 = ax.boxplot(dataTest, patch_artist = True,
#                 notch ='True')

# ##make a bp2 for the bSUPG data
# bp2 = ax.boxplot(dataB, patch_artist = True,
#                 notch ='True')

colors = ['darkkhaki', 'royalblue', 'orange', 'green', 'grey']
numboxs = 10
counter = 0

for patch in bp['boxes']:
    patch.set_facecolor(colors[counter%5])
    counter+=1

for whisker in bp['whiskers']:
    whisker.set(color ='#8B008B',
                linewidth = 1.5,
                linestyle =":")

# # for whisker in bp2['whiskers']:
#     whisker.set(color ='#8B008B',
#                 linewidth = 1.5,
#                 linestyle =":")
                
# for patch in bp1['boxes']:
#     patch.set_facecolor('b')

# for patch in bp2['boxes']:
#     patch.set_facecolor('g')

for median in bp['medians']:
    median.set(color ='red',
               linewidth = 3)

# for median in bp2['medians']:
#     median.set(color ='red',
#                linewidth = 3)

for flier in bp['fliers']:
    flier.set(marker ='D',
              color ='#e7298a',
              alpha = 0.5)

# for flier in bp2['fliers']:
#     flier.set(marker ='D',
#               color ='#e7298a',
#               alpha = 0.5)

# plt.set_xticklabels(['Trial1', 'Trial2',
#                     'Trial3', 'Trial4', 'Trial5'])


#legend
ax.legend([bp["boxes"][0], bp["boxes"][1], bp["boxes"][2], bp["boxes"][3], bp["boxes"][4]], ["CSUPG", "ME-40k CPG", "ME-10k NEAT", "ME-5k HyperNEAT", "ME-10K Reference"], loc='upper right')
#ax.legend([bp1["boxes"][0]], ["CSUPG"], loc='upper right')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], ['S0', 'S0','S0','S0', 'S0',
                    'S1', 'S1', 'S1', 'S1','S1',
                    'S2', 'S2', 'S2', 'S2', 'S2',
                    'S3', 'S3', 'S3', 'S3', 'S3',
                    'S4', 'S4', 'S4', 'S4','S4'])
plt.title("Controller Performance")
plt.ylabel('Task Performance')

plt.savefig(r'BoxPlotComparative5Fin.png')
plt.show()