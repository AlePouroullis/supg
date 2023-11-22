import csv

for i in range(10,20):
    file = open(r'FitnessHistoryEBSUPG' + str(i+1) + '.csv')

    csvreader = csv.reader(file)
    
    rows = []
    for row in csvreader:
        rows.append(row)
        
    for row in rows:
        row[0] = str(float(row[0])/5.9030470443880985)    

    with open(r'genomeFitness\FitnessHistoryEBSUPG' + str(i+1) + '.csv', 'w', newline='') as file2:
        writer = csv.writer(file2)
        writer.writerows(rows)


    

