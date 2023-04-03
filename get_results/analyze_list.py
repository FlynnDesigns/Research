import numpy as np
import os 

# True list location 
listLocation = "A:\\Research\\Last_minute_paper_stuff\\combined_stats\\real_stats.txt"
predictedLocation = 'D:\\godMode_1000_new_train\\prediction\\predicted_temp.txt'

#@# Reading in the true file data
trueDict = {}
total = []
with open(listLocation, 'r', encoding="UTF-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip('\n')
        line = line.split(', ')
        trueDict[line[0]] = line[1]
        total.append(float(line[1]))

trueKeys = trueDict.keys()
trueKeys = sorted(trueKeys)
sortedTrueDict = {}

for key in trueKeys:
    sortedTrueDict[key] = trueDict[key]
#@# Reading in the true file data

#$# Reading in the predicted data 
predictedDict = {}
with open(predictedLocation, 'r', encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.replace('Design: ', '')
        line = line.replace(', Temp:', '')
        line = line.replace('\n', '')
        line = line.split(' ')
        predictedDict[line[0]] = line[1]

predictedKeys = predictedDict.keys()
predictedKeys = sorted(predictedKeys)
sortedPredictedDict = {}

for key in predictedKeys:
    sortedPredictedDict[key] = predictedDict[key]
#$# Reading in the predicted data 

## Getting pre optimization results 
preOptimDiff = []
for i in range(100):
    try:
        diff = float(sortedPredictedDict[str(i)]) - float(sortedTrueDict[str(i)])
        preOptimDiff.append(diff)
    except:
        pass
print("UNET model accuracy:")
print(f"Mean pre optim diff = {np.mean(preOptimDiff):.3f}")

# Getting post optimization results
postOptimDiff = []
for i in range(100, 200):
    try:
        diff = float(sortedPredictedDict[str(i)]) - float(sortedTrueDict[str(i)])
        postOptimDiff.append(diff)
    except:
        pass
print(f"Mean post optim diff = {np.mean(postOptimDiff):.3f}\n")

# Getting average true improvement 
improvement = []
for i in range(100):
    try:
        diff = abs(float(sortedTrueDict[str(i)]) - float(sortedTrueDict[str(i + 100)]))
        improvement.append(diff)
    except:
        pass
print("Performance stats:")
print(f"Mean true improvement = {np.mean(improvement):.3f}")
print(f"Best temp = {np.min(total):.3f}")
print(f"Largest improvement = {np.max(improvement):.3f}")