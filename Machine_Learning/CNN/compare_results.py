import numpy as np 
import os
#home = '/home/nathan/Desktop/Outputs/run_1_12_19_one/'
home = '/home/nathan/Desktop/Research/Machine_Learning/CNN/test/'
max = 0 
fileName = ''

for file in os.listdir(home):
    values = np.loadtxt(home + file)
    inlet = np.average(values[0:66])
    outlet = np.average(values[4817:4884])
    deltaT = outlet - inlet 
    if deltaT > max:
        max = deltaT
        fileName = file 

# Printing out the results
print("Max delta T = ", max)
print("File = ", fileName)
