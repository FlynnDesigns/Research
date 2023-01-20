import numpy as np 
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sci

def csv2Array(location):
    layout = pd.read_csv(location)
    layout = np.array(layout, dtype=np.uint8)
    layout = layout[:, 1:]
    return layout

def getMaxTemp(tempField):
    return np.max(tempField)

def loadTemp(location):
    values = np.loadtxt(location)
    values = values.reshape((74,66))
    values = values[5:69, 1:65]
    values = np.rot90(values, 2)
    return values

def getAvgTemp(designField, tempField):
    numSolidPixels = np.sum(designField)
    solidTemps = np.multiply(designField, tempField)
    totalSolidTemp = np.sum(solidTemps)
    avgTemp = totalSolidTemp / numSolidPixels
    return avgTemp

def plot_design(designField, tempField, name, dir):
    maxTemp = getMaxTemp(tempField)
    avgTemp = getAvgTemp(designField, tempField)
    plt.rcParams["figure.autolayout"] = True 
    
    # Subplot 1 settings
    ax1 = plt.subplot(1,2,1)
    ax1.set_title('Density Field')
    im1 = plt.imshow(designField, aspect='equal')
    
    # Subplot 2 settings
    ax2 = plt.subplot(1,2,2)
    ax2.set_title('Temperature Field')
    im2 = plt.imshow(tempField, aspect='equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax, orientation='vertical')
    
    # Main plot settings
    plt.suptitle(name + '\n\n' + "Avg temp =" + str(round(avgTemp,3)) + ", Max temp =" + str(round(maxTemp,3)))
    plt.savefig(dir + name + '.jpg', dpi = 300)
    plt.close('all')

#######################################################################################################################
# Modes GAN, parallel, CNN
mode = 'CNN'
totalSimulations = 200000
#######################################################################################################################

# Directory of the files you want to parse 

######################################################################################################################
temperatures = 'D:\\Research\\0\\'
coordinatesLoc = 'D:\\Research\\Research\\coordinates\\0\\'
best_best ='D:\\Research\\best_best\\'
mat_files = 'D:\\Research\\mat_files\\train\\train\\'

# Peformance metric to beat 
avgTempParallel = 313.608

# Temp variables and initial values 
count = 0
total = 0 
fileName = ''
minTemp = 400
maxTemp = 0
newCount = 0
    
for file in os.listdir(temperatures):
    if file.endswith('.txt'):
        # Loading in the temperature data 
        tempField = loadTemp(temperatures + file)
        
        # Loading in the density field data 
        name = os.path.splitext(file)[0]
        orientation = name.split('_')[0]
        try:
            number = name.split('_')[2]
        except:
            number = name.split('_')[1]

        design = np.zeros((74,66))
        fileNamer = coordinatesLoc + number + ".txt"
        coordinates = np.loadtxt(fileNamer, delimiter=" ")
        coordinates = coordinates - 0.5
        coordinates = np.array(coordinates, dtype=int)

        for i in range (len(coordinates)):
            x = coordinates[i, 0]
            y = coordinates[i, 1]
            design[y, x] = 1
        designField = design[5:69, 1:65]

        # Making sure that the csv files are in the right orientation 
        maxTempTest = 0
        orientationCorrected = 1
        for i in range(4):
            designFieldTest = np.rot90(designField, i)
       
            # Calculating the stats of the design
            avgTemp = getAvgTemp(designFieldTest, tempField)
            maxTemp = getMaxTemp(tempField)  
            if avgTemp > maxTempTest:
                maxTempTest = avgTemp
                orientationCorrected = i

        designField = np.rot90(designField, orientationCorrected)
        
        # Calculating the stats of the design
        avgTemp = getAvgTemp(designField, tempField)
        # print(avgTemp)
        maxTemp = getMaxTemp(tempField)  
        
        # Incrementing the total number of files 
        total = total + 1

        if avgTemp < minTemp:
                    minTemp = avgTemp
                    maxTemp = np.max(tempField)
                    fileName = file 
                    temp = tempField
                    design = designField
                    plot_design(design, temp, number,  best_best)

        # Saving all designs better than parallel to be used in the CNN
        if avgTemp < avgTempParallel:
            if mode == 'CNN':
                # Copying the best images 
                newName = orientation +  '_' + number + '.jpg'

                # Saving the u and F arrays into .mat files 
                mdict = {"u": tempField, "F": designField}
                matFileName = str(count) + '.mat'

                # Creating training sets of data 
                # sci.savemat(training_dir + 'train\\' + matFileName, mdict)
                # with open(training_dir + 'train_val.txt', 'a') as fileMat:
                #     fileMat.write(matFileName + '\n')
                
                # Storing the design with the best performance
                # if avgTemp < minTemp:
                #     minTemp = avgTemp
                #     maxTemp = np.max(tempField)
                #     fileName = file 
                #     temp = tempField
                #     design = designField
                #     plot_design(design, temp, file,  best_best)

            print(count)
            count = count + 1

# Showing stats 
print('Best design: ', str(fileName), ', Min avg temp = ', minTemp, '\n')
print('Total number of designs better than parallel fin:', count, '\\', total, '\n')
print('Failed designs: ', totalSimulations - total, ' or ', 100 - round(total / totalSimulations * 100, 2) , '%')
