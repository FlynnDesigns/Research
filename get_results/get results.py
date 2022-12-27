import numpy as np 
import os
import glob 
import skimage

# Directory of the files you want to parse 
home = 'A:\\Research\\Training data\\raw_data\\Run_5\\extracted\\'

# Directory of the filtered images
filtered_images = 'A:\\Research\\Training data\\raw_data\\Run_5\\filtered_images\\'

# Directory of the paralel fin
parallel = 'A:\\Research\\parallel\\'

# Loading in the temperature data 
values = np.loadtxt(parallel + 'T_parallel.txt')
values = values.reshape((74,66))
values = values[4:68, 1:65]

# Loading in the density field data 
array = np.zeros((64, 64))
array[:,0] = 1
array[:,63] = 1
array[:,3:5] = 1
array[:,8:10] = 1
array[:,13:15] = 1
array[:,18:20] = 1
array[:,23:25] = 1
array[:,28:30] = 1
array[:,33:35] = 1
array[:,38:40] = 1
array[:,43:45] = 1
array[:,48:50] = 1
array[:,53:55] = 1
array[:,58:60] = 1

imageShaped = np.array(array)
totalNumSolid = np.sum(imageShaped)

# Calculating the total solid temp
temps = np.multiply(imageShaped, values)
totalTempParallel = np.sum(temps)
avgTempParallel = totalTempParallel / totalNumSolid

min = 360
maxRatio = 1
count = 0
fileName = ''
for file in os.listdir(home):
    if file.endswith('.txt'):
        # Loading in the temperature data 
        values = np.loadtxt(home + file)
        values = values.reshape((74,66))
        values = values[4:68, 1:65]

        # Loading in the density field data 
        name = os.path.splitext(file)[0]
        try:
            number = name.split('_')[2]
        except:
            number = name.split('_')[1]

        imageName  = filtered_images + number + '.jpg'
        image = skimage.io.imread(imageName)
        imageShaped = image[4:68, 1:65]
        imageShaped = imageShaped / 255
        imageShaped = np.array(imageShaped)
        imageShaped = np.rint(imageShaped)
        totalNumSolid = np.sum(imageShaped)

        # Calculating the total solid temp
        temps = np.multiply(imageShaped, values)
        totalTemp = np.sum(temps)
        avgTemp = totalTemp / totalNumSolid
        

        # Performance metric 
        if avgTemp < avgTempParallel:
            ratio = totalTemp / totalTempParallel
            print("File: ", file, ", Average temp=", round(avgTemp, 3), ", Energy ratio=", round(ratio, 3))
            count = count + 1
            if ratio > maxRatio:
                maxRatio = ratio
                fileName = file

print('Max ratio =', maxRatio, 'File =', fileName, ', Count =', count)
