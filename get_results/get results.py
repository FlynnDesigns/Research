import numpy as np 
import os
import skimage
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sci

# Directory of the files you want to parse 
home = 'A:\\Research\\Training data\\good\\run_1_1_1_23\\final_outputs\\'

# Directory of the filtered images
filtered_images = 'A:\\Research\\Training data\\good\\run_1_1_1_23\\filtered_images\\'

# Directory of the paralel fin
parallel = 'A:\\Research\\Research\\get_results\\parallel\\'

# Directory to save good images and matching temperatures 
images = 'A:\\Research\\Training data\\good\\run_1_1_1_23\\cnn_data\\images\\'
temperatures = 'A:\\Research\\Training data\\good\\run_1_1_1_23\\cnn_data\\temperatures\\'

# Loading in the temperature data for benchmark parallel fin case 
values = np.loadtxt(parallel + 'T_parallel.txt')
values = values.reshape((74,66))
values = values[5:69, 1:65]

# Loading in the density field data 
array = np.zeros((64, 64))
array[:, 0:2] = 1
array[:, 5:7] = 1
array[:, 9:11] = 1
array[:, 13:15] = 1
array[:, 17:19] = 1
array[:, 21:23] = 1
array[:, 25:27] = 1
array[:, 29:31] = 1
array[:, 33:35] = 1
array[:, 37:39] = 1
array[:, 41:43] = 1
array[:, 45:47] = 1
array[:, 49:51] = 1
array[:, 53:55] = 1
array[:, 57:59] = 1
array[:, 62:64] = 1

# Converting list to array 
imageShaped = np.array(array)
totalNumSolid = np.sum(imageShaped)

# Calculating the total solid temp
temps = np.multiply(imageShaped, values)
totalTempParallel = np.sum(temps)
avgTempParallel = totalTempParallel / totalNumSolid

# Temp variables 
count = 0
total = 0 
fileName = ''
minTemp = 400
maxTemp = 0
newCount = 0
sum1Solid = np.zeros((64, 64))
sum2Solid = np.zeros((64, 64))
sum1Temp = np.zeros((64, 64))
sum2Temp = np.zeros((64, 64))

for file in os.listdir(home):
    if file.endswith('.txt'):
        # Loading in the temperature data 
        values = np.loadtxt(home + file)
        values = values.reshape((74,66))
        values = values[5:69, 1:65]

        # Loading in the density field data 
        name = os.path.splitext(file)[0]
        try:
            number = name.split('_')[2]
        except:
            number = name.split('_')[1]

        # Reading in the image and computing the performance
        imageName  = filtered_images + number + '.jpg'
        try:
            image = skimage.io.imread(imageName)
            imageShaped = image
            imageShaped = imageShaped / 255
            imageShaped = np.array(imageShaped)
            imageShaped = np.rint(imageShaped)
            totalNumSolid = np.sum(imageShaped)
        except: 
            # Break out of the loop if there are no matches 
            break

        # Calculating the total solid temp
        temps = np.multiply(imageShaped, values)
        totalTemp = np.sum(temps)
        avgTemp = totalTemp / totalNumSolid
        
        # Incrementing the total number of files 
        total = total + 1

        # Performance metric comparing parallel vs current design
        if avgTemp < avgTempParallel:
            count = count + 1

            # Saving the best designs and their temperatures and images 
            tempDF = pd.DataFrame(values)
            tempDF.to_csv(temperatures + str(count) + '.csv')
            imageShaped = np.array(imageShaped, dtype=np.uint8)
            skimage.io.imsave(images + str(count) + '.jpg', imageShaped, check_contrast=False)

            # Summing solid for later use in calculating mean and std
            sum1Solid = imageShaped + sum1Solid
            sum2Solid = sum2Solid + sum1Solid**2

            # Summing temp for later use in calculating mean and std
            sum1Temp = sum1Temp + values
            sum2Temp = sum2Temp + values**2

            # Storing the design with the best performance
            if avgTemp < minTemp:
                minTemp = avgTemp
                maxTemp = np.max(values)
                fileName = file 
                tempField = values

# Calculating mean and std
solidMean = sum1Solid / count 
solidSTD = np.sqrt(sum2Solid / count - solidMean**2)

tempMean = sum1Temp / count 
tempSTD = np.sqrt(sum2Temp / count - tempMean**2)

# Program outputs 
# Showing images 
plt.imshow(np.flip(tempField))
plt.title("Avg temp =" + str(round(avgTemp,3)) + ", Max temp =" + str(round(maxTemp,3)))
plt.show()

# Showing solid std and mean
print("Solid Mean:", solidMean, ', STD:', solidSTD)

# Showing temp std and mean 
print("Temp Mean:", tempMean, ', STD:', tempSTD)

# Showing stats 
print('Parallel fin Min avg temp =', avgTempParallel, '\n')
print('Best design: ', fileName, ', Min avg temp = ', minTemp, '\n')
print('Total number of designs better than parallel fin:', count, '\\', total)
