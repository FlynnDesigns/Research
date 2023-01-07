import numpy as np 
import os
import skimage
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sci

# Modes GAN, parallel, CNN
mode = 'GAN'

# Directory of the files you want to parse 
home = 'A:\\Research\\Training data\\CNN\\good\\run_1_1_1_23\\final_outputs\\'

# Directory of the filtered images
filtered_images = 'A:\\Research\\Training data\\CNN\\good\\run_1_1_1_23\\filtered_images\\'

# Directory of the paralel fin
parallel = 'A:\\Research\\Research\\get_results\\parallel\\'

## Directory to save good images and matching temperatures (no longer needed)
# images = 'A:\\Research\\Training data\\good\\run_1_1_1_23\\cnn_data\\images\\'
# temperatures = 'A:\\Research\\Training data\\good\\run_1_1_1_23\\cnn_data\\temperatures\\'

# Directory to write and save GAN images 
images_GAN = 'A:\\Research\\Research\\get_results\\images_GAN\\'

# Directory to write the mat output files and stats 
training_dir = 'A:\\Research\\Research\\get_results\\training_data\\train\\'
testing_dir = 'A:\\Research\\Research\\get_results\\training_data\\test\\'
extra_dir = 'A:\\Research\\Research\\get_results\\training_data\\extra_data\\'

# Loading in the temperature data for benchmark parallel fin case 
values = np.loadtxt(parallel + 'T_parallel.txt')
values = values.reshape((74,66))
values = values[5:69, 1:65]

# Parallel fin structure hard coded
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

# Temp variables and initial values 
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
            
            # Rotating the images so that they pop up correctly 
            values = np.rot90(values, 2)
            imageShaped = np.rot90(imageShaped, 1)
            imageShaped = np.array(imageShaped, dtype=np.uint8)

            ## Saving the best designs and their temperatures and images (no longer needed)
            # if mode == 'CNN':
            #     tempDF = pd.DataFrame(values)
            #     tempDF.to_csv(temperatures + str(count) + '.csv')
            #     skimage.io.imsave(images + str(count) + '.jpg', imageShaped, check_contrast=False)

            # Plotting to make sure that both plots are correct 
            if count == 0 and mode == 'CNN':
                plt.rcParams["figure.autolayout"] = True 
                plt.subplot(1,2,1)
                plt.imshow(imageShaped)
                plt.subplot(1,2,2)
                plt.imshow(values)
                plt.show()

            # Saving the mat files / training 
            u = values
            F = imageShaped

            # Saving the u and F arrays into .mat files 
            mdict = {"u": u, "F": F}
            matFileName = str(count) + '.mat'

            # Creating 10,240 training data sets
            if mode == 'CNN':
                if count < 1000000:
                    sci.savemat(training_dir + '\\train\\' + matFileName, mdict)
                    with open(training_dir + 'train_val.txt', 'a') as file:
                        file.write(matFileName + '\n')
                
                # Creating 4,096 test data sets
                elif count >= 10241 and count < 14337:
                    sci.savemat(testing_dir + '\\test\\' + matFileName, mdict)
                    with open(testing_dir + 'test_val.txt', 'a') as file:
                        file.write(matFileName + '\n')
                
                # Saving the remaining to training 
                else:
                    break
                    sci.savemat(extra_dir+ '//test//' + matFileName, mdict)
                    with open(extra_dir + 'train_val.txt', 'a') as file:
                        file.write(matFileName + '\n')

                # Storing the design with the best performance
                if avgTemp < minTemp:
                    minTemp = avgTemp
                    maxTemp = np.max(values)
                    fileName = file 
                    tempField = values
            
            if mode == 'GAN':
                imageShaped[imageShaped == 0] = 255
                imageShaped[imageShaped == 1] = 0
                skimage.io.imsave(images_GAN + str(count) + '.jpg', imageShaped, check_contrast=False)
                

            print(count)

if mode == 'CNN':
    # Showing images 
    plt.imshow(np.flip(tempField))
    plt.title("Avg temp =" + str(round(minTemp,3)) + ", Max temp =" + str(round(maxTemp,3)))
    plt.show()

    # Showing stats 
    print('Parallel fin Min avg temp =', avgTempParallel, '\n')
    print('Best design: ', str(fileName), ', Min avg temp = ', minTemp, '\n')
    print('Total number of designs better than parallel fin:', count, '\\', total, '\n')
    print('Failed designs: ', 80000 - total, ' or ', 100 - round(total/80000 * 100, 2) , '%')
