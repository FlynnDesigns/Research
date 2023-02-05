import numpy as np 
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sci
import multiprocessing as mp 
import skimage

def sig(x, k):
    return 1 / (1 + np.exp(-k * (x - 0.5 + 0.157 - (0.3529411764705882 - 0.34))))

def norm(x):
    norm = (x - x.min()) * 2.0
    denorm = x.max() - x.min()
    return norm/denorm - 1.0

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
    im1 = plt.imshow(designField, aspect='auto')
    
    # Subplot 2 settings
    ax2 = plt.subplot(1,2,2)
    ax2.set_title('Temperature Field')
    im2 = plt.imshow(tempField, aspect='equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax, orientation='vertical')
    
    # Main plot settings
    volfrac = np.sum(designField) / (64 * 64)
    plt.suptitle(name + '\n\n' + "Vol frac = " + str(round(volfrac, 3)) + ", Avg temp = " + str(round(avgTemp,3)) + ", Max temp = " + str(round(maxTemp,3)))
    plt.savefig(dir + name + '.jpg', dpi = 300)
    plt.close()

def getResults(offset, process, numberOfSimsPerProcess, home, prefix, run, graph):
    # Peformance metric to beat 
    avgTempParallel = 313.608

    # Temp variables and initial values 
    count = 0
    total = 0 
    minTemp = 400
    vol_frac_list = []
    
    for fileNum in range(offset, offset + numberOfSimsPerProcess):
        file = prefix + "T_" + str(fileNum) + ".txt"
        try:
            # Loading in the temperature data
            tempField = loadTemp(home + "temperatures\\" + file)
        except:
            continue

        # Creating the full design domain 
        design = np.zeros((74,66))
        fullFileName = home + "coordinates\\" + prefix + str(fileNum) + ".txt"

        # Loading in the coordinates 
        coordinates = np.loadtxt(fullFileName, delimiter=" ")
        coordinates = coordinates - 0.5
        coordinates = np.array(coordinates, dtype=int)

        # Loading in the image 
        image = skimage.io.imread(home + "sim_images\\images\\" + str(fileNum) + ".jpg")
        image = np.array(image)
        image = image[:, :, 0]

        # Rotating the image 
        # if prefix == "0":
        #     rot = 0
        # elif prefix == "90":
        #     rot = 1
        # elif prefix == "180":
        #     rot = 2
        # elif prefix == "270":
        #     rot = 3
        # image = np.rot90(image, rot)

        # Normalizing and applying filter 
        # img_normal = np.copy(image)
        # for j in range(64):
        #     for i in range(64):
        #         if img_normal[i,j] >= 90:
        #             img_normal[i,j] = 0
        #         else:
        #             img_normal[i,j] = 1

        # img_normal = 1 - img_normal
        # img_normal = abs(img_normal)
        # image = img_normal
        #image = norm(image)
        image = image / 255
        # image = 1 - image 
        # image = abs(image)
        # image = sig(image, 20)
        # image = 1 - image
        # image = np.abs(image) 

        designField = design[5:69, 1:65]

        # Making sure that the temperature field is oriented in the correct way 
        # maxTempTest = 0
        # orientationCorrected = 1
        # for i in range(4):
        #     designFieldTest = np.rot90(designField, i)
            
        #     # Calculating the stats of the design
        #     avgTemp = getAvgTemp(designFieldTest, tempField) 

        #     if avgTemp > maxTempTest:
        #         maxTempTest = avgTemp
        #         orientationCorrected = i

        # designField = np.rot90(designField, orientationCorrected)
            
        # Calculating the stats of the design
        avgTemp = getAvgTemp(designField, tempField)
        vol_frac = np.sum(designField) / (64 * 64)

        # Writing the stats to a text file 
        with open(home + 'stats\\' + prefix + "_" + str(process) + '.txt', 'a') as f:
            newName =str(run) + "_" + prefix + "_" + str(fileNum)
            f.write(newName + ", " + str(avgTemp) + "\n")

        # Incrementing the total number of files (used later)
        total = total + 1

        # Creating and writing a mat file for each design
        tempField = np.array(tempField, dtype=np.float32)
        tempFieldMat = np.copy(tempField)
        tempFieldMat = np.rot90(tempFieldMat, -2)

        image = np.array(image, dtype=np.float32)
        imageMat = np.copy(image)
        imageMat = np.rot90(imageMat, -1)

        mdict = {"u": tempFieldMat, "F": imageMat}
        matFileName = str(run) + "_" + prefix + "_" + str(fileNum) + '.mat'
        sci.savemat(home + 'mat_files\\train\\train\\' + matFileName, mdict)
        # plot_design(designField, tempField, str(run) + "_" + prefix + "_" + str(fileNum), home + "best_best\\")
        
        image = np.transpose()
        # Tracking the best designs
        if avgTemp < minTemp:
            minTemp = avgTemp # Recording the smallest temp 
            design = designField # Recordint the best design 

        # Tracking the number of data points better than parallel fin 
        if avgTemp < avgTempParallel:
            vol_frac_list.append(vol_frac)
            if graph == True:
                plot_design(designField, tempField, str(run) + "_" + prefix + "_" + str(fileNum), home + "best_best\\")
    
    # Writing the volumetric data to a file 
    # volFracs = np.array(vol_frac_list)
    # minFracs = volFracs.min()
    # maxFracs = volFracs.max()
    # avgFracs = volFracs.mean()
    # with open(home + 'vol_frac_info\\general_stats.txt', 'a') as  file:
    #     file.write(str(round(minFracs, 3)) + ", " + str(round(maxFracs, 3)) + ", " + str(round(avgFracs, 3)) + "\n")

    with open(home + 'stats\\' + prefix + "_" + str(process) + '.txt', 'a') as f:
            newName =str(run) + "_" + prefix + "_" + str(fileNum)
            f.write(newName + ", " + str(avgTemp) + "\n")

def multiP(totalNumSimulations, home, key="", numProcesses=4, run=0, graph=False):
    processes = numProcesses
    number_of_sims_per_process = totalNumSimulations / processes
    for i in range(processes):
        offset = i * number_of_sims_per_process
        p = mp.Process(target=getResults, args=(int(offset), int(i), int(number_of_sims_per_process), home, key, int(run), graph))
        p.start()
        
if __name__ == "__main__":
    run = 0
    home = 'D:\\GAN\\run_' + str(run) + '\\'
    simulations = 500000
    threads = 20
    graph = True
    clean_best = True
    clean_mat = True
    clean_stats = True

    if clean_best == True:
        # Cleaning best best dir 
        try:
            shutil.rmtree(home + 'best_best\\')
        except:
            print("No best best to remove")
        os.mkdir(home + 'best_best\\')

    if clean_mat == True:
        # Cleaning mat files dir 
        try:
            shutil.rmtree(home + 'mat_files\\')
        except:
            print("No mat files to remove!")
        os.mkdir(home + 'mat_files\\')
        os.mkdir(home + 'mat_files\\train\\')
        os.mkdir(home + 'mat_files\\train\\train\\')

    if clean_stats == True:
        # Cleaning stats file dir 
        try: 
            shutil.rmtree(home + "stats\\")
        except:
            print("No stats file dir to remove")
        os.mkdir(home + "stats\\")

    multiP(simulations, home, "", threads, run, graph)

    # # Running for 0
    # multiP(simulations, home, "0", threads, run, graph)

    # # Running for 90
    # multiP(simulations, home, "90", threads, run, graph)

    # # Running for 180
    # multiP(simulations, home, "180", threads, run, graph)

    # # Running for 270
    # multiP(simulations, home, "270", threads, run, graph)
