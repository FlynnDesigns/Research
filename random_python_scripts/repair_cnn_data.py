from scipy.io import loadmat
from scipy.io import savemat
import os
import numpy as np 

# Home directory! 
home_dir = 'A:\\Research\\Research\\Machine_Learning\\U_NET\\samples\\data\\mat_files_270Gan_50True\\test\\test\\'

for file in os.listdir(home_dir):
    if '__' in file:
        # Loading files 
        matFile = loadmat(home_dir + file)
        design = np.array(matFile['F'])
        temp = np.array(matFile['u'])

        # Correcting files
        design = np.rot90(design, 2)
        temp = np.rot90(temp , 2)

        # Creating new matfile 
        mdict = {"u": temp, "F": design}

        # Saving mat file 
        savemat(home_dir + file, mdict)
    else:
        # Loading files 
        matFile = loadmat(home_dir + file)
        design = np.array(matFile['F'])
        temp = np.array(matFile['u'])

        # Correcting files
        design = np.rot90(design, -1)

        # Creating new matfile 
        mdict = {"u": temp, "F": design}

        # Saving mat file 
        savemat(home_dir + file, mdict)