# Custom libraries 
import skimage
from PIL import Image as im
import matplotlib.pyplot as plt 
from numpy import asarray
import numpy as np
import cv2
import pandas as pd
import tarfile
import multiprocessing as mp
import PIL
import math

# Default libraries
import os
import os.path
import glob
import shutil

# Function used to create coordinates 
def createCoordinates(designLayout, outputName):
    with open(outputName, 'w') as f: 
        for k in range(66):
            for i in range(74):
                # Applying offset to get the coordinates to the right positon 
                x_val = k + 0.5 
                y_val = i + 0.5 

                # If the coordinates are the middle of the physical domain, write the coordinates
                if designLayout[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                    f.write("%s " % (x_val))
                    f.write("%s\n" % (y_val))

# Create tarfile of coordinates in the end 
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir)) 

# Converting raw_images to coordinates 
def convertImagesToTxt(number_of_images, offset, home):
    # Converting all of assigned images per process into text 
    for k in range(number_of_images):
        fileNumber = k + offset

        # Reading in each design
        design = np.loadtxt(f"{home}raw_images\\{fileNumber}.txt")
        design = design * 255

        """
        Adds fluid borders to the image 74 x 66. 
        In the future this should be removed so that the images are only 64 x 64 to save space.
        """
        full_domain = np.ones((74,66), dtype=np.uint8) * 255
        full_domain[5:69, 1:65] = design

        # Filtering the design 
        for j in range(66):
            for i in range(74):
                if full_domain[i,j] < 90:
                    full_domain[i,j] = 0
                else:
                    full_domain[i,j] = 1

        # Rotating current coordinates so that they show up correctly 
        full_domain[5:69, 1:65] = np.rot90(full_domain[5:69, 1:65], 3)

        # Transposing everything to get it centered correctly 
        full_domain = np.transpose(full_domain)

        # Creating physical coordinates
        createCoordinates(full_domain, f"{home}coordinates\\{fileNumber}.txt")

# Function to run conversion with multiple processes 
def multiP_img_2_txt(totalNumImages, home):
    processes = 20
    number_of_images = int(totalNumImages) / processes
    for i in range(processes):
        offset = i * number_of_images
        p = mp.Process(target=convertImagesToTxt, args=(int(number_of_images), int(offset), home))
        p.start()
    
if __name__ == "__main__":
    # Directory the code will run out of 
    home = 'C:\\Users\\Nate\\Documents\\'

    # Cleaning / working on coordinates folder 
    try:
        shutil.rmtree(home + "coordinates\\")
        print("Removing coordinates dir")
    except:
        print("Not coordinates folder to remove")
    print("Making coordinates dir")
    os.mkdir(home + "coordinates\\")

    # Cleaning / working on raw_gz_files folder 
    try:
        shutil.rmtree(home + '\\raw_gz_files\\')
        print("Removing raw_gz_files dir")
    except:
        print("No raw gz files folder to remove")
    print("Making raw_gz_files dir")
    os.mkdir(home + "raw_gz_files\\")
    
    # Running conversion of images to coordinates in parallel 
    print("Running image to text conversion")
    multiP_img_2_txt(750000, home)