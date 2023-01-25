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

# Default libraries
import os
import os.path
import glob
import shutil

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

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir)) 

def convertImagesToTxt(number_of_images, offset, home):
    for k in range(number_of_images):
        fileNumber = k + offset

        if fileNumber > 49999:
            save_dir = home + "coordinates\\set_1\\"
        else:
            save_dir = home + "coordinates\\set_0\\"

        # Reading in each image, image by image
        image = skimage.io.imread(home + "images\\images\\" + str(fileNumber) + ".jpg")
        original = asarray(image) 
        current = np.copy(original)

        # Resizing the image to 64 x 64 if it is not the correct size 
        current = cv2.resize(current, dsize=(64,64),  interpolation=cv2.INTER_CUBIC)
        size = current.shape
        current = current[:,:,0]
        current = np.array(current, dtype=np.uint8)

        # Adding fluid borders to image. If this is the correct size, it doesnt matter
        if size[0] == 64 or size[1] == 64:
            output = np.ones((74,66), dtype=np.uint8) * 255
            output[5:69, 1:65] = current
            current = output 
        
        # Filtering the image 
        for j in range(66):
            for i in range(74):
                if current[i,j] >= 90:
                    current[i,j] = 0
                else:
                    current[i,j] = 1

        # Forcing current to be uint8
        current0 = current[5:69, 1:65]
        current90 =  np.rot90(current0, -1)
        current180 = np.rot90(current90, -1)
        current270 = np.rot90(current180, -1)

        # Splitting up the name of the file so the correct number can be saved
        outputName = str(fileNumber)

        # Rotating current coordinates so that they show up correctly 
        current[5:69, 1:65] = np.rot90(current[5:69, 1:65], 3)
        
        current90 = np.copy(current)
        current90[5:69, 1:65] = np.rot90(current90[5:69, 1:65], -1)
        
        current180 = np.copy(current)
        current180[5:69, 1:65] = np.rot90(current180[5:69, 1:65], -2)

        current270 = np.copy(current)
        current270[5:69, 1:65] = np.rot90(current270[5:69, 1:65], -3)

        # Transposing everything to get it centered correctly 
        current0 = np.transpose(current)
        current90 = np.transpose(current90)
        current180 = np.transpose(current180)
        current270 = np.transpose(current270)

        # Writing the solid coordinates file for 0, 90, 180, 270
        createCoordinates(current0, save_dir + "0\\" + outputName + '.txt')
        createCoordinates(current90, save_dir + "90\\" + outputName + '.txt')
        createCoordinates(current180, save_dir + "180\\" + outputName + '.txt')
        createCoordinates(current270, save_dir + "270\\" +  outputName + '.txt')

def multiP_img_2_txt(totalNumImages, home):
    processes = 10
    number_of_images = int(totalNumImages) / processes
    for i in range(processes):
        offset = i * number_of_images
        p = mp.Process(target=convertImagesToTxt, args=(int(number_of_images), int(offset), home))
        p.start()

def multiP_tar(home):
    processes = 8
    O_dir_list = []
    # 0 - 49,999
    O_dir_list.append(home + "raw_gz_files\\0.gz")
    O_dir_list.append(home + "raw_gz_files\\90.gz")
    O_dir_list.append(home + "raw_gz_files\\180.gz")
    O_dir_list.append(home + "raw_gz_files\\270.gz")
    # 50,000 - 99,999
    O_dir_list.append(home + "raw_gz_files\\500.gz")
    O_dir_list.append(home + "raw_gz_files\\5090.gz")
    O_dir_list.append(home + "raw_gz_files\\50180.gz")
    O_dir_list.append(home + "raw_gz_files\\50270.gz")

    s_dir_list = []
    # 0 - 49,999
    s_dir_list.append(home + "coordinates\\set_0\\0\\")
    s_dir_list.append(home + "coordinates\\set_0\\90\\")
    s_dir_list.append(home + "coordinates\\set_0\\180\\")
    s_dir_list.append(home + "coordinates\\set_0\\270\\")
    # 50,000 - 99,999
    s_dir_list.append(home + "coordinates\\set_1\\0\\")
    s_dir_list.append(home + "coordinates\\set_1\\90\\")
    s_dir_list.append(home + "coordinates\\set_1\\180\\")
    s_dir_list.append(home + "coordinates\\set_1\\270\\")
    
    for i in range(processes):
        output_filename = O_dir_list[i]
        source_dir = s_dir_list[i]
        p = mp.Process(target=make_tarfile, args=(output_filename, source_dir))
        p.start()

if __name__ == "__main__":
    # Home directory settings: 
    home = 'A:\\Research\\Training_data\\run_4\\'
    ############################################################################

    # Cleaning / working on coordinates folder 
    # try:
    #     shutil.rmtree(home + "coordinates\\")
    # except:
    #     print("Not coordinates folder to remove")
    # os.mkdir(home + "coordinates\\")
    
    # Cleaning / working on raw_gz_files folder 
    try:
        shutil.rmtree(home + '\\raw_gz_files\\')
    except:
        print("No raw gz files folder to remove")
    os.mkdir(home + "raw_gz_files\\")
    
    # # Making set 0 dirs
    # os.mkdir(home + "coordinates\\set_0\\")
    # os.mkdir(home + "coordinates\\set_0\\0\\")
    # os.mkdir(home + "coordinates\\set_0\\90\\")
    # os.mkdir(home + "coordinates\\set_0\\180\\")
    # os.mkdir(home + "coordinates\\set_0\\270\\")

    # # Making set 1 dirs 
    # os.mkdir(home + "coordinates\\set_1\\")
    # os.mkdir(home + "coordinates\\set_1\\0\\")
    # os.mkdir(home + "coordinates\\set_1\\90\\")
    # os.mkdir(home + "coordinates\\set_1\\180\\")
    # os.mkdir(home + "coordinates\\set_1\\270\\")

    # # Running conversion of images to coordinates in parallel 
    # multiP_img_2_txt(100000, home)

    # Compressing all of the files in parallel 
    multiP_tar(home)