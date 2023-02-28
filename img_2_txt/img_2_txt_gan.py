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

# Default libraries
import os
import os.path
import glob
import shutil

def saveImage(name, array):
    plt.imshow(array)
    plt.colorbar()
    plt.savefig(name, dpi=300)
    plt.close()

def sig(x, k):
    x = 1 / (1 + np.exp(-x / k))
    return x

def norm(x):
    norm = (x - x.min()) * 2.0
    denorm = x.max() - x.min()
    return norm/denorm - 1.0

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
    # Converting raw_images to coordinates 
    for k in range(number_of_images):

        # Assigning output location 
        fileNumber = k + offset
        if fileNumber >= 0 and fileNumber <= 49999:
            save_dir = home + 'coordinates\\set_0\\'
        elif fileNumber >= 50000 and fileNumber <= 99999:
            save_dir = home + 'coordinates\\set_1\\'
        elif fileNumber >= 100000 and fileNumber <= 149999:
            save_dir = home + 'coordinates\\set_2\\'
        elif fileNumber >= 150000 and fileNumber <= 199999:
            save_dir = home + 'coordinates\\set_3\\'
        elif fileNumber >= 200000 and fileNumber <= 249999:
            save_dir = home + 'coordinates\\set_4\\'
        elif fileNumber >= 250000 and fileNumber <= 299999:
            save_dir = home + 'coordinates\\set_5\\'
        elif fileNumber >= 300000 and fileNumber <= 349999:
            save_dir = home + 'coordinates\\set_6\\' 
        elif fileNumber >= 350000 and fileNumber <= 399999:
            save_dir = home + 'coordinates\\set_7\\' 
        elif fileNumber >= 400000 and fileNumber <= 449999:
            save_dir = home + 'coordinates\\set_8\\' 
        elif fileNumber >= 450000 and fileNumber <= 499999:
            save_dir = home + 'coordinates\\set_9\\' 
        elif fileNumber >= 500000 and fileNumber <= 549999:
            save_dir = home + 'coordinates\\set_10\\' 
        elif fileNumber >= 550000 and fileNumber <= 599999:
            save_dir = home + 'coordinates\\set_11\\' 
        elif fileNumber >= 600000 and fileNumber <= 649999:
            save_dir = home + 'coordinates\\set_12\\' 
        elif fileNumber >= 650000 and fileNumber <= 699999:
            save_dir = home + 'coordinates\\set_13\\' 
        elif fileNumber >= 700000 and fileNumber <= 749999:
            save_dir = home + 'coordinates\\set_14\\' 

        # Reading in each design
        design = np.loadtxt(f"{home}images\\images\\{fileNumber}.jpg")
        design = design * 255

        # Adding fluid borders to the design 
        full_domain = np.ones((74,66), dtype=np.uint8) * 255
        full_domain[5:69, 1:65] = design

        # Before inverting 
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

        # Writing the solid coordinates file for 0, 90, 180, 270
        createCoordinates(full_domain, f"{save_dir}{fileNumber}.txt")

    """
    Zipping the file below  
    """
    # Output dirs: 
    O_dir_list = []
    # 0 - 249,999
    O_dir_list.append(home + "raw_gz_files\\0.gz")
    O_dir_list.append(home + "raw_gz_files\\50.gz")
    O_dir_list.append(home + "raw_gz_files\\100.gz")
    O_dir_list.append(home + "raw_gz_files\\150.gz")
    O_dir_list.append(home + "raw_gz_files\\200.gz")
    # 250,000 - 499,999
    O_dir_list.append(home + "raw_gz_files\\250.gz")
    O_dir_list.append(home + "raw_gz_files\\300.gz")
    O_dir_list.append(home + "raw_gz_files\\350.gz")
    O_dir_list.append(home + "raw_gz_files\\400.gz")
    O_dir_list.append(home + "raw_gz_files\\450.gz")
    # 500,000 - 749,999
    O_dir_list.append(home + "raw_gz_files\\500.gz")
    O_dir_list.append(home + "raw_gz_files\\550.gz")
    O_dir_list.append(home + "raw_gz_files\\600.gz")
    O_dir_list.append(home + "raw_gz_files\\650.gz")
    O_dir_list.append(home + "raw_gz_files\\700.gz")

    # Source dirs
    s_dir_list = []
    # 0 - 249,999
    s_dir_list.append(home + "coordinates\\set_0\\")
    s_dir_list.append(home + "coordinates\\set_1\\")
    s_dir_list.append(home + "coordinates\\set_2\\")
    s_dir_list.append(home + "coordinates\\set_3\\")
    s_dir_list.append(home + "coordinates\\set_4\\")
    # 250,000 - 499,999
    s_dir_list.append(home + "coordinates\\set_5\\")
    s_dir_list.append(home + "coordinates\\set_6\\")
    s_dir_list.append(home + "coordinates\\set_7\\")
    s_dir_list.append(home + "coordinates\\set_8\\")
    s_dir_list.append(home + "coordinates\\set_9\\")
    # 500,000 - 749,999
    s_dir_list.append(home + "coordinates\\set_10\\")
    s_dir_list.append(home + "coordinates\\set_11\\")
    s_dir_list.append(home + "coordinates\\set_12\\")
    s_dir_list.append(home + "coordinates\\set_13\\")
    s_dir_list.append(home + "coordinates\\set_14\\")

    # Zipping the files
    process_id = int(offset / number_of_images)
    if process_id == 1:
        print("Zipping files")
    if process_id < 15:
        make_tarfile(O_dir_list[process_id, s_dir_list[process_id]])

# Function to run conversion with multiple processes 
def multiP_img_2_txt(totalNumImages, home):
    processes = 20
    number_of_images = int(totalNumImages) / processes
    for i in range(processes):
        offset = i * number_of_images
        p = mp.Process(target=convertImagesToTxt, args=(int(number_of_images), int(offset), home))
        p.start()
    
if __name__ == "__main__":
    # Home directory settings: 
    home = 'D:\\GAN\\run_combined\\'

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
    
    # Making set directories for 750,000 designs (50,000 per set)
    os.mkdir(home + "coordinates\\set_0\\")
    os.mkdir(home + "coordinates\\set_1\\")
    os.mkdir(home + "coordinates\\set_2\\")
    os.mkdir(home + "coordinates\\set_3\\")
    os.mkdir(home + "coordinates\\set_4\\")
    os.mkdir(home + "coordinates\\set_5\\")
    os.mkdir(home + "coordinates\\set_6\\")
    os.mkdir(home + "coordinates\\set_7\\")
    os.mkdir(home + "coordinates\\set_8\\")
    os.mkdir(home + "coordinates\\set_9\\")
    os.mkdir(home + "coordinates\\set_10\\")
    os.mkdir(home + "coordinates\\set_11\\")
    os.mkdir(home + "coordinates\\set_12\\")
    os.mkdir(home + "coordinates\\set_13\\")
    os.mkdir(home + "coordinates\\set_14\\")

    # # Running conversion of images to coordinates in parallel 
    print("Running image to text conversion")
    multiP_img_2_txt(750000, home)