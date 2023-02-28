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

        # Reading in each image, image by image
        image = skimage.io.imread(home + "images\\images\\" + str(fileNumber) + ".jpg")
        original = asarray(image) 
        current = np.copy(original)

        # Resizing the image to 64 x 64 if it is not the correct size 
        current = cv2.resize(current, dsize=(64,64),  interpolation=cv2.INTER_CUBIC)
        size = current.shape
        try:
            current = current[:,:,0]
        except:
            pass
        current = np.array(current, dtype=np.uint8)

        # Adding fluid borders to image. If this is the correct size, it doesnt matter
        if size[0] == 64 or size[1] == 64:
            output = np.ones((74,66), dtype=np.uint8) * 255
            output[5:69, 1:65] = current
            current = np.copy(output) 

        # Before inverting 
        for j in range(66):
            for i in range(74):
                if current[i,j] < 90:
                    current[i,j] = 0
                else:
                    current[i,j] = 1
        
        # Rotating the bounds of the image 
        current0 = current[5:69, 1:65]

        # Splitting up the name of the file so the correct number can be saved
        outputName = str(fileNumber)

        # Rotating current coordinates so that they show up correctly 
        current[5:69, 1:65] = np.rot90(current[5:69, 1:65], 3)

        # Transposing everything to get it centered correctly 
        current0 = np.transpose(current)

        # Writing the solid coordinates file for 0, 90, 180, 270
        createCoordinates(current0, save_dir + outputName + '.txt')

def multiP_img_2_txt(totalNumImages, home):
    processes = 20
    number_of_images = int(totalNumImages) / processes
    for i in range(processes):
        offset = i * number_of_images
        p = mp.Process(target=convertImagesToTxt, args=(int(number_of_images), int(offset), home))
        p.start()
    

def multiP_tar(home):
    processes = 15
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
    
    for i in range(processes):
        output_filename = O_dir_list[i]
        source_dir = s_dir_list[i]
        p = mp.Process(target=make_tarfile, args=(output_filename, source_dir))
        p.start()

if __name__ == "__main__":
    # Home directory settings: 
    home = 'C:\\Users\\Nate\\run_combined\\'
    mode = 'create'
    ############################################################################
    if mode == 'credate':
        # Cleaning / working on coordinates folder 
        print("Removing coordinates dir")
        try:
            shutil.rmtree(home + "coordinates\\")
        except:
            print("Not coordinates folder to remove")
        print("Making coordinates dir")
        os.mkdir(home + "coordinates\\")
        
        # Making set 0 dirs
        os.mkdir(home + "coordinates\\set_0\\")
        os.mkdir(home + "coordinates\\set_1\\")
        os.mkdir(home + "coordinates\\set_2\\")
        os.mkdir(home + "coordinates\\set_3\\")
        os.mkdir(home + "coordinates\\set_4\\")

        # Making set 1 dirs 
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
    else:
        # Cleaning / working on raw_gz_files folder 
        try:
            shutil.rmtree(home + '\\raw_gz_files\\')
        except:
            print("No raw gz files folder to remove")
        os.mkdir(home + "raw_gz_files\\")

        # Compressing all of the files in parallel 
        multiP_tar(home)