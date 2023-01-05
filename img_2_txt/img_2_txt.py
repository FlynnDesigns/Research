# Custom libraries 
import skimage
import matplotlib.pyplot as plt 
from numpy import asarray
import numpy as np
import cv2

# Default libraries
import os
import glob

"""
This script is used to take a 66 x 74 pixel image and convert it over to a numpy array.
Later, the numpy array is then converted to coordinates that OpenFOAM can read and turn into a mesh.
The script takes one input, the images name.
The script runs and saves out of the current working directory 
"""

# Home directory settings: 
home = 'A:\\Research\Research\\img_2_txt\\'
coordinates = home + '0\\'
coordinates90 = home + '90\\'
coordinates180 = home + '180\\'
coordinates270 = home + '270\\'
images = home + 'images\\'
filteredImages = home + '\\filtered_images\\'

# Changing to home directory
os.chdir(home)

# Using base design setting for testing 
useBaseDesign = False

# Running settings 
run = input("Would you like to run?\n This will clear all previous runs\n enter y to run:")
if run == 'y':
    print('Clearing old files')
    files = glob.glob(coordinates + '*')
    for f in files:
        os.remove(f)
    files = glob.glob(coordinates90 + '*')
    for f in files:
        os.remove(f)
    files = glob.glob(coordinates180 + '*')
    for f in files:
        os.remove(f)
    files = glob.glob(coordinates270 + '*')
    for f in files:
        os.remove(f)

os.system('cls')
print('Converting images to coordinates')
# Opening the image and converting it to an array 
for image_name in os.listdir(images):
    # Reading in each image, image by image
    image = skimage.io.imread(images + image_name)
    original = asarray(image) 
    current = np.copy(original)

    # Resizing the image to 64 x 64 if it is not the correct size 
    current = cv2.resize(current, dsize=(64,64),  interpolation=cv2.INTER_CUBIC)
    size = current.shape
    current = current[:,:,0]
    current = np.array(current, dtype=np.uint8)

    if useBaseDesign == True:
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
        current = 256 * array

    # Adding fluid borders to image. If this is the correct size, it doesnt matter
    if size[0] == 64 or size[1] == 64:
        output = np.ones((74,66), dtype=np.uint8) * 255
        output[5:69, 1:65] = current
        current = output 

    # Applying filter to convert region to fluid and solid
    for j in range(66):
        for i in range(74):
            if current[i,j] >= 90:
                current[i,j] = 0
            else:
                current[i,j] = 1
                
    # Saving the image for 0 degrees 
    skimage.io.imsave(filteredImages + image_name, 255 * current[5:69, 1:65])

    # Saving the image for 90 degrees
    skimage.io.imsave(filteredImages + '90_' + image_name, 255 * np.rot90(current[5:69, 1:65], -1))

    # Saving the image for 180 degrees
    skimage.io.imsave(filteredImages + '180_' + image_name, 255 * np.rot90(current[5:69, 1:65], -2))

    # Saving the image for 270 degrees
    skimage.io.imsave(filteredImages + '270_' + image_name, 255 * np.rot90(current[5:69, 1:65], -3))

    # Splitting up the name of the file so the correct number can be saved
    outputName = image_name.strip('.jpg')

    # Rotating current coordinates so that they show up correctly 
    current[5:69, 1:65] = np.rot90(current[5:69, 1:65], 3)
    
    current90 = np.copy(current)
    current90[5:69, 1:65] = np.rot90(current90[5:69, 1:65], -1)
    
    current180 = np.copy(current)
    current180[5:69, 1:65] = np.rot90(current180[5:69, 1:65], -2)

    current270 = np.copy(current)
    current270[5:69, 1:65] = np.rot90(current270[5:69, 1:65], -3)

    # Transposing everything to get it centered correctly 
    current = np.transpose(current)
    current90 = np.transpose(current90)
    current180 = np.transpose(current180)
    current270 = np.transpose(current270)

    # Writing the solid coordinates file for 0
    with open(coordinates + outputName + '.txt', 'w') as f: 
            for k in range(66):
                for i in range(74):
                    # Applying offset to get the coordinates to the right positon 
                    x_val = k + 0.5 
                    y_val = i + 0.5 

                    # If the coordinates are the middle of the physical domain, write the coordinates
                    if current[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                        f.write("%s " % (x_val))
                        f.write("%s\n" % (y_val))
    
    # Writing the solid coordinates file for 90
    with open(coordinates90 + outputName + '.txt', 'w') as f: 
            for k in range(66):
                for i in range(74):
                    # Applying offset to get the coordinates to the right positon 
                    x_val = k + 0.5 
                    y_val = i + 0.5 

                    # If the coordinates are the middle of the physical domain, write the coordinates
                    if current[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                        f.write("%s " % (x_val))
                        f.write("%s\n" % (y_val))
    
    # Writing the solid coordinates file for 180
    with open(coordinates180 + outputName + '.txt', 'w') as f: 
            for k in range(66):
                for i in range(74):
                    # Applying offset to get the coordinates to the right positon 
                    x_val = k + 0.5 
                    y_val = i + 0.5 

                    # If the coordinates are the middle of the physical domain, write the coordinates
                    if current[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                        f.write("%s " % (x_val))
                        f.write("%s\n" % (y_val))

     # Writing the solid coordinates file for 270
    with open(coordinates270 + outputName + '.txt', 'w') as f: 
            for k in range(66):
                for i in range(74):
                    # Applying offset to get the coordinates to the right positon 
                    x_val = k + 0.5 
                    y_val = i + 0.5 

                    # If the coordinates are the middle of the physical domain, write the coordinates
                    if current[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                        f.write("%s " % (x_val))
                        f.write("%s\n" % (y_val))

# Zipping all of the files using tar and placing them into the coordinates folder
import tarfile
import os.path

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir)) 
os.system('cls')
# Zipping up the 0 degree coordinates 
print('Zipping 0')
make_tarfile(home + 'coordinates0.gz', home + '0//')

# Zipping up the 90 degree coordinates
print('Zipping 90')
make_tarfile(home + 'coordinates90.gz', home + '90//')

# Zipping up the 180 degree coordinates 
print('Zipping 180')
make_tarfile(home + 'coordinates180.gz', home + '180//')

# Zipping up the 270 degree coordinates
print('Zipping 270')
make_tarfile(home + 'coordinates270.gz', home + '270//')
