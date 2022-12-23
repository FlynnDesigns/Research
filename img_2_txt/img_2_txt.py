# Custom libraries 
import skimage
from numpy import asarray
import numpy as np
import cv2

# Default libraries
import sys
import os

"""
This script is used to take a 66 x 74 pixel image and convert it over to a numpy array.
Later, the numpy array is then converted to coordinates that OpenFOAM can read and turn into a mesh.
The script takes one input, the images name.
The script runs and saves out of the current working directory 
"""

# Home directory settings: 
home = '/home/nathan/Desktop/Research/img_2_txt/'
coordinates = '/home/nathan/Desktop/Research/img_2_txt/coordinates/'
images = '/home/nathan/Desktop/Research/img_2_txt/images/'
os.chdir(home)

# Opening the image and converting it to an array 
for image_name in os.listdir(images):
    image = skimage.io.imread(images + image_name)
    original = asarray(image) 
    current = np.copy(original)

    # Resizing 
    current = cv2.resize(current, dsize=(64,64),  interpolation=cv2.INTER_CUBIC)
    size = current.shape
    current = current[:,:,0]

    # Adding fluid borders to image. If this is the correct size, it doesnt matter
    if size[0] == 64 or size[1] == 64:
        output = np.ones((74,66)) * 255
        output[5:69,1:65] = current
        current = output 
       
    # Applying filter to convert region to fluid and solid
    for j in range (0, 65):
        for i in range (0, 73):
            if current[i,j] >= 90:
                current[i,j] = 0
            else:
                current[i,j] = 1

    # Converting the array into coordinates
    current = np.rot90(current,3)

    # Splitting up the name of the file so the correct number can be saved
    outputName = image_name.strip('.jpg')

    # Writing the solid coordinates file
    with open(coordinates + outputName + '.txt', 'w') as f: 
            for k in range(0, 65):
                for i in range(0, 73):
                    # Applying offset to get the coordinates to the right positon 
                    x_val = k + 0.5
                    y_val = i + 0.5 

                    # If the coordinates are the middle of the physical domain, write the coordinates
                    if current[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                        f.write("%s " % (x_val))
                        f.write("%s\n" % (y_val))