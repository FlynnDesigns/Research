# Custom libraries 
import skimage
from numpy import asarray
import numpy as np
# Default libraries
import sys
import os

"""
This script is used to take a 66 x 74 pixel image and convert it over to a numpy array.
Later, the numpy array is then converted to coordinates that OpenFOAM can read and turn into a mesh.
The script takes one input, the images name.
The script runs and saves out of the current working directory 
"""

# User inputs
dir = str(os.getcwd()) + '/'
if len(sys.argv) >= 2:
    image = sys.argv[1]
else:
    image = "test_output.jpg"

# Used for loading / saving the images
image_name = dir + image

# Opening the image and converting it to an array 
image = skimage.io.imread(image_name)
original = asarray(image) 
current = np.copy(original)

# Currently all of the solid parts will be 1
for j in range (0, 65):
    for i in range (0, 73):
        if current[i,j] >= 120:
            current[i,j] = 0
        else:
            current[i,j] = 1

# Converting the array into coordinates
current = np.rot90(current,3)

# Writing the solid coordinates file
with open(dir + 'solid_coordinates.txt', 'w') as f: 
    for k in range(0, 65):
        for i in range(0, 73):
            # Applying offset to get the coordinates to the right positon 
            x_val = k + 0.5
            y_val = i + 0.5 

            # If the coordinates are the middle of the physical domain, write the coordinates
            if current[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                f.write("%s " % (x_val))
                f.write("%s\n" % (y_val))