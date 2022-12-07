# Saving the output 
from PIL import Image
from numpy import asarray
import numpy as np
import os 
N = 64 # Image size 

# Opening the image and converting it to an array 
image = Image.open('test')
original1 = asarray(image)
original = original1[:,:,1]
current = np.copy(original)

# Currently all of the solid parts will be 1
for i in range (0, N):
    for j in range (0, N):
        if current[j,i] >= 220:
            current[j,i] = 0
        else:
            current[j,i] = 1

# Saving the array as a txt file
#np.savetxt('output_' + str(i) + '.txt', current, fmt = '%0.d', delimiter=',')

# Converting the array into coordinates
x_offset = 1
y_offset = 5
current = np.rot90(current,3)
with open('solid_coordinates'  + '.txt', 'w') as f:
    # Writing the top portion of the file
        for k in range(0, N):
           for i in range(0, N):
                x_val = x_offset + k + 0.5
                y_val = y_offset + i + 0.5 
                if current[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                    f.write("%s " % (x_val))
                    f.write("%s\n" % (y_val))