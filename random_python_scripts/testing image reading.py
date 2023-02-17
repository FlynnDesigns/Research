import skimage 
import numpy as np
from numpy import asarray

import matplotlib.pyplot as plt 
import cv2 

home = 'D:\\GAN\\run_0\\'
fileNumber = 1
# Reading in each image, image by image
image = skimage.io.imread(home + "sim_images\\images\\" + str(fileNumber) + ".jpg")
original = asarray(image) 
current = np.copy(original)

# Resizing the image to 64 x 64 if it is not the correct size 
current = cv2.resize(current, dsize=(64,64),  interpolation=cv2.INTER_CUBIC)
size = current.shape
current = current[:,:,0]
current[current < 90] = 0
current[current >= 90] = 1
plt.imshow(current)
plt.show()
current = np.array(current, dtype=np.uint8)