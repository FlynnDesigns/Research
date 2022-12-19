import numpy as np
from numpy import asarray
import skimage

image = skimage.io.imread('test.jpg')
data = asarray(image)
data = data[:,:,1]
output = np.ones((74,66)) * 255
output[5:69,1:65] = data
output.astype(np.uint8)
skimage.io.imsave('test_output.jpg', output)
