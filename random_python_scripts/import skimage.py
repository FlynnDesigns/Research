import skimage.io 
import numpy as np 
home = 'D:\\GAN\\run_0\\'

image = skimage.io.imread(home + "sim_images\\images\\" + str(0) + ".jpg")
image = np.array(image, dtype=np.float32)
image = image[:, :, 0]
image = image / 255
print("REE")