import numpy as np
from PIL import Image


num_images = 1000
for i in range(num_images):
    # gradient between 0 and 1 for 256*256
    array = np.random.randint(0,10, size = (40,40))

    # reshape to 2d

    # Creates PIL image
    img = Image.fromarray(np.uint8(array)*255, 'L')
    img.save('%d.jpg' % (i), quality=100, subsampling=0)
