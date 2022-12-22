import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

home = '/home/nathan/Desktop/'
os.chdir(home)
# Creating the array
array = np.zeros((64, 64))

# Creating col 1 
array[:,0] = 1
array[:,63] = 1
array[:,3:5] = 1
array[:,8:10] = 1
array[:,13:15] = 1
array[:,18:20] = 1
array[:,23:25] = 1
array[:,28:30] = 1
array[:,33:35] = 1
array[:,38:40] = 1
array[:,43:45] = 1
array[:,48:50] = 1
array[:,53:55] = 1
array[:,58:60] = 1

plt.axis("off")
fig=plt.imshow(array, interpolation='nearest', cmap='gray_r')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('test',
    bbox_inches='tight', pad_inches=0, format='PNG')

img = Image.open('test')
img = img.resize((64,64))
img.save('0.png')