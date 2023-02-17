import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def getMaxTemp(tempField):
    return np.max(tempField)

def loadTemp(location):
    values = np.loadtxt(location)
    values = values.reshape((74,66))
    values = values[5:69, 1:65]
    values = np.rot90(values, 2)
    return values

def getAvgTemp(designField, tempField):
    numSolidPixels = np.sum(designField)
    solidTemps = np.multiply(designField, tempField)
    totalSolidTemp = np.sum(solidTemps)
    avgTemp = totalSolidTemp / numSolidPixels
    return avgTemp

def plot_design(designField, tempField, name, dir):
    maxTemp = getMaxTemp(tempField)
    avgTemp = getAvgTemp(designField, tempField)
    plt.rcParams["figure.autolayout"] = True 
    
    # Subplot 1 settings
    ax1 = plt.subplot(1,2,1)
    ax1.set_title('Density Field')
    ax1.axis('off')
    im1 = plt.imshow(designField, aspect='equal')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im1, cax=cax, orientation='vertical')
    
    # Subplot 2 settings
    ax2 = plt.subplot(1,2,2)
    ax2.set_title('Temperature Field')
    ax2.axis('off')
    im2 = plt.imshow(tempField, aspect='equal')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im2, cax=cax, orientation='vertical')
    
    # Main plot settings
    volfrac = np.sum(designField) / (64 * 64)
    plt.suptitle('Design: ' + name + '\n\n' + "Vol frac = " + str(round(volfrac, 3)) + ", Avg temp = " + str(round(avgTemp,3)) + ", Max temp = " + str(round(maxTemp,3)))
    plt.savefig(dir + name + '.jpg', dpi = 300)
    plt.close()

mat1 = scipy.io.loadmat('A:\\Research\\Research\\Machine_Learning\\U_NET\\samples\\data\\mat_files_270Gan_50True\\train\\train\\0__0.mat')
mat2 = scipy.io.loadmat('A:\\Research\\Research\\Machine_Learning\\U_NET\\samples\\data\\mat_files_270Gan_50True\\train\\train\\7_270_99999.mat')
design1 = np.array(mat1['F'])
design1 = np.rot90(design1, 2)
temp1 = np.array(mat1['u'])
temp1 = np.rot90(temp1, 2)
design2 = np.array(mat2['F'])
design2 = np.rot90(design2, -1)
temp2 = np.array(mat2['u'])
plot_design(design1, temp1, 'True', 'D:\\')
plot_design(design2, temp2, 'Fake', 'D:\\')

print("REE")