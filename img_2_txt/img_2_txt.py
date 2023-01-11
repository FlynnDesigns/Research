# Custom libraries 
import skimage
from PIL import Image as im
import matplotlib.pyplot as plt 
from numpy import asarray
import numpy as np
import cv2
import pandas as pd

# Default libraries
import os
import glob

# Home directory settings: 
home = 'A:\\Research\\Training data\\run_1\\'
############################################################################
coordinates0 = home + 'coordinates\\0\\'
coordinates90 = home + 'coordinates\\90\\'
coordinates180 = home + 'coordinates\\180\\'
coordinates270 = home + 'coordinates\\270\\'
images = home + 'images\\images\\'
filteredImages = home + '\\filtered_images\\'
tarFiles = 'A:\\Research\\Research\\coordinates\\'
csvDir= home + 'csv\\'
tarFiles = home + 'coordinates\\'

# Running settings 
run = input("Would you like to run?\n This will clear all previous runs\n enter y to run: ")
if run == 'y':
    print('Clearing old files')
    files = glob.glob(coordinates0 + '*')
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

    # Adding fluid borders to image. If this is the correct size, it doesnt matter
    if size[0] == 64 or size[1] == 64:
        output = np.ones((74,66), dtype=np.uint8) * 255
        output[5:69, 1:65] = current
        current = output 
    
    # Forcing current to be uint8
    current0 = current[5:69, 1:65]
    current90 =  np.rot90(current0, -1)
    current180 = np.rot90(current90, -1)
    current270 = np.rot90(current180, -1)

    # Splitting up image name 
    name = os.path.splitext(image_name)[0]
    try:
        number = name.split('_')[0]
    except:
        number = name.split('_')[1]

    # Saving the images for 0 degrees
    skimage.io.imsave(filteredImages + '0_' + image_name, 255 * current0)
    DF = pd.DataFrame(current0)
    DF.to_csv(csvDir + '0_' + number + '.csv')
    
    # Saving the image for 90 degrees
    skimage.io.imsave(filteredImages + '90_' + image_name, 255 * current90)
    DF = pd.DataFrame(current90)
    DF.to_csv(csvDir + '90_' + number + '.csv')

    # Saving the image for 180 degrees
    skimage.io.imsave(filteredImages + '180_' + image_name, 255 * current180)
    DF = pd.DataFrame(current180)
    DF.to_csv(csvDir + '180_' + number + '.csv')

    # Saving the image for 270 degrees
    skimage.io.imsave(filteredImages + '270_' + image_name, 255 * current270)
    DF = pd.DataFrame(current270)
    DF.to_csv(csvDir + '270_' + number + '.csv')

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
    with open(coordinates0 + outputName + '.txt', 'w') as f: 
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
make_tarfile(tarFiles + 'coordinates0.gz', coordinates0)

# Zipping up the 90 degree coordinates
print('Zipping 90')
make_tarfile(tarFiles + 'coordinates90.gz', coordinates90)

# Zipping up the 180 degree coordinates 
print('Zipping 180')
make_tarfile(tarFiles + 'coordinates180.gz', coordinates180)
# Zipping up the 270 degree coordinates
print('Zipping 270')
make_tarfile(tarFiles + 'coordinates270.gz', coordinates270)