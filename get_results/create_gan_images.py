import numpy as np
import os
from PIL import Image
import glob

# How many samples to generate? 
samples = 25000

# Cleaning dir function 
def cleanDir(location):
    files = glob.glob(location + '*')
    for f in files:
        os.remove(f)

# Directory locations 
run_0_dir = 'A:\\Research\\Training_data\\run_0\\'
run_1_dir = 'A:\\Research\\Training_data\\run_1\\'
gan_images = 'A:\\Research\\Training_data\\GAN\\run_0_and_1\\images\\'

# Reading in the files from run_0
run_0_dict = {}
with open(run_0_dir + 'stats.txt') as f:
    for line in f:
        (key, val) = line.split()
        run_0_dict[key] = val

# Reading in the files from run_1
run_1_dict = {}
with open(run_1_dir + 'stats.txt') as f:
    for line in f:
        (key, val) = line.split()
        run_1_dict[key] = val

# Merging the two dictionaries together 
total_dict = run_0_dict | run_1_dict

# Sorting the dictionary 
sorted_dict = {}
sorted_keys = sorted(total_dict, key=total_dict.get)
for w in sorted_keys:
    sorted_dict[w] = total_dict[w]

# Cleaning image dir 
cleanDir(gan_images)

# Iterating through dictionary 
count = 0
for key in sorted_dict:
    # exit condition 
    if count >= samples:
        break
    # Parsing through the string 
    line = str(key)
    items = line.split('_')
    runNum = items[1]
    orientation = int(items[2])
    number = items[3]

    if runNum == 0:
        original_image = Image.open(run_0_dir + 'images\\images\\' + str(number) + '.jpg')
    else:
        original_image = Image.open(run_1_dir + 'images\\images\\' + str(number) + '.jpg')

    # Correcting image rotation
    rotated_image = original_image.rotate(orientation)

    # Saving image 
    rotated_image.save(gan_images + line + '.jpg')

    # Increment count 
    count = count + 1

