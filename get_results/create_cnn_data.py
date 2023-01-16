import numpy as np
import os
from PIL import Image
import glob
import shutil

# How many samples to generate? 
train_samples = 100000
test_samples = 9922

# Cleaning dir function 
def cleanDir(location):
    files = glob.glob(location + '*')
    for f in files:
        os.remove(f)

# Directory locations 
run_0_dir = 'A:\\Research\\Training_data\\run_0\\'
run_0_mat = run_0_dir + 'mat_files\\train\\train\\'
run_1_dir = 'A:\\Research\\Training_data\\run_1\\'
run_1_mat = run_1_dir + 'mat_files\\train\\train\\'
train_dir = 'A:\\Research\\Training_data\\run_1_gan\\mat_files\\train\\train\\'
train_val_dir = 'A:\\Research\\Training_data\\run_1_gan\\mat_files\\train\\train_val.txt'
test_dir = 'A:\\Research\\Training_data\\run_1_gan\\mat_files\\test\\test\\'
test_val_dir = 'A:\\Research\\Training_data\\run_1_gan\\mat_files\\test\\test_val.txt'

# Cleaning both the test and train directories 
cleanDir(train_dir)
cleanDir(test_dir)
try:
    os.remove(train_val_dir)
except:
    print("No train val text file to delete")
try:
    os.remove(test_val_dir)
except:
    print("No test val text file to delete")

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

# Iterating through dictionary 
count = 0
for key in sorted_dict:
    # Used to split up test and training data 
    if count >= train_samples:
        active_dir = train_dir
        active_val_file = train_val_dir
    else:
        active_dir = test_dir
        active_val_file = test_val_dir

    # Parsing through the string 
    line = str(key)
    items = line.split('_')
    runNum = items[1]
    orientation = int(items[2])
    number = items[3]

    # Getting the source directory name 
    activeFileName = 'run_' + runNum + '_' + str(orientation) + '_T_' + number + '.mat'
    if runNum == '0':
        source = run_0_mat + activeFileName
    else:
        source = run_1_mat + activeFileName
        
    # Copying the file from source to the active dir 
    shutil.copy(source, active_dir)

    # Writing the mat file name to a list 
    with open(active_val_file, 'a') as fileMat:
        fileMat.write(activeFileName + '\n')

    # Increment count 
    count = count + 1

