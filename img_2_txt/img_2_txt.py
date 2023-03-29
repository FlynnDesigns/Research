# Installed libraries 
import skimage.io as ski
import numpy as np

# Default libraries
import os
import os.path
import shutil
import multiprocessing as mp
import tarfile

# Custom llibraries
from coordinates_to_tar import multiP_coordinates_to_tar

# Function used to create coordinates 
def createCoordinates(designLayout, outputName):
    with open(outputName, 'w') as f: 
        for k in range(66):
            for i in range(74):
                # Applying offset to get the coordinates to the right positon 
                x_val = k + 0.5 
                y_val = i + 0.5 

                # If the coordinates are the middle of the physical domain, write the coordinates
                if designLayout[k][i] == 1 and x_val > 0.5 and x_val < 65.5 and y_val > 4.5 and y_val< 73.5:
                    f.write("%s " % (x_val))
                    f.write("%s\n" % (y_val))

# Create tarfile of coordinates in the end 
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir)) 

# Converting raw_images to coordinates 
def convertImagesToTxt(files, home_dir, design_dir):
    # Converting all of assigned images per process into text 
    for file in files:
        # Code for reading in .txt files 
        if ".txt" in file:
            design = np.loadtxt(f"{design_dir}{file}")
            design = design * 255
            name = file.replace(".txt","")

        # Code for reading in .png files 
        if ".png" in file:
            design = ski.imread(f"{design_dir}{file}")
            design = design[:,:,0]
            name = file.replace(".png","")

        # Code for reading in .jpg files 
        if ".jpg" in file:
            design = ski.imread(f"{design_dir}{file}")
            design = design[:,:,0]
            name = file.replace(".jpg","")

        if ".csv" in file:
            design = np.loadtxt(f"{design_dir}{file}", delimiter=",")
            design = design * 255
            name = file.replace(".csv","")
            
        # Adds fluid borders to the image 64x64 -> 74x66 
        full_domain = np.ones((74,66), dtype=np.uint8) * 255
        full_domain[5:69, 1:65] = design

        # Filtering the design 
        for j in range(66):
            for i in range(74):
                if full_domain[i,j] < 90:
                    full_domain[i,j] = 0
                else:
                    full_domain[i,j] = 1

        # Rotating current coordinates so that they show up correctly 
        full_domain[5:69, 1:65] = np.rot90(full_domain[5:69, 1:65], 2)
        full_domain = np.flip(full_domain, axis=1)
        
        # Transposing everything to get it centered correctly 
        full_domain = np.transpose(full_domain)
        # print(name)
        # Creating physical coordinates
        createCoordinates(full_domain, f"{home_dir}coordinates\\{name}.txt")

# Function to run conversion with multiple processes 
def multiP_img_2_txt(home_dir, design_dir, processes=20):
    # Cleaning / working on coordinates folder 
    try:
        shutil.rmtree(home_dir + "coordinates\\")
        print("Removing coordinates dir")
    except:
        pass
    print("Making coordinates dir")
    os.mkdir(home_dir + "coordinates\\")

    # Cleaning / working on raw_gz_files folder 
    try:
        shutil.rmtree(home_dir + '\\raw_gz_files\\')
        print("Removing raw_gz_files dir")
    except:
        pass
    print("Making raw_gz_files dir")
    os.mkdir(home_dir + "raw_gz_files\\")

    # Breaking up the list of files
    files = list(os.listdir(design_dir))
    number_of_files = len(files)
    number_of_files_per_process = int(number_of_files / processes)
    
    # Launching the processes
    for i in range(processes):
        offset = int(i * number_of_files_per_process)
        try:
            process_files = files[offset:(offset + number_of_files_per_process)]
        except:
            process_files = files[offset:None]
        p = mp.Process(target=convertImagesToTxt, args=(process_files, home_dir, design_dir))
        p.start()
        p.join()

    # Zipping files when done
    multiP_coordinates_to_tar(home_dir)

    
if __name__ == "__main__":
    # Directory the code will run out of 
    home_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1_gan\\"
    design_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1_gan\\designs\\"

    # Running conversion of images to coordinates in parallel 
    print("Running image to text conversion")
    multiP_img_2_txt(home_dir, design_dir)