import os 
import skimage.io as ski
import numpy as np
from multiprocessing import Process
from multiprocessing import Barrier
import subprocess
import shutil

# Function used to combine a group of text files into one file in WINDOWS
def combineText(input, output):
    open(output, 'w')
    print("Combining: ", input)
    command = "cat " + input + "*.txt > " + output
    subprocess.run(['powershell', '-Command', command])

# Function used to get the volume fraction of each design 
def get_vol_frac(files, input_dir, process_num, processes):
    print(f"Starting {process_num+1}/{processes}")
    for file in files:
        file = str(file)
        # Getting vol frac from image
        if '.png' in file:
            image = ski.imread(f"{input_dir}{file}")
            try:
                image = image[:,:,0]
            except:
                pass

        if '.jpg' in file:
            image = ski.imread(f"{input_dir}{file}")
            try:
                image = image[:,:,0]
            except:
                pass

        # Getting vol frac from text file 
        if '.txt' in file:
            image = np.loadtxt(f"{input_dir}{file}", delimiter=" ")
            image = image * 255

        # Getting vol frac from csv file 
        if '.csv' in file:
            image = np.loadtxt(f"{input_dir}{file}", delimiter=',')
            volFrac = image.sum() / (64 * 64)

        # Filterting the image 
        if '.csv' not in file:
            image[image < 90] = 0
            image[image >= 90] = 1 
            volFrac = image.sum() / (64 * 64)
        
        # Removing extension names on the file
        file = file.replace(".txt", "").replace(".png", "").replace(".jpg", "").replace(".csv", "")
        # Writing stats to the temp vol frac file 
        with open(f"{input_dir}temp_vol_frac_multi\\temp_{process_num}.txt", "a") as txt_file:
            txt_file.write(f"{file}, {volFrac:.5f}\n")

    # barrier.wait()

# Function used to launch multiple processes to speed up getting results 
def multiP_get_vol_frac(input_dir, output_file_name, processes=20):
    # # Creating a process barrier 
    # barrier = Barrier(processes)
    
    # Removing old output files
    try:
        os.remove(output_file_name)
        print("Removing old output file")
    except:
        pass

    # 4 lines below may not be needed (used to removed broken temp dir)
    try:
        shutil.rmtree(f"{input_dir}temp_vol_frac_multi\\")
    except:
        print("No temp dir to remove")

    # Creating temp dir to process files
    try:
        os.mkdir(f"{input_dir}temp_vol_frac_multi\\")
    except:
        return print("Couldn't create temp dir")
    
    # Breaking up the files into a set of processes
    files = list(os.listdir(input_dir))
    number_of_files = len(files)
    number_of_files_per_process = int(number_of_files / processes)

    # Launching the processes
    print("Running")
    my_proceses = []
    for i in range(processes):
        offset = int(i * number_of_files_per_process)
        try:
            process_files = files[offset:(offset + number_of_files_per_process)]
        except:
            process_files = files[offset:None]
        p = Process(target=get_vol_frac, args=(process_files, input_dir, int(i), int(processes)))
        p.start()
        my_proceses.append(p)

    # Making sure all the processes end at the same time 
    for jobs in my_proceses:
        jobs.join()

    # Cleaning up files and combining 
    combineText(f"{input_dir}temp_vol_frac_multi\\", output_file_name)
    print("Cleaning up temp dir")
    shutil.rmtree(f"{input_dir}temp_vol_frac_multi\\")
    print("Done")
    return

# Running the script 
if __name__ == "__main__":
    input_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1_gan\\designs\\"
    output_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1_gan\\stats\\fake_vol_frac_stats.txt"

    multiP_get_vol_frac(input_dir, output_dir)