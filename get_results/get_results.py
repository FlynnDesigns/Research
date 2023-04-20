import numpy as np 
import os
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Process
import subprocess
import scipy.io as sci

# Function used to combine a group of text files into one file in WINDOWS
def combineText(input, output):
    open(output, 'w')
    print("Combining: ", input)
    command = "cat " + input + "*.txt > " + output
    subprocess.run(['powershell', '-Command', command])

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
    plt.savefig(dir + name + '.jpg', dpi = 150)
    plt.close()

def getResults(process_files, process, start_index, home, graph):
    current_index = start_index
    for file in process_files:
        # Loading in the  temperature data (will bypass / skip files that cause errors)
        try:
            tempField = loadTemp(f"{home}temperatures\\{file}")
            tempField = np.array(tempField, dtype=np.float32)
        except:
            print(f"{file} failed")
            continue

        # Loading in the coordinates 
        fullFileName = str(file).replace("T_", "")
        coordinates = np.loadtxt(f"{home}coordinates\\{fullFileName}", delimiter=" ")
        coordinates = coordinates - 0.5
        coordinates = np.array(coordinates, dtype=int)

        # Loading in the design field 
        design = np.zeros((74,66))
        for i in range (len(coordinates)):
            x = coordinates[i, 0]
            y = coordinates[i, 1]
            design[y, x] = 1
        designField = design[5:69, 1:65]

        # Writing the vol_frac to a file 
        volFracName = fullFileName.replace(".txt", "")
        volfrac = designField.sum() / (64 * 64)
        with open(f"{home}temp_vol_frac_stats\\{process}_fake.txt", "a") as file:
            file.write(f"{volFracName}, {volfrac:.5f}\n")

        baseTemp = 300
        
        for i in range(4):  
            tempDesignField = np.rot90(designField, -i)
            currentTemp = getAvgTemp(tempDesignField, tempField)
            if currentTemp > baseTemp:
                designField = tempDesignField
                avgTemp = currentTemp
        
        # Loading in the image 
        avgTemp = getAvgTemp(designField, tempField)

        # Writing the stats to a text file 
        with open(home + 'temp_temp_stats\\' + str(process) + '.txt', 'a') as f:
            f.write(f"{volFracName}, {avgTemp}\n")

        # Plotting the temp and design field (turn this on for debugging)
        if graph == True:
            plot_design(designField, tempField, volFracName, home + "best_best\\")

        # Saving the mat version of the file to test or train
        fullFileName = fullFileName.replace(".txt", ".mat")
        tempField = np.rot90(tempField, 2)
        designField = np.rot90(designField, 2)
        mdict = {"u": tempField, "F": designField}

        # Splitting up the data set into test and train 
        if current_index < 500000:
            sci.savemat(f"{home}mat_files\\train\\train\\{fullFileName}", mdict)
        else:
            sci.savemat(f"{home}mat_files\\test\\test\\{fullFileName}", mdict)

        # Keeping track of current index
        current_index += 1


def multiP(home, processes=20, graph=False, data=None):
    # Cleaning best best dir 
    try:
        shutil.rmtree(f"{home}best_best\\")
    except:
        print("No best best to remove")
    os.mkdir(f"{home}best_best\\")

    # Cleaning mat files dir 
    try:
        shutil.rmtree(home + 'mat_files\\')
    except:
        print("No mat files to remove!")
    os.mkdir(home + 'mat_files\\')
    os.mkdir(home + 'mat_files\\train\\')
    os.mkdir(home + 'mat_files\\train\\train\\')
    os.mkdir(home + 'mat_files\\test\\')
    os.mkdir(home + 'mat_files\\test\\test\\')

    # Cleaning stats file dir 
    try: 
        shutil.rmtree(home + "temp_temp_stats\\")
    except:
        print("No stats file dir to remove")
    os.mkdir(f"{home}temp_temp_stats\\")

    # Cleaning volume fraction stats dir 
    try:
        shutil.rmtree(f"{home}temp_vol_frac_stats\\")
    except:
        print("No volume fraction stats to remove")
    os.mkdir(f"{home}temp_vol_frac_stats\\")

    # Creating stats dir if not there
    try:
        os.mkdir(f"{home}stats\\")
    except:
        pass

    # Breaking up the list of files 
    files = list(os.listdir(f"{home}temperatures\\"))
    number_of_files = len(files)
    number_of_files_per_process = int(number_of_files / processes)

    # Launching the processes
    print("Running")
    my_processes = []
    for i in range(processes):
        offset = int(i * number_of_files_per_process)
        try:
            process_files = files[offset:(offset + number_of_files_per_process)]
        except:
            process_files = files[offset:number_of_files]

        p = Process(target=getResults, args=(process_files, int(i), offset, home, graph))
        p.start()
        my_processes.append(p)

    # Syncing all processes
    for job in my_processes:
        job.join()

    # # Combining temperature stats and cleaning up files 
    if data != None:
        temp_name = f"{data}_temperature_stats.txt"
        vol_name = f"{data}_vol_frac_stats.txt"
    else:
        temp_name = f"temperature_stats.txt"
        vol_name = f"vol_frac_stats.txt"

    combineText(f"{home}temp_temp_stats\\", f"{home}stats\\{temp_name}")
    print("Cleaning up temp dir")
    shutil.rmtree(f"{input_dir}temp_temp_stats\\")
    print("Done")

    # Combining volume fraction stats and cleaning up files 
    combineText(f"{home}temp_vol_frac_stats\\", f"{home}stats\\{vol_name}")
    print("Cleaning up temp dir")
    shutil.rmtree(f"{home}temp_vol_frac_stats\\")
    print("Done")

    # Writing the file names for mat test and train 
    train_files = os.listdir(f"{home}mat_files\\train\\train\\")
    with open(f"{home}mat_files\\train\\train_val.txt", "a") as f:
        for file in train_files:
            f.write(f"{file}\n")

    test_files = os.listdir(f"{home}mat_files\\test\\test\\")
    with open(f"{home}mat_files\\test\\test_val.txt", "a") as f:
        for file in test_files:
            f.write(f"{file}\n")

if __name__ == "__main__":
    # Visualization settings 
    graph = False
    data = None

    # Input directory 
    input_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1_sim\\"
    
    # Running with parallel processes
    multiP(input_dir, graph=graph, data=data)