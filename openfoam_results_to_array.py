# Openfoam temperature results to array
# This function is used to extract temperature data from the simulation and save it to an text file
import os 
import numpy as np

# Default settings 
cells_in_x = 66 # Number of cells in the x direction
cells_in_y = 74 # Number of cells in the y direction
image_size = 64 # Size of the image pixel x pixel

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

# Changing to working directory
home = '/home/nathan/Desktop/Current_SIM/'
os.chdir(home)

# Default values for the program
max_size = 0
top_dir = " "

# Getting the most recent results from the simulation
files = [f for f in os.listdir(home) if os.path.isdir(f)]
for f in files:
    # Converting the folder to a string
    name = str(f)
    # Checking if the folder name is an integer
    if name.isdigit():
        name = int(name)
        # Checking to see if the folder is the largest in the folder
        if name > max_size:
            # Checking to see if the folder has all the sub dirs we need to get the results
            if len(next(os.walk(f))[1]) > 2:
                max_size = name
                top_dir = f

# Getting the cell zone IDs
os.chdir(home + 'constant/polyMesh')

# Variables for file parsing 
start = 0
end = 0

# List to store all cellZone IDS
cellZoneSolid = []
cellZoneFluid = []

# Function to get values from openfoam output files
def getValues(top_dir, fileName):
    # Variables for file parsing 
    start = 0
    end = 0
    list = []
    with open(top_dir + fileName) as myFile:
        for myLine in myFile:

            if myLine == '(\n':
                start = start + 1
            
            if myLine == ')\n':
                end = end + 1
            
            # Removing newline character
            myLine = myLine.strip()

            if start == 1 and end == 0 and isfloat(myLine):
                # Converting string to int
                myNewLine = float(myLine)

                # Adding ID to list
                list.append(myNewLine) 
    return list

# Function to convert dict to array
def getArray(dict, cells_in_x = 66, cells_in_y = 74, image_size = 64):
    output = list(dict.values())
    output = np.array(output).reshape(-1, cells_in_x)
    output = np.flip(output, 0)
    column_spacing = (cells_in_x - image_size) / 2 
    row_spacing = (cells_in_y - image_size) / 2
    col_start = int(column_spacing)
    col_end = int(column_spacing + image_size)
    row_start = int(row_spacing - 1)
    row_end = int(row_spacing + image_size - 1)
    output = output[row_start:row_end, col_start:col_end]
    return output

# Reading the polyMesh file for numbers
with open('cellZones') as myFile:
    for myLine in myFile:

        if myLine == '(\n':
            start = start + 1
        
        if myLine == ')\n':
            end = end + 1
        
        # Removing newline character
        myLine = myLine.strip()

        if start == 2 and end == 0 and myLine.isdigit():
            # Converting string to int
            myNewLine = int(myLine)

            # Adding ID to list
            cellZoneFluid.append(myNewLine)

        elif start == 3 and end == 1 and myLine.isdigit():
            # Converting string to int
            myNewLine = int(myLine)

            # Adding ID to list
            cellZoneSolid.append(myNewLine)

os.chdir(home)
###########################################################################################################################
# Getting fluid temperatures
temperaturesFluid = getValues(top_dir, "/fluid/T")
fluid_temp_dict = {cellZoneFluid[i]: temperaturesFluid[i] for i in range(len(cellZoneFluid))}

# Getting solid temperatures
temperaturesSolid = getValues(top_dir, "/solid/T")
solid_temp_dict = {cellZoneSolid[i]: temperaturesSolid[i] for i in range(len(cellZoneSolid))}

# Creating a temp dict to store both solid and fluid temps
temp_dict = {**solid_temp_dict, **fluid_temp_dict}

# Sorting the dictionary based off of keys
temp_dict_sorted = sorted(temp_dict)
temp_dict = {key:temp_dict[key] for key in temp_dict_sorted}
temp = getArray(temp_dict, cells_in_x)
#############################################################################################################################
# Getting fluid pressures
pressuresFluid = getValues(top_dir, "/fluid/p")
fluid_pressure_dict = {cellZoneFluid[i]: pressuresFluid[i] for i in range(len(cellZoneFluid))}

# Getting solid pressures
pressuresSolid = [0] * len(temperaturesSolid)
solid_pressure_dict = {cellZoneSolid[i]: pressuresSolid[i] for i in range(len(cellZoneSolid))}

# Creating a temp dict to store both solid and fluid pressures
pressure_dict = {**solid_pressure_dict, **fluid_pressure_dict}

# Sorting the dictionary based off of keys
pressure_dict_sorted = sorted(pressure_dict)
pressure_dict = {key:pressure_dict[key] for key in pressure_dict_sorted}
pressure = getArray(pressure_dict, cells_in_x)
###########################################################################################################################
# Getting fluid density field 
densityFluid = [0] * len(temperaturesFluid)
fluid_density_dict = {cellZoneFluid[i]: densityFluid[i] for i in range(len(cellZoneFluid))}

# Getting solid density field 
densitySolid = [1] * len(temperaturesSolid)
solid_density_dict = {cellZoneSolid[i]: densitySolid[i] for i in range(len(cellZoneSolid))}

# Creating a temp dict to store both solid and fluid pressures
density_dict = {**solid_density_dict, **fluid_density_dict}

# Sorting the dictionary based off of keys
density_dict_sorted = sorted(density_dict)
density_dict = {key:density_dict[key] for key in density_dict_sorted}
density = getArray(density_dict, cells_in_x)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(density)
ax.set_aspect('equal')
cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()

fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(pressure)
ax.set_aspect('equal')
cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()