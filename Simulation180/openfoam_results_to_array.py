# Openfoam temperature results to array
# This function is used to extract temperature data from the simulation and save it to an text file
import os 

# Default settings
refine = 4
cells_in_x = 66 * refine # Number of cells in the x direction
cells_in_y = 74 * refine # Number of cells in the y direction
image_size = 64 * refine # Size of the image pixel x pixel

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

# Default values for the program
max_size = 90
top_dir = " "
home = str(os.getcwd())

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
os.chdir(os.getcwd() + '/constant/polyMesh')

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

# Changing back to the current directory 
os.chdir(home)
###########################################################################################################################
# Getting fluid temperatures and creating a dictionary 
temperaturesFluid = getValues(top_dir, "/fluid/T")
fluid_temp_dict = {cellZoneFluid[i]: temperaturesFluid[i] for i in range(len(cellZoneFluid))}
fluidTempAvg = sum(temperaturesFluid)/len(cellZoneFluid)

# Getting solid temperatures and creating a dictionary 
temperaturesSolid = getValues(top_dir, "/solid/T")
solid_temp_dict = {cellZoneSolid[i]: temperaturesSolid[i] for i in range(len(cellZoneSolid))}

# Creating a temp dict to store both solid and fluid temps
temp_dict = {**solid_temp_dict, **fluid_temp_dict}

# Sorting the dictionary based off of keys
temp_dict_sorted = sorted(temp_dict)
temp_dict = {key:temp_dict[key] for key in temp_dict_sorted}
temp_dict = list(temp_dict.values())

# Average every 4 values 
length = len(temp_dict)
avg_temp = []

# With 4 levels of refinement there are now for cets of the same cell to map back...
# This can scale 
for i in range(0, 4884):
    valOne = temp_dict[i]
    valTwo = temp_dict[i + 4884]
    valThree = temp_dict[i + 4884 * 2]
    valFour = temp_dict[i + 4884 * 3]
    sum = valOne + valTwo + valThree + valFour
    sum = sum / 4 
    sum = round(sum, 3)
    avg_temp.append(sum)

os.system("cd ..")
with open('temp_results.txt', 'w') as f:
    for line in avg_temp:
        f.write(f"{line}\n")
