import csv
import os

# Zone ID lists
fluid_zone_ID = []
solid_zone_ID = []

# Reading the fluid ID into memmory
with open('fluidID.csv',"r") as csvfile:
    csv_reader = csv.reader(csvfile, delimiter = ',')
    for row in csv_reader:
        a = int(row[0])
        fluid_zone_ID.append(a)
    fluid_length = len(fluid_zone_ID)

# Reading the solid ID into memmory
with open('solidID.csv',"r") as csvfile:
    csv_reader = csv.reader(csvfile, delimiter = ',')
    for row in csv_reader:
        a = int(row[0])
        solid_zone_ID.append(a)
    solid_length = len(solid_zone_ID)

with open('cellZonesStart',"r") as start:
    start_lines = [line.rstrip() for line in start]

with open('cellZonesMiddle',"r") as middle:
    middle_lines = [line.rstrip() for line in middle]

#with open('cellZonesEnd',"r") as end:
 #   end_lines = [line.rstrip() for line in end]

with open('cellZones', 'w') as f:
    # Writing the top portion of the file
    for line in start_lines:
        f.write("%s\n" % line)

    # Writing the number of zones in the fluid 
    f.write("%s\n" % fluid_length)
    f.write("%s\n" % "(")

    # Writing the fluid zone entries
    for line in fluid_zone_ID:
        f.write("%s\n" % line)

    # Writing the middle portion of the file
    for line in middle_lines:
        f.write("%s\n" % line)
    
    # Writing the numebr of zones in the solid
    f.write("%s\n" % solid_length)
    f.write("%s\n" % "(")

    # Writing the solid zone entries
    for line in solid_zone_ID:
        f.write("%s\n" % line)

    # Writing the last lines of the file
    #for line in end_lines:
    f.write("%s\b" % ");})")

print("\nDone writing File\n")