#!/bin/bash
# Command line inputs 
ProcId=$2
Cluster=$3
Process=$4
NumberOfSimulations=$5
DesignsPerFile=$6
FileName=$7
CustomOffset=$8

# Number of simulations to be ran
simNum="$NumberOfSimulations"

# Unzipping the files 
tar xf *.gz

# Moving all of the coordinates 
mkdir coordinates
mv *.txt coordinates/

# Moving the coordinates folder into the simulation dir 
mv coordinates Simulation/

cd Simulation
source /opt/openfoam7/etc/bashrc

i="0"
while [ $i != $simNum ]
do
	# Grabbing a set of new solid coordinates 
	fileNum=$(($ProcId*$simNum+$i))
	offset=$(($DesignsPerFile*$FileName+$CustomOffset))
	trueFileNum=$(($fileNum+$offset))
	fileNum=$trueFileNum
	echo "$(printf '%0.0f.txt' $fileNum)"

	# Pulling in the correct coordinate
	cp -r coordinates/$(printf '%0.0f.txt' $fileNum) solid_coordinates.txt
	
	# Running the simulation
	./Allrun 
	
	# Changing the name of the output 
	cp -r temp_results.txt $(printf 'T_%0.0f.txt' $fileNum)
	
	# Clearing old solid_coordinates.txt
	rm -r solid_coordinates.txt
	
	# Clearing old temp results 
	rm -r temp_results.txt

	# Incrementing i
	i=$(($i+1))
done

# Zipping all of the files into one final tar file
tar cf $(printf '%0.0f_%0.0f.gz' $Cluster $Process) *.txt