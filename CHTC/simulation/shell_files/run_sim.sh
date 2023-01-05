#!/bin/bash

ProcId=$2

# Number of simulations to be ran
simNum="100"
i="0"

#mv coordinates Simulation/
cd Simulation
source /opt/openfoam7/etc/bashrc

while [ $i != $simNum ]
do
	# Grabbing a set of new solid coordinates 
	fileNum=$((ProcId*$simNum+$i))
	echo "$(printf '%0.0f.txt' $fileNum)"
	cp -r coordinates/$(printf '%0.0f.txt' $fileNum) solid_coordinates.txt
	
	# Running the simulation
	./Allrun 
	
	# Changing the name of the output 
	cp -r temp_results.txt $(printf '0_T_%.0f.txt' $fileNum)
	
	# Incrementing i
	i=$(($i+1))
	
	# Clearing old solid_coordinates.txt
	rm -r solid_coordinates.txt
	
	# Clearing old temp results 
	rm -r temp_results.txt
done

# Zipping all of the files into one final tar file
tar cf $(printf '0_run_%0.0f.gz' $ProcId) *.txt