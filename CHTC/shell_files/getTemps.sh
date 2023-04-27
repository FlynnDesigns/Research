# Moving all of the files
mv *.gz run_files
cd run_files 

# Unziping the files in 0
echo "Processing 0"
mkdir temp_files
mv 0_run_* temp_files/
cd temp_files
for f in *.gz; do tar xf "$f"; done # Unzipping 
for f in *.txt; do mv $f 0_$f; done # Changing the name 
mv *.txt ../../final_outputs/
mv *.gz ../processed_files/
cd ..
rm -r temp_files
clear

# Unziping the files in 90
echo "Processing 90"
mkdir temp_files
mv 90_run_* temp_files/
cd temp_files
for f in *.gz; do tar xf "$f"; done # Unzipping 
for f in *.txt; do mv $f 90_$f; done # Changing the name 
mv *.txt ../../final_outputs/
mv *.gz ../processed_files/
cd ..
rm -r temp_files
clear

# Unziping the files in 180
echo "Processing 180"
mkdir temp_files
mv 180_run_* temp_files/
cd temp_files
for f in *.gz; do tar xf "$f"; done # Unzipping 
for f in *.txt; do mv $f 180_$f; done # Changing the name 
mv *.txt ../../final_outputs/
mv *.gz ../processed_files/
cd ..
rm -r temp_files
clear

# Unziping the files in 90
echo "Processing 270"
mkdir temp_files
mv 270_run_* temp_files/
cd temp_files
for f in *.gz; do tar xf "$f"; done # Unzipping 
for f in *.txt; do mv $f 270_$f; done # Changing the name 
mv *.txt ../../final_outputs/
mv *.gz ../processed_files/
cd ..
rm -r temp_files
clear

