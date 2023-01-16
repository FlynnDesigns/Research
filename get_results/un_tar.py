import tarfile
import os 

# Directory locations
run_0_dir = 'A:\\Research\\Training_data\\run_0\\raw_gz_files\\0\\'
run_90_dir = 'A:\\Research\\Training_data\\run_0\\raw_gz_files\\90\\'
run_180_dir = 'A:\\Research\\Training_data\\run_0\\raw_gz_files\\180\\'
run_270_dir = 'A:\\Research\\Training_data\\run_0\\raw_gz_files\\270\\'

def untar(location):
    for file in os.listdir(location):
        if file.endswith('.gz'):
            try:
                tar = tarfile.open(location + file)
                tar.extractall(location)
            except:
                print("Couldn't extract: ", file)
    print("Done extracting!")

untar(run_0_dir)
untar(run_90_dir)
untar(run_180_dir)
untar(run_270_dir)


