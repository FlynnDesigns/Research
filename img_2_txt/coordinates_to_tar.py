import os 
import shutil
import multiprocessing as mp
import tarfile
from natsort import natsorted
import math

# Function to write the tar files
def make_tarfile(process_files, process, output_filename, source_dir):
    print(f"Copying files for: {process}")
    for file in process_files:
        shutil.copy2(f"{source_dir}coordinates\\{file}", f"{source_dir}raw_gz_files\\{process}\\")
    
    print(f"Zipping: {process}")
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(f"{source_dir}raw_gz_files\\{process}\\", arcname=os.path.basename(source_dir)) 

    print(f"Cleaning temp dir: {process}")
    shutil.rmtree(f"{source_dir}raw_gz_files\\{process}")

def multiP_coordinates_to_tar(source_dir):
    files = os.listdir(f"{source_dir}coordinates\\")
    files = natsorted(files)
    num_files = len(files)
    if num_files % 50000 != 0:
        raise Exception("Number of files must be divisable by 50,000")

    approx_processes = int(num_files / 50000)
    processes = min(20, approx_processes)
    files_per_process =int(num_files / processes)
    
    print("Cleaning raw_gz_files_dir")
    try:
        shutil.rmtree(f"{source_dir}raw_gz_files\\")
    except:
        print("No raw_gz_files dir to remove")
    os.mkdir(f"{source_dir}raw_gz_files\\")

    for i in range(processes):
        print(f"Making temp dir: {i}")
        os.mkdir(f"{source_dir}raw_gz_files\\{i}\\")
        offset = i * files_per_process
        process_files = files[offset:offset+files_per_process]
        output_filename = f"{source_dir}raw_gz_files\\{50*i}.gz"
        p = mp.Process(target=make_tarfile, args=(process_files, i, output_filename, source_dir))
        p.start()

if __name__ == "__main__":
    # Running the function in parallel
    source_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1_gan\\"
    multiP_coordinates_to_tar(source_dir)