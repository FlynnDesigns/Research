import os 
import tarfile
import multiprocessing as mp
##########################################################################
def multi_tar_and_move(files, input_dir, output_dir):
    for file in files: 
        my_tarfile = tarfile.open(input_dir + file)
        my_tarfile.extractall(output_dir)

def multi_launch(files, input_dir, output_dir, processes=20):
    number_of_files_per_process = int(len(files)) / processes
    # Launching all processes
    my_processes = []
    for i in range(processes):
        offset = int(i * number_of_files_per_process)
        # Breaking up our files into smaller chunks
        try:
            process_files = files[offset:(offset + number_of_files_per_process)]
        except:
            process_files = files[offset:None]

        p = mp.Process(target=multi_tar_and_move, args=(process_files, input_dir, output_dir))
        p.start()
        my_processes.append(p)

    # Syncing all processes
    for job in my_processes:
        job.join()

if __name__ == "__main__":
    os.system('cls')
    print("Running...\n")
    input_dir = "A:\\godMode\\raw_gz_temp\\"
    output_dir = "A:\\godMode\\temperatures\\"

    folders = list(os.listdir(input_dir))
    for folder in folders:
        print(f"Unziping: {folder}")
        files = list(os.listdir(f"{input_dir}{folder}\\"))
        multi_launch(files, f"{input_dir}{folder}\\", output_dir)

