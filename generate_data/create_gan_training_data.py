import numpy as np 
import multiprocessing as mp 
import os 
import shutil


def createGanData(process_files, home_dir):
    for file in process_files:
        try:
            shutil.copy2(f"{home_dir}ep_images\\images\\{file}.jpg", f"{home_dir}gan_images\\images\\{file}.jpg")
        except:
            shutil.copy2(f"{home_dir}ep_images\\images\\{file}.png", f"{home_dir}gan_images\\images\\{file}.png")

# Function to run conversion with multiple processes 
def multiP_create_gan_data(home_dir, processes=20):
    # Cleaning / working on coordinates folder 
    try:
        shutil.rmtree(f"{home_dir}gan_images\\")
        print("Removing GAN images")
    except:
        pass
    print("Making GAN images dir")
    os.mkdir(f"{home_dir}gan_images\\")
    os.mkdir(f"{home_dir}gan_images\\images\\")

    # Pull in random data 
    total_dict = {}
    with open(f"{home_dir}\\stats\\temperature_stats.txt", "r", encoding="UTF-16") as f:
        lines = f.readlines()

        for line in lines:
            line = line.replace("\n", "")
            items = line.split(", ")
            total_dict[items[0]] = float(items[1])
    keys = list(total_dict.keys())
    random_files = np.random.choice(keys, 50000, replace=False)

    # Writing the stats of the random data to a file 
    with open(f"{home_dir}\\stats\\gan_stats.txt", "a") as f:
        for design in random_files:
            temp = total_dict[design]
            f.write(f"{design}, {temp:.4f}\n")

    # Multiprocess pre-work
    number_of_files = len(random_files)
    number_of_files_per_process = int(number_of_files / processes)
    
    # Launching the processes
    for i in range(processes):
        offset = int(i * number_of_files_per_process)
        try:
            process_files = random_files[offset:(offset + number_of_files_per_process)]
        except:
            process_files = random_files[offset:None]
        p = mp.Process(target=createGanData, args=(process_files, home_dir))
        p.start()
    
if __name__ == "__main__":
    # Directory the code will run out of 
    home_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1\\"

    # Running conversion of images to coordinates in parallel 
    print("Creating GAN data")
    multiP_create_gan_data(home_dir)