from tarfile import open
import os.path

location = 'A://Research//Training_data//run_1_gan//'
coordinates = 'A://Research/Training_data//run_1_gan//coordinates//'

def make_tarfile(output_filename, source_dir):
    with open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir)) 
os.system('cls')

# Zipping all files 
for i in range(8,10):
    print('Zipping ' + str(i))
    make_tarfile(location + str(i) + '.gz', coordinates + str(i) + '//')