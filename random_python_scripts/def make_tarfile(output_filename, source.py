import tarfile
import os 
source_dir = "D:\\GAN\\run_0\\coordinates\\set_"
output_filename = "D:\\GAN\\run_0\\raw_gz_files\\ 0.gz"

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir)) 

make_tarfile(output_filename, source_dir)