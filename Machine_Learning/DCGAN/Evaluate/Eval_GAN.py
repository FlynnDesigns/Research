import os
from pytorch_gan_metrics import get_inception_score_and_fid_from_directory
from Test_GAN import gen_fake_images 

# Home dir 
model_dir = '/home/nathan/Desktop/ML/'
os.chdir(model_dir)

def getStats(model_dir):
    for filename in sorted(os.listdir(model_dir), key=lambda x: int(x.replace(".pt", ""))):
        print(filename)
        
        # Generating 10000 fake images for testing
        model_name = model_dir + filename
        gen_fake_images(model_name)

        # Evaluating the Performance of the generated images 
        fake_img_dir = '/home/nathan/Desktop/ML/fake'
        stats = '/home/nathan/Desktop/ML/inputData/4x4/stats/stats.npz'
        (IS_fake, IS_std), FID_fake = get_inception_score_and_fid_from_directory(fake_img_dir, stats)

        # print("IS fake = ", IS_fake)
        # print("FID fake = ", FID_fake)
        with open(model_dir + 'FID.txt', 'a') as f:
            f.write(str(FID_fake) + ",\n")

        with open(model_dir + 'IS.txt', 'a') as f:
            f.write(str(IS_fake) + ",\n")

target_home = '/home/nathan/Desktop/ML/outputData/4x4/gray/'
batch_20k = target_home + '20K_batch_8/models/'
batch_10k = target_home + '10K_batch_8/models/'
batch_5k =  target_home + '5K_batch_8/models/'
batch_3k =  target_home + '2.5K_batch_8/models/'
batch_1k =  target_home + '1.25K_batch_8/models/'

getStats(batch_20k)
getStats(batch_10k)
getStats(batch_5k)
getStats(batch_3k)