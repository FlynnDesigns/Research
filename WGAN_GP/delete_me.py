import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from model import Generator
import torch
import pytorch_gan_metrics as metrics
import torchvision.utils as vutils
from natsort import natsorted

gan_model_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_3_gan\\output_models\\"
test_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_3_gan\\test_dir\\"
output = "A:\\Research\\Last_minute_paper_stuff\\attempt_3_gan\\plots\\"
stats_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_3_gan\\stats\\fid_and_is_stats.npz"
gan_models = os.listdir(gan_model_dir)
gan_models = natsorted(gan_models)

print(gan_models)

# GAN settings 
nz = 25 # Dimension of the latent space
ngf = 64 # Number of features in the generator 
nc = 1 # Number of channels in the image 

# Cleaning up old stats file 
try:
    os.remove(f"{stats_dir}fid_and_is_results.txt")
    print("Old file removed")
except:
    pass

# Model count 
m_count = 0 
for gan_model in gan_models:
    print(f"Model {m_count}/{len(gan_models)}")
    # Loading in the GAN model 
    model_dir = f"{gan_model_dir}{gan_model}"
    netG = Generator(nz, nc, ngf)
    netG.load_state_dict(torch.load(model_dir))
    netG.cuda()

    # Generating test images 
    count = 0
    while count < 5000:
        fixed_noise = torch.randn((64, nz, 1, 1)).cuda()
        images = netG(fixed_noise)
        for num, image in enumerate(torch.split(images, 1)):
            vutils.save_image(image, f"{test_dir}{count}.jpg")
            count = count + 1
            if count > 5000:
                break 

    # Creating stats
    (IS_fake, IS_std), FID_fake = metrics.get_inception_score_and_fid_from_directory(test_dir, stats_dir)
    with open(f"{output}fid_and_is_results.txt", "a") as f:
        f.write(f"{IS_fake}, {FID_fake}\n")

    # Incrementing m count 
    m_count += 1