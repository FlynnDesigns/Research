"""
Training of WGAN-GP

Programmed by Aladdin Persson <aladdin.persson at hotmail dot com>
* 2020-11-01: Initial coding
* 2022-12-20: Small revision of code, checked that it works with latest PyTorch version
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights

## Libraries I have add in ##
import torchvision.datasets as dset
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import subprocess
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pytorch_gan_metrics as metrics
import torchvision.utils as vutils

# Home dir 
home_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_3_gan\\"

# Image dir 
input_dir = f"{home_dir}gan_images\\"

# Running directories
test_dir = f"{home_dir}test_dir\\"
output_model_dir = f"{home_dir}output_models\\"  
image_dir = f"{home_dir}fake_images\\"
loss_dir =  f"{home_dir}plots\\"

# Fid and is stats dir 
stats_dir = f"{home_dir}stats\\fid_and_is_stats.npz"

# Hyperparameters and running settings
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE_G = 1e-4
LEARNING_RATE_D = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
WORKERS = 4
CHANNELS_IMG = 1
Z_DIM = 25
NUM_EPOCHS = 200
FEATURES_CRITIC = 64 
FEATURES_GEN = 64 
CRITIC_ITERATIONS = 5 # Was 5
LAMBDA_GP = 10 # Was 10

#######################################################################################
# Computer / GPU settings
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
######################################################################################

######################################################################################
# Dataset loading 
dataset = dset.ImageFolder(root=input_dir,
                            transform=transforms.Compose([
                                transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Grayscale(),
                            ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=WORKERS)
####################################################################################

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_G, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE_D, betas=(0.0, 0.9))  # was lr = LEARNING_RATE

# for tensorboard plotting
fixed_noise = torch.randn((32, Z_DIM, 1, 1)).cuda()
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

# Storing the loss values for plotting 
loss_gen_list = []
loss_critic_list = []
fid_score_list = []
is_score_list = []
epoch_list = []
epoch_fid_and_is_list = []

gen.train()
critic.train()
if __name__ == "__main__":  
    epoch_num = 0
    for epoch in range(NUM_EPOCHS):
        # Target labels not needed! <3 unsupervised
        torch.multiprocessing.freeze_support()
        for batch_idx, real in enumerate(loader, 0):
            real = real[0].to(device)
            cur_batch_size = real.size(0)
    
            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for ree in range(CRITIC_ITERATIONS):
                noise = torch.randn((cur_batch_size, Z_DIM, 1, 1)).cuda()
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        # Adding the loss values to a list
        loss_gen_temp = loss_gen.cpu().detach().numpy()
        loss_gen_list.append(loss_gen_temp)
        with open(f"{loss_dir}loss_gen.txt", "a") as file:
            file.write(f"{loss_gen_temp}\n")

        loss_critic_temp = loss_critic.cpu().detach().numpy()
        loss_critic_list.append(loss_critic_temp)
        with open(f"{loss_dir}loss_critic.txt", "a") as file:
            file.write(f"{loss_critic_temp}\n")
        
        with torch.no_grad():
            # Adding to epoch list 
            epoch_list.append(epoch_num)

            # take out (up to) 32 examples
            fake = gen(fixed_noise)
            img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    
            # Tensor board 
            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                    
            # Saving images 
            save_image(img_grid_fake,image_dir + "%d.png" % epoch, nrow=5, normalize=True)

            # Saving the models 
            torch.save(gen.state_dict(),  output_model_dir + str(epoch) + '.pt')
            
            # Plotting the loss
            try:
                os.remove(loss_dir + 'loss.jpg')
            except:
                print("No scores to remove")

            plt.plot(epoch_list, loss_gen_list)
            plt.plot(epoch_list, loss_critic_list)
            plt.savefig(loss_dir + 'loss.jpg')
            plt.close()

        # Calculating scores here: 
        if epoch_num % 5 == 0:
            # Adding to epoch list 
            epoch_fid_and_is_list.append(epoch_num)

            # Generating images for is and fid 
            count = 0
            while count < 10000:
                fixed_noise = torch.randn((1024, Z_DIM, 1, 1)).cuda()
                images = gen(fixed_noise)
                for num, image in enumerate(torch.split(images, 1)):
                    vutils.save_image(image, test_dir + "{}.jpg".format(count))
                    count = count + 1
                    if count > 10000:
                        break 

            # Calculating is and fid 
            (IS_fake, IS_std), FID_fake = metrics.get_inception_score_and_fid_from_directory(test_dir, stats_dir)
            fid_score_list.append(FID_fake)
            is_score_list.append(IS_fake)

            # Plotting scores here 
            try:
                os.remove(loss_dir + 'scores.jpg')
            except:
                print("No scores to remove")

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.plot(epoch_fid_and_is_list, is_score_list)
            ax1.set_title('IS Score (Image Diversity)')
            ax2.plot(epoch_fid_and_is_list, fid_score_list)
            ax2.set_title("FD Score (Image Quality)")
            fig.savefig(loss_dir + 'scores.jpg')
            plt.close()

        # Generating visuals each epoch and saving model 
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)}, IS: {IS_fake:.3f}, FID: {FID_fake:.3f}, Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
        
        # Incrementing the epoch number 
        epoch_num = epoch_num + 1


    
