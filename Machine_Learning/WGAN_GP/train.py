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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pytorch_gan_metrics as metrics
import torchvision.utils as vutils

## Running directories ##
input_dir = 'A:\\Research\\Training_data\\run_2\\fid_is\\images\\'
stats_dir = 'A:\\Research\\Training_data\\run_2\\fid_is\\run_2_stats.npz'
test_dir = 'A:\\Research\\Training_data\\run_2\\fid_is\\test\\'
##
output_model_dir = 'A:\\Research\\Research\\Machine_Learning\\WGAN_GP\\model_dir\\'
image_dir = 'A:\\Research\\Research\\Machine_Learning\\WGAN_GP\\images\\'
loss_dir =  'A:\\Research\\Research\\Machine_Learning\\WGAN_GP\\'
image_size = 64
batch_size = 128
workers = 4
##                  ##


# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE_G = 1e-3 # was 1e-4
LEARNING_RATE_D = 1e-3 # was 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 25 # was 100
NUM_EPOCHS = 400
FEATURES_CRITIC = 64 # was 16
FEATURES_GEN = 64 # was 16
CRITIC_ITERATIONS = 10 # was 5
LAMBDA_GP = 500 # was 10

#######################################################################################
# Computer / GPU settings
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
######################################################################################

######################################################################################
# Dataset loading 
dataset = dset.ImageFolder(root=input_dir,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Grayscale(),
                            ]))

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
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
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

# Storing the loss values for plotting 
loss_gen_list = []
loss_critic_list = []
fid_score_list = []
is_score_list = []
epoch_list = []

gen.train()
critic.train()
if __name__ == "__main__":  
    epoch_num = 0
    for epoch in range(NUM_EPOCHS):
        epoch_list.append(epoch_num)
        # Target labels not needed! <3 unsupervised
        torch.multiprocessing.freeze_support()
        for batch_idx, real in enumerate(loader, 0):
            real = real[0].to(device)
            cur_batch_size = real.size(0)
    
            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
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
        loss_gen_list.append(loss_gen.cpu().detach().numpy())
        loss_critic_list.append(loss_critic.cpu().detach().numpy())
        
        with torch.no_grad():
            fake = gen(fixed_noise)
            # take out (up to) 32 examples
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

        # Generating stats for the model fid and is scores
        count = 0
        while count < 10000:
            fixed_noise = torch.randn(1024, 25, 1, 1, device=device)
            images = gen(fixed_noise)
            for num, image in enumerate(torch.split(images, 1)):
                vutils.save_image(image, test_dir + "{}.jpg".format(count))
                count = count + 1
                if count > 10000:
                    break 
        
        # Calculating scores here: 
        (IS_fake, IS_std), FID_fake = metrics.get_inception_score_and_fid_from_directory(test_dir, stats_dir)
        fid_score_list.append(FID_fake)
        is_score_list.append(IS_fake)

        # Plotting scores here 
        try:
            os.remove(loss_dir + 'scores.jpg')
        except:
            print("No scores to remove")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(epoch_list, is_score_list)
        ax1.set_title('IS Score (Image Diversity)')
        ax2.plot(epoch_list, fid_score_list)
        ax2.set_title("FD Score (Image Quality)")
        fig.savefig(loss_dir + 'scores.jpg')
        plt.close()

        # Generating visuals each epoch and saving model 
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)}, IS: {IS_fake:.3f}, FID: {FID_fake:.3f}, Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
        
        # Incrementing the epoch number 
        epoch_num = epoch_num + 1


    
