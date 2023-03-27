import torch
import torchvision.utils as vutils
## 
from torch.autograd import Variable
import torch.optim as optim
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
import scipy.io 
import argparse
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
from tqdm import tqdm
import pandas as pd 
##
from pathlib import Path
import configargparse
from src.LayoutDeepRegression import Model
from model import Generator
###################################################################################################################
def main(hparams):
    # Run settings 
    batch_size = 64
    iterations = 10000
    clip = 'stochastic'

    # Cleaning up dir 
    try:
        shutil.rmtree("D:\\godMode")
    except:
        print("No dir to remove")
    os.mkdir('D:\\godMode')

    # Loading UNET Model
    UNET = Model(hparams).cuda()
    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    ckpt = list(Path(model_path).glob("*.ckpt"))[0]
    UNET = UNET.load_from_checkpoint(str(ckpt), hparams = hparams)
    UNET.eval()
    UNET.cuda()

    # Loading WGAN-GP Model
    model_name  = "C:\\Users\\Nate\\Documents\\run_0\\gan_models\\100.pt"
    netG = Generator(100, 1, 64).cuda()
    netG.load_state_dict(torch.load(model_name))

    # Generating the design from latent variable Z
    z_approx = torch.rand((batch_size, 100, 1, 1)).cuda()
    z_approx.requires_grad = True

    # Optimizer for the latent space 
    optimizer_approx = optim.ASGD([z_approx], lr=0.0009)

    # Optimization loop
    pbar = tqdm(range(iterations), desc='', ncols=100)
    for iter in pbar:
        # Generating and evaluating the design 
        design = netG(z_approx) # Creating the design
        heat_pre = UNET(design) # Evaluating the design

        # Loss function 
        heat_pre = heat_pre * 100 + 297 # Scaling prediction back to K
        avg_temp_loss = (torch.mul(heat_pre, design).sum() / design.sum()).mean()

        # Updating the optimizer and back propagating 
        optimizer_approx.zero_grad()
        avg_temp_loss.backward()
        optimizer_approx.step()

        # Writing the loss to the progress bar
        pbar.set_description(f"Batch loss = {avg_temp_loss:.3f}")

        # Applying clipping the the latent space 
        if clip == 'standard':
            z_approx.data[z_approx.data > 1] = 1
            z_approx.data[z_approx.data < -1] = -1
        if clip == 'stochastic':
            z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
            z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)

        # Saving the initial and final designs and their temps
        if iter == 0 or iter == iterations - 1:
            offset = 0
            design_filtered = 255 * torch.clone(design)
            design_filtered[design_filtered < 90] = 0
            design_filtered[design_filtered >= 90] = 1

            if iter == iterations - 1:
                offset = batch_size
            
            for i in range(batch_size):
                # Calculating the heat field / average temp
                temp_plot = heat_pre[i, 0, :, :].detach().cpu().numpy()
                design_plot = design_filtered[i, 0, :, :].detach().cpu().numpy()
                avg_temp = np.around(np.sum(np.multiply(temp_plot, design_plot)) / np.sum(design_plot), 3)

                # Writing the design to csv file 
                DF = pd.DataFrame(design_plot)
                DF.to_csv("D:\\godMode\\" + str(int(i + offset)) + ".csv", header=False, index=False)

                # Writing the stats of the design 
                with open("D:\\godMode\\predicted_temp.txt", "a") as f:
                    f.write(f"Design: {int(i + offset)}, Temp: {avg_temp:.3f}\n")
