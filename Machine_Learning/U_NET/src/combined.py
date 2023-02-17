import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsolutePercentageError
from torch.backends import cudnn
import pytorch_lightning as pl
## 
from torch.distributions import normal
from torch.autograd import Variable
import torch.optim as optim
import random
import shutil
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy.io as sio
import argparse
import os
import sklearn
from sklearn import preprocessing
os.environ['KMP_DUPLICATE_LIB_OK']='True' 
from tqdm import tqdm
import pandas as pd 
##
from pathlib import Path
import configargparse
from src.LayoutDeepRegression import Model
import skimage
###################################################################################################################
def main(hparams):
    # Cleaning up dir 
    try:
        shutil.rmtree("D:\\godMode")
    except:
        print("No dir to remove")
    os.mkdir('D:\\godMode')

    # Loading CNN model 
    ckpt = 'A:\\Research\\Research\\Machine_Learning\\U_NET\\lightning_logs\\version_81\\checkpoints\\epoch=78-step=246875.ckpt'
    
    UNET = Model(hparams).cuda()
    UNET = UNET.load_from_checkpoint(str(ckpt), hparams = hparams)
    UNET.eval()
    UNET.cuda()

    # GAN model settings 
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', default='stochastic', help='disabled|standard|stochastic')
    parser.add_argument('--z_distribution', default='uniform', help='uniform | normal')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--batchSize', type=int, default=1, help='batch optimization size')
    parser.add_argument('--nc', type=int, default=1, help='number of channels in the generated image')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='dcgan_out/netG_epoch_10.pth', help="path to netG (to continue training)")
    parser.add_argument('--outf', default='dcgan_out', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manu    al seed')
    parser.add_argument('--profile', action='store_true', help='enable cProfile')

    # Printing out user arguments 
    opt = parser.parse_args()
    print(opt)

    # Initializing the GAN ############################################################################### 
    model_name = 'D:\\GAN\\run_0\\GAN_models\\70.pt'
    
    # WGAN Generator  
    class Generator(nn.Module):
        def __init__(self, channels_noise, channels_img, features_g):
            super(Generator, self).__init__()
            self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        def _block(self, in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        def forward(self, x):
            return self.net(x)

    netG = Generator(opt.nz, opt.nc, opt.ngf)
    netG.load_state_dict(torch.load(model_name))
    netG.cuda() 

    for param in netG.parameters():
        param.requires_grad = False

    for iter_design in range(100):
        ## Stats bar ##
        pbar = tqdm(range(opt.niter), desc='', ncols=100)

        # Generating the design from latent variable Z
        # torch.manual_seed(3)
        z_approx = torch.rand(opt.batchSize, opt.nz, 1, 1).cuda()
        z_approx = Variable(z_approx)
        z_approx.requires_grad = True

        ## optimizer ##
        optimizer_approx = optim.ASGD([z_approx], lr=0.0009)

        base_temp = 330
        for iter in pbar:
            # Generating the design from the generator 
            design = netG(z_approx)

            # New filtering technique
            design = design.squeeze(0)
            design = design * 255 
            design = design + 0.5 
            design = design.clamp_(0, 255)
            design = design / 255
            if opt.batchSize == 1:
                design = design.unsqueeze(0)
            
            # Prediciting the heat for the design 
            heat_pre = UNET(design) 
            heat_pre = heat_pre * 100 + 297

            # Loss function and optimizer 
            design_filtered = torch.clone(design) * 255
            design_filtered[design_filtered < 90] = 0
            design_filtered[design_filtered >= 90] = 1
            avg_temp_approx = (torch.mul(heat_pre, design_filtered).sum() / design_filtered.sum()).mean()
            avg_temp_approx_cpu = np.around(avg_temp_approx.detach().cpu().numpy(), 3)

            if avg_temp_approx_cpu < base_temp:
                base_temp = avg_temp_approx_cpu
                best_design = torch.clone(design_filtered)
                best_heat = torch.clone(heat_pre)

            with open("D:\\godMode\\plot.txt", "a") as file:
                file.write(str(avg_temp_approx_cpu) + "\n")
            pbar.set_description("Approx avg temp solid = " + str(avg_temp_approx_cpu))

            # Updating the optimizer
            optimizer_approx.zero_grad()
            avg_temp_approx.backward()
            optimizer_approx.step()
            
            # Saving the first iteration's design
            if iter == 0:
                for iter in range(opt.batchSize):
                    temp_plot = heat_pre[iter, 0, :, :].detach().cpu().numpy()
                    design_plot = design_filtered[iter, 0, :, :].detach().cpu().numpy()

                    avg_temp = np.around(np.sum(np.multiply(temp_plot, design_plot)) / np.sum(design_plot), 3)
                    DF = pd.DataFrame(design_plot)
                    DF.to_csv("D:\\godMode\\" + str(iter_design) + ".csv", header=False, index=False)
                    with open("D:\\godMode\\predicted_temp.txt", "a") as f:
                        f.write(f"Design: {iter_design}, Temp: {avg_temp:.3f}\n")

        # Plotting and saving all designs 
        design_plot = best_design[0, 0, :, :].detach().cpu().numpy()
        temp_plot = best_heat[0, 0, :, :].detach().cpu().numpy()
        avg_temp = np.around(np.sum(np.multiply(temp_plot, design_plot)) / np.sum(design_plot), 3)
        DF = pd.DataFrame(design_plot)
        DF.to_csv("D:\\godMode\\" + str(iter_design + 100) + ".csv", header=False, index=False)
        with open("D:\\godMode\\predicted_temp.txt", "a") as f:
            f.write(f"Design: {iter_design + 100}, Temp: {avg_temp:.3f}\n")