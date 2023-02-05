import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MeanAbsolutePercentageError
from torch.backends import cudnn
import pytorch_lightning as pl
## 
from torch.autograd import Variable
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import numpy as np
import PIL
import scipy.io as sio
import argparse
import os
import sklearn
from sklearn import preprocessing
os.environ['KMP_DUPLICATE_LIB_OK']='True'
##s
from pathlib import Path
import configargparse
from src.LayoutDeepRegression import Model
###################################################################################################################
def main(hparams):
    # # Fixing the seed 
    # seed = hparams.seed
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # cudnn.deterministic = True
    
    # Loading CNN model 
    UNET = Model(hparams).cuda()
    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    ckpt = list(Path(model_path).glob("*.ckpt"))[0]
    print(ckpt)

    model = UNET.load_from_checkpoint(str(ckpt), hparams = hparams)
    model.eval()
    model.cuda()

    # GAN model settings 
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip', default='stochastic',
                        help='disabled|standard|stochastic')
    parser.add_argument('--z_distribution', default='uniform',
                        help='uniform | normal')
    parser.add_argument('--nz', type=int, default=25,
                        help='size of the latent z vector')
    parser.add_argument('--nc', type=int, default=1,
                        help='number of channels in the generated image')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=50000,
                        help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--netG', default='dcgan_out/netG_epoch_10.pth',
                        help="path to netG (to continue training)")
    parser.add_argument('--outf', default='dcgan_out',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--profile', action='store_true',
                        help='enable cProfile')

    # Printing out user arguments 
    opt = parser.parse_args()
    print(opt)

    # Initializing the GAN ############################################################################### 
    # GAN settings 
    channels_noise = 25
    channels_img = 1
    features_g = 64
    model_name = 'A:\\Research\\Research\\Machine_Learning\\WGAN_GP\\best_100\\model_dir\\99.pt'
    ngpu = 1
    
    # Loading the model onto the gpu 
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
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

    netG = Generator(channels_noise, channels_img, features_g).to(device)
    netG.load_state_dict(torch.load(model_name))

    for param in netG.parameters():
        param.requires_grad = False

    #######################################################################################################
    # Desired temperature field 
    heat = torch.ones(64,64) * 400
    heat = (heat - 297) / 100
    heat = heat.unsqueeze(0)
    heat = heat.unsqueeze(0)
    heat = heat.cuda()

    # Storing for heat
    # z = torch.FloatTensor(1, opt.nz, 1, 1).normal_(0, 1)
    # z = Variable(z)
    # z.data.resize_(1, opt.nz, 1, 1)
    # z = z.cuda()
    try:
        os.remove("A:\\godMode\\ree.txt")
    except:
        print("Nothing to remove~")
    # transfer to gpu
    netG.cuda() 

    # Inverse Design settings 
    clip = 'stochastic'
    assert clip in ['disabled', 'standard', 'stochastic']

    # loss metrics
    mse_loss = nn.SmoothL1Loss()
    mse_loss.cuda()

    # Generating the design from latent variable Z
    z_approx = torch.randn(1, 25, 1, 1, device=device)
    
    # z_approx = Variable(z_approx)
    z_approx.requires_grad = True

    # optimizer
    # optimizer_approx = optim.Adam([z_approx], lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_approx = optim.Adam([z_approx], lr=0.01, betas=(opt.beta1, 0.999))

    # Running optimization 
    # torch.random.set_rng_state(torch.ByteTensor([148,130,74]))
    # torch.random.manual_seed(1)
    minAvgTemp = 400
    for i in range(opt.niter):
        # Generating the design from the generator 
        design = netG(z_approx)

        # Normalizing the data between 0 and 1 and then scaling to 255 
        design = design.view(design.size(0), -1)
        design = design - design.min(1, keepdim=True)[0]
        design = design / (design.max(1, keepdim=True)[0])
        design = design.view(1, 1, 64, 64)

        # Applying the sigmoid filter
        k = 100 
        design = 1 / (1 + torch.exp(-k * (design - 0.5)))
        design = 1 - design
        
        # Prediciting the heat for the design 
        heat_pre_raw = UNET(design) 
        heat_pre = heat_pre_raw * 100 + 297
        

        # Applying the loss function 
        heat_mass = torch.mul(heat_pre, design)
        heat_mass = torch.sum(heat_mass)
        mse_g_z = heat_mass / torch.sum(design)
        print("Loss = ", mse_g_z)
        ree = torch.clone(mse_g_z)

        with open("A://godMode//ree.txt", 'a') as file:
            file.write(str(ree.detach().cpu().numpy()) + "\n")

        # Updating the optimizer
        optimizer_approx.zero_grad()
        mse_g_z.backward()
        optimizer_approx.step() 

        vutils.save_image(design, 'A://godMode//design' + str(i) + '.png')

        # PLotting 
        # loss = torch.clone(mse_g_z)
        # loss = loss.detach().cpu().numpy()
        # if loss < minAvgTemp:
        #     minAvgTemp = loss
        #     print(loss)
        #     vutils.save_image(heat_pre,  'A://godMode//heat_pre' + str(i) + '.png', normalize=True)
        #     vutils.save_image(design,  'A://godMode//design' + str(i) + '.png')
        
        # with open('A://godMode//ree.txt', 'a') as ree:
            # ree.write(str(loss) + "\n")
    
        # total_solid_pixels = torch.sum(design)
        # heat_array = torch.mul(temp_heat_pre, design)
        # heat_sum = torch.sum(heat_array)
        # avg_solid_temp = torch.clone(heat_sum / total_solid_pixels)
        # avg_solid_temp = avg_solid_temp.detach().cpu().numpy()

        # # Metric to store lowest temp 
        # if avg_solid_temp < minAvgTemp:
        #     minAvgTemp = avg_solid_temp
        #     print(avg_solid_temp, ", i = ", str(i))

        # Applying heaviside filter 
        # design[design >= 100] = 0
        # design[design != 0] = 1
        # total_solid_pixels = torch.sum(design)
        # heat_array = torch.mul(heat_pre, z_approx)

        # vutils.save_image(design,  'A://godMode//design' + str(i) + '.png', normalize=True)

        # total_solid_pixels = torch.sum(z_approx)
        # heat_array = torch.mul(heat_pre, z_approx)
        # heat_sum = torch.sum(heat_array)
        # avg_solid_temp = torch.clone(heat_sum / total_solid_pixels)
        # avg_solid_temp = avg_solid_temp.detach().cpu().numpy()

        # # Metric to store lowest temp 
        # if avg_solid_temp < minAvgTemp:
        #     minAvgTemp = avg_solid_temp
        #     print(avg_solid_temp, ", i = ", str(i))
        #     vutils.save_image(z_approx,  'A://godMode//' + str(avg_solid_temp) + '.png', normalize=True)


        # Updating the calulating the error and updating z
        # g_z_approx = netG(z_approx)
        # g_z = netG(z)

        # g_z.data = heat.data
        # g_z_approx.data = heat_pre_raw.data
        
        # Loss calculation here 
        # mse_g_z = mse_loss(heat_pre, heat) - (200/8000) * torch.sum(z_approx)
        
        
        # print("[Iter {}] mse_g_z: {}".format(i, mse_g_z.item()))
        # if i % 1000 == 0:
        #     print("[Iter {}] mse_g_z: {}".format(i, mse_g_z.item()))
        #         vutils.save_image(heat_pre,  'A://godMode//g_z_approx' + str(i) + '.png', normalize=True)
            
        #     # Updating the optimizer
            # optimizer_approx.zero_grad()
            # mse_g_z.backward()
            # optimizer_approx.step() 

        # clipping
        # if clip == 'standard':
        #     z_approx.data[z_approx.data > 1] = 1
        #     z_approx.data[z_approx.data < -1] = -1
        # if clip == 'stochastic':
        #     z_approx.data[z_approx.data > 1] = random.uniform(-1, 1)
        #     z_approx.data[z_approx.data < -1] = random.uniform(-1, 1)
        
