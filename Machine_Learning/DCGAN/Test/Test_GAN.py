import torch
import torchvision.utils as vutils
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os

def gen_fake_images(model_name):
    num_images = 50000
    save_dir = '/home/nathan/Desktop/Research/ML/fake'
    nz = 25
    ngpu = 1
    ngf = 64 # Leave this value alone
    batch_size = 64 # This value depends
    nc = 1 # Leave this value alone 

    # Computer settings
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Default generator model
    # Stock from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                ## We hope to get State size. (ngf*8) x 5 x 5
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8

                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        def forward(self, input):
            return self.main(input)

    # Create the generator
    netG = Generator(ngpu).to(device)
    netG.load_state_dict(torch.load(model_name))
    os.chdir(save_dir)

    # Creating random noise vector to pass in as input to the model
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    count = 0
    while count < num_images:
        with torch.no_grad():
            images = netG(fixed_noise).detach().cpu()

        for num, image in enumerate(torch.split(images, 1)):
            vutils.save_image(image, "{}.jpg".format(count))
            count = count + 1
            if count > num_images:
                break 
gen_fake_images('/home/nathan/Desktop/Research/ML/35000.pt')