import torch
import torchvision.utils as vutils
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import Generator
import multiprocessing as mp

def run_generator(num_images, offset, model_dir, save_dir):
    nz = 100 # Dimension of the latent space
    ngf = 64 # Number of features in the generator 
    nc = 1 # Number of channels in the image 

    # Loading in the GAN model 
    netG = Generator(nz, nc, ngf)
    netG.load_state_dict(torch.load(model_dir))
    torch.manual_seed(0) 
    # Creating random noise vector to pass in as input to the model
    count = 0
    while count < num_images:
        
        fixed_noise = torch.randn((64, nz, 1, 1))

        images = netG(fixed_noise)

        
        for num, image in enumerate(torch.split(images, 1)):
            # Saving the raw images as a text file 
            # array = np.array(image.squeeze(0).squeeze(0).detach().numpy(), dtype=np.float32)
            # array[array < 90/255] = 0
            # array[array >= 90/255] = 1
            # np.savetxt(f"{save_dir}{count + offset}.txt", array, fmt='%f'
            # AA = torch.clone(image)
            # AA = AA.view(image.size(0), -1)
            # AA -= AA.min(1, keepdim=True)[0]
            # AA /= AA.max(1, keepdim=True)[0]
            # AA = AA.view(1, 64, 64)
            AA = torch.clone(image)
            AA += 1 
            AA *= 0.5
            array = np.array(AA.squeeze(0).squeeze(0).detach().numpy(), dtype=np.float32)
            array[array < 90/255] = 0
            array[array >= 90/255] = 1
            image = torch.tensor(array)
            
            vutils.save_image(image, f"{save_dir}{int(count + offset)}.jpg")
            # return
            # Making sure that we produce a set amount of images
            count += 1 

            # Breaking if we go over count 
            if count >= num_images:
                break 
            
# Running the main function 
if __name__ == "__main__":
    processes = 10
    totalNumImages = 25000
    model_dir = "A:\\Research\\Last_minute_paper_stuff\\attempt_1_gan\\output_models\\199.pt"
    save_dir = "A:\\review_test\\images\\"
    number_of_images = int(totalNumImages) / processes
    for i in range(processes):
        offset = i * number_of_images
        p = mp.Process(target=run_generator, args=(int(number_of_images), int(offset), model_dir, save_dir))
        p.start()