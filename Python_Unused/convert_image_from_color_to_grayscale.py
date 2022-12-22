import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.io import read_image

# Source folder
path = '/home/nathan/Desktop/ML/inputData/3x3/10k/real/'

# Destination folder
dstpath = '/home/nathan/Desktop/output_3x3/'

for image in os.listdir(path):
    # Original image directory
    color_image_dir = path + image
    
    # Grayscale color directory
    gray_image_dir = dstpath + image

    # Reading the image in 
    original = read_image(color_image_dir)

    # Converting the image to Grayscale
    grayscale_transform = transforms.Grayscale()
    grayscale = grayscale_transform(original)
    resize = transforms.Compose([transforms.Resize((64,64))])
    grayscale = resize(grayscale)

    # Saving the image 
    save_image(grayscale/255, gray_image_dir)
   


