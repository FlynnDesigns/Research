# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:39:34 2015
@author: nguarin
"""
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
sys.path.append("./..")
import ellipse_packing as ep             
import os
import multiprocessing as mp
import skimage 
import cv2

def generate_random_ellipse_structure(number_of_images, offset, save_dir):
    count = 0
    while count < number_of_images:
        # Nate custom parameters
        # param_1 = 1 #np.random.uniform(0.25, 2) # Some sort of scaling term
        # param_2 = np.random.uniform(1.25, 2.75) # Some sort of scaling term - semi major axis
        # param_3 = np.random.uniform(0.5, 1.25) # Some sort of scaling term - semi minor axis
        # scaling = 1 #np.random.uniform(0.25, 2) # Some sort of scaling term

        param_1 = 1 #np.random.uniform(0.25, 2) # Some sort of scaling term
        param_2 = np.random.uniform(0.5, 2.75) # Some sort of scaling term - semi major axis
        param_3 = np.random.uniform(0.5, 1.25) # Some sort of scaling term - semi minor axis
        scaling = 1 #np.random.uniform(0.25, 2) # Some sort of scaling term

        num_x = np.random.randint(2, 10) # Number of elipses in the x direction
        num_y = np.random.randint(2, 10) # Number of elipses in the y directoin 

        #%% Delaunay and Voronoi
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        nx = num_x
        ny = num_y
        xmin = 0
        xmax = 64
        ymin = 0
        ymax = 64
        x, y = np.mgrid[xmin:xmax:nx*1j,
                        ymin:ymax:ny*1j]
        x[:, 1::2] = x[:, 1::2] + 1/nx
        x.shape = (nx*ny, 1)
        y.shape = (nx*ny, 1)
        pts = np.hstack([x, y]) + param_1*np.random.normal(size=(nx*ny, 2))
        scal = scaling
        del_ellipses = ep.delaunay_ellipses(pts)
        # Generating the ellipses 
        for ellipse in del_ellipses:
            centroid, semi_minor, semi_major, ang = ellipse
            ellipse = Ellipse(centroid, param_2*scal*semi_major, param_3*scal*semi_minor,
                            angle=ang, facecolor="white", alpha=1)
            ax.add_artist(ellipse)

        # Plotting the images here: 
        plt.style.use('dark_background')
        plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
        plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
        plt.tight_layout()
        plt.axis('off')
        ax = plt.gca()
        ax.set_xlim(0.0, 64)
        ax.set_ylim(64, 0.0)
        fileNumber = count + offset
        file_name_color = save_dir + str(fileNumber) + ".jpg"
        plt.savefig(file_name_color, format='JPG', bbox_inches='tight', dpi=14.5)
        plt.close()

        # Reading the images back in here:
        image = skimage.io.imread(file_name_color)
        original = np.asarray(image)
        current = np.copy(original)

        # Resizing the image to 64 x 64 if it is not the correct size 
        current = cv2.resize(current, dsize=(64,64),  interpolation=cv2.INTER_CUBIC)
        size = current.shape
        current = current[:,:,0]
        current = np.array(current, dtype=np.uint8)

        # Filtering the image 
        current[current < 90] = 0
        current[current >= 90] = 1

        # Checking the volume fraction
        totalSolid = current.sum()
        volFrac = totalSolid / (64 * 64)
        if volFrac > 0 and volFrac < 0.40:
            # print(volFrac)
            count += 1

def multiP(totalNumImages, save_dir):
    processes = 20
    number_of_images = int(totalNumImages) / processes
    for i in range(processes):
        offset = i * number_of_images
        p = mp.Process(target=generate_random_ellipse_structure, args=(int(number_of_images), int(offset), save_dir))
        p.start()

if __name__ == "__main__":
    save_dir = "A:\\Research\\Last_minute_paper_stuff\\full_images\\images\\"
    totalNumImages = 100000 # input("How many images would you like to generate? ")
    multiP(totalNumImages, save_dir)