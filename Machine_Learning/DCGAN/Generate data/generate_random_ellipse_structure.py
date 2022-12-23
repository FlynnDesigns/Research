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

def generate_random_ellipse_structure(number_of_images = 20000):
    run_directory = '/home/nathan/Desktop/Research/Datasets/GAN/Input/RandxRand/'
    os.chdir(run_directory)
    
    image_list = []
    for i in range(number_of_images):
        # Nate custom parameters
        param_1 = np.random.uniform(0.25, 2) # Some sort of scaling term
        param_2 = np.random.uniform(0.25, 1.5) # Some sort of scaling term - semi major axis
        param_3 = np.random.uniform(0.25, 1.5) # Some sort of scaling term - semi minor axis
        scaling = np.random.uniform(0.25, 2) # Some sort of scaling term

        num_x = np.random.randint(4, 10) # Number of elipses in the x direction
        num_y = np.random.randint(4, 10) # Number of elipses in the y directoin 

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

        for ellipse in del_ellipses:
            centroid, semi_minor, semi_major, ang = ellipse
            ellipse = Ellipse(centroid, param_2*scal*semi_major, param_3*scal*semi_minor,
                            angle=ang, facecolor="black", alpha=1)
            ax.add_artist(ellipse)

        plt.xlim(np.min(pts[:,0]), np.max(pts[:,0]))
        plt.ylim(np.min(pts[:,1]), np.max(pts[:,1]))
        plt.tight_layout()
        plt.axis('off')
        ax = plt.gca()
        ax.set_xlim(0.0, 64)
        ax.set_ylim(64, 0.0)

        file_name_color = "image" + str(i) + ".jpg"
        plt.savefig(file_name_color, format='JPG', bbox_inches='tight')
        plt.close()

generate_random_ellipse_structure()