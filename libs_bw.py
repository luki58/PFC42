# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:59:15 2024

@author: Lukas
"""

from __future__ import division, unicode_literals, print_function # Für die Kompatibilität mit Python 2 und 3.
import matplotlib
import numpy as np
import math
from tsmoothie.smoother import *

### --- Generate Greyscale Horizontal ---- ###

def grayscale_h2(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel/imgshape[0];
    return grayscale

def grey_sum(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel;
    return grayscale

### --- Generate Greyscale Vertical ---- ###
       
def grayscale_v(frame):
    imgshape = frame.shape
    print(imgshape)
    grayscale = np.empty(imgshape[0], dtype=float)
    numrow=0;
    
    for row in frame:
        sumpixel=0;
        for column in range(imgshape[0]):
            sumpixel += row[column];
        grayscale[numrow] = sumpixel/imgshape[1];
        numrow+=1;
    return grayscale    

### --- Autocrop Streamline --- ###

def crop_coord_y(frame):
    grayscaley = grayscale_v(frame)
  
    ## operate data smoothing ##
    smoother = ConvolutionSmoother(window_len=50, window_type='ones')
    smoother.smooth(grayscaley)
    
    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=3)
    
    data = smoother.smooth_data[0];
    
    data = smooth(data,2)
    
    #plot_a(data)
    
    return np.array(data).argmax()
           
### smoothen data ###

def smooth(data,sigma):
    smoother = ConvolutionSmoother(window_len=50, window_type='ones')
    smoother.smooth(data)
    # generate intervals
    low, up = smoother.get_intervals('sigma_interval', n_sigma=sigma)
    return smoother.smooth_data[0]  