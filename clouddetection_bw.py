# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 13:04:42 2021

Grayscale Analysis

@author: Lukas Wimmer
"""

from __future__ import division, unicode_literals, print_function # Für die Kompatibilität mit Python 2 und 3.
import time
from tqdm import tqdm
startzeit = time.time()
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import SubplotDivider, Size, make_axes_locatable
from mpl_toolkits.axes_grid1.mpl_axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
from tsmoothie.smoother import *
import pims
import scipy.ndimage as nd
import scipy.stats
from scipy.stats import t
from scipy.signal import find_peaks, hilbert, chirp, argrelmax
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit
import scipy.special as scsp
from sklearn import linear_model

import optuna
import optuna.visualization as vis
from functools import partial
import plotly
import json

# change the following to %matplotlib notebook for interactive plotting
#get_ipython().run_line_magic('matplotlib', 'inline')

# img read

folder = 'D://PFC42-D3/Parabola#16-20pa-100trial70'

#data/VM2-AVI-240606-090245_30pa_0p5mA_tr100
#data/VM2-AVI-240606-085645_40pa_0p5mA_tr90

#D://PFC42-D3/Parabola#16-20pa-100trial70
#D://PFC42-D3/Parabola#0-40pa-100trial70
#Parabola#10-25pa-100trial70
#Parabola#16-20pa-100trial70
#Parabola#19-15pa-100trial70

group_frames = pims.open(folder+'/100/head/*.bmp')

background_frame = pims.open(folder+'/*.bmp')[0]
#%%
test = gaussian_filter((group_frames[1] - (background_frame*.99))[1100:1300,:],sigma=15)>.2 #5-8
#blurred_image = gaussian_filter(test, sigma=10)[850:1100,:]


# Optionally, tweak styles.
matplotlib.rc('figure',  figsize=(10, 5),dpi=600)
matplotlib.rc('image', cmap='gray')
plt.imshow(test)

### --- Generate Greyscale Horizontal ---- ###

def grey_sum(frame):
    imgshape = frame.shape
    grayscale = np.empty(imgshape[1], dtype=float)
    
    for column in range(imgshape[1]):
        sumpixel=0;
        for row in frame:
            sumpixel += row[column];
        grayscale[column] = sumpixel;
    return grayscale

### --- Sience Grayscale Plot --- ###

def plot_fit_group(limit, pix_size, fps):
    positions = np.array(limit) * pix_size  # Convert pixel indices to millimeters
    times = np.arange(len(limit)) / fps     # Convert frame indices to seconds
    
    poly_coeffs = np.polyfit(times, positions, 1)  # Linear fit
    poly_result = np.poly1d(poly_coeffs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, positions, 'o', label='Data Points')
    plt.plot(times, poly_result(times), '-', label='Polynomial Fit')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (mm)')
    plt.legend()
    plt.title('Polynomial Fit of Cloud Head Positions')
    plt.show()
    
    return poly_result #in mm/s



#%%

### --- Main ---- ###

### - Estimate Group Velocity of Cloud-Head - ### 

#Global

global_speed = []
global_error = []

def envelope(sig, distance):
    # split signal into negative and positive parts
    u_x = np.where(sig > 0)[0]
    l_x = np.where(sig < 0)[0]
    u_y = sig.copy()
    u_y[l_x] = 0
    l_y = -sig.copy()
    l_y[u_x] = 0
    
    # find upper and lower peaks
    u_peaks, _ = scipy.signal.find_peaks(u_y, distance=distance)
    l_peaks, _ = scipy.signal.find_peaks(l_y, distance=distance)
    
    # use peaks and peak values to make envelope
    u_x = u_peaks
    u_y = sig[u_peaks]
    l_x = l_peaks
    l_y = sig[l_peaks]
    
    # add start and end of signal to allow proper indexing
    end = len(sig)
    u_x = np.concatenate((u_x, [0, end]))
    u_y = np.concatenate((u_y, [0, 0]))
    l_x = np.concatenate((l_x, [0, end]))
    l_y = np.concatenate((l_y, [0, 0]))
    
    # create envelope functions
    u = scipy.interpolate.interp1d(u_x, u_y)
    l = scipy.interpolate.interp1d(l_x, l_y)
    
    return u, l


def cloudhead_pos_data(group_frames,threshold, gate, gauss_sigma, envelope_step, cut_width, cut, reverse_data, fps, pix_size):
    limit = []

    #cut = 1250 #int(group_frames[0].shape[0] - cut_width)  #cut out the bottom

    ### PLOT ###
    fig = plt.figure(figsize = (10,10), dpi=200) # create a 5 x 5 figure
    ax = fig.add_subplot(111)
    #ax limit
    ax.set_ylim(ymin=0, ymax=30)

    ### PLOT ###
    fig = plt.figure(figsize = (10,10), dpi=200) # create a 5 x 5 figure
    ax = fig.add_subplot(111)
    #ax limit
    #ax.set_ylim(ymin=0, ymax=210)
     

    items = range(len(group_frames))
    for item in tqdm(items, desc="Processing items", unit="item"):
        frame = group_frames[item]
        frame = frame - background_frame * 0.99
        prog = gaussian_filter1d(grey_sum(frame[int(cut):int(cut+cut_width),200:] > threshold), sigma=gauss_sigma)
        #prog = gaussian_filter1d(grey_sum(frame[int(cut-(cut_width/2)):int(cut+(cut_width/2)),:] >threshold), sigma=gauss_sigma)
        #prog = grey_sum(gaussian_filter(frame[int(cut-(cut_width/2)):int(cut+(cut_width/2)),:],sigma=gauss_sigma))
        #
        u, l = envelope(prog, envelope_step)
        # 
        x = np.arange(len(prog))
        value = u(x)
        if reverse_data == True:    
            prog = prog[::-1]
            value = value[::-1]
            
        check = 0
        for i in range(len(value)):
            if value[i] > gate and check == 0:
                limit.append(i)
                check = i      
        
        #for i in range(len(prog)):
        #    if prog[i] > gate and check == 0:
        #        limit.append(i)
        #        check = i
        
        plt.plot(x, value, label="envelope")
        ax.plot(prog, linewidth=0.8, label="Flux")           #, color='#00429d'
        #
        if check != 0:
            ax.axvline(check, linestyle='dashed', color='r');
        #
    plt.show()

    if reverse_data == True:    
        limit = limit[::-1]
    
    poly_result2 = plot_fit_group(limit, pix_size, fps)
    result2 = np.polyder(poly_result2).coeffs[0]    #first coefficient -> slope
    
    print('df: ' + str(result2) + 'function: ' + str(poly_result2))
    

    # Calculate the standard deviation of the residuals
    s2 = np.sqrt(np.sum((np.array(limit) - poly_result2(np.arange(len(limit)) / fps) / pix_size) ** 2) / (len(limit) - 1))
    # Calculate the error of the slope
    dx2 = s2 / np.sqrt(len(limit))
    
    ### error of v ###
    
    dx2_in_mm = dx2
    v_in_mms = result2
    s_in_mm = (limit[-1]-limit[0])
    dv_in_mms = v_in_mms*dx2_in_mm/s_in_mm
    
    print("Group speed v = " + str(v_in_mms) + " /pm " + str(dv_in_mms) + " /frac(mm)(s)")
    
    return dv_in_mms, v_in_mms    # dv_in_mms is the error

# function to minimize
def objective(trial, group_frames, cut, cut_width, reverse_data, fps, pix_size):
    try:    
        #----------Parameter Space----------
        threshold = trial.suggest_float('threshold',1.2, 2.4) #1.2, 2.5(1-4) xiq #5-7 vm2-pk4gi
        gate = trial.suggest_float('gate', 4,20)   #15-25 pk4gi # 5, 15 iss
        gauss_sigma = trial.suggest_float('gauss_sigma', 5, 25)
        envelope_step = trial.suggest_int('envelope_step', 20, 100)
        #-----------------------------------
        error, velocity = cloudhead_pos_data(
            group_frames[2:12], threshold, gate, gauss_sigma,
            envelope_step, cut_width, cut, reverse_data, fps, pix_size
        )

        global_speed.append(velocity)
        global_error.append(error)

        return error
    except Exception as e:
        # Return infinity when an exception occurs (threshold or gate too high -> no data points for fit)
        print(f"Exception: {e}")
        return float('inf')

# Adjustables
fps = 60    #vm2-pk-4gi 30
pix_size = 0.0118  # in mm #iss 0.0143 #ixQ 0.0118
cut = 1100
cut_width = 200
reverse_data = False #True cloud coming from the left; False cloud coming from the right of the image
trials = 30

# Create a partial function to pass additional fixed parameters to the objective function
objective_partial = partial(
    objective, group_frames=group_frames, cut=cut, cut_width=cut_width,
    reverse_data=reverse_data, fps=fps, pix_size=pix_size
)

# Bayesian optimization with optuna
study = optuna.create_study(direction='minimize')
study.optimize(objective_partial, n_trials=trials)      # n_trials: number of iterations

# extract best parameters and corresponding error from the created study
best_params = study.best_params
best_error = study.best_value

# Print the best parameters and error
print('----------------------------------------------------------')
print("Best Parameters:", best_params)
print("Best Error:", best_error)
print('----------------------------------------------------------')

optimization_history_plot = vis.plot_optimization_history(study)
optimization_history_plot.show()

param_importance_plot = vis.plot_param_importances(study)
param_importance_plot.show()

contour_plot = vis.plot_contour(study, params=["threshold", "gate"])
contour_plot.show()
#%%
# Combine errors and velocities using zip
data = list(zip(global_error, global_speed))

# Sort the data based on error values
sorted_data = sorted(data, key=lambda x: x[0])

# Extract the top 5 minimum errors and corresponding velocities
top_5_errors_and_velocities = sorted_data[:20]

# Separate errors and velocities into two lists
top_5_errors, top_5_velocities = zip(*top_5_errors_and_velocities)

# Create a dictionary to hold the data
output_data = {
    "errors": list(top_5_errors),
    "velocities": list(top_5_velocities)
}
#%%
#Save the data to a JSON file
output_filename = folder[17:]+"_headspeed.json"
#output_filename = folder[13:28]+"-t30_headspeed.json"
with open(output_filename, 'w') as json_file:
    json.dump(output_data, json_file)





















