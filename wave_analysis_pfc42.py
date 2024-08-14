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
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import scipy.special as scsp
from sklearn import linear_model
import json

import libs_bw as lbw

# change the following to %matplotlib notebook for interactive plotting
#get_ipython().run_line_magic('matplotlib', 'inline')

# img read incoming

folder = 'C:\\Users\Lukas\Documents\GitHub\Make_BMP\VM1_AVI_230125_104431_20pa_t12'

background = pims.open(folder+'/*.bmp')[0]

forward_frames = pims.open(folder+'/wave/*.bmp')

# %%
# First look
test2 = forward_frames[0]>13
matplotlib.rc('figure',  figsize=(10, 5), dpi = 300)
matplotlib.rc('image', cmap='gray')
plt.imshow(test2)


### --- Sience Grayscale Plot --- ###

def plot_fit_group(data, pix_size, fps):
    
    arr_referenc =  np.arange(len(data))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 6)
    
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.scatter(arr_referenc,data, color='#00429d', marker='^', facecolors='none')
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Time [frames]')
    plt.ylabel('Wave position [pixel]')

    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    #ax.set_ylim(ymax=20)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    coef = np.polyfit(arr_referenc,data,1)
    poly1d_fn = np.poly1d(coef)             # poly1d_fn is now a function which takes in x and returns an estimate for y
    
    ax.plot(arr_referenc, poly1d_fn(arr_referenc), '--k') #'--k'=black dashed line, 'yo' = yellow circle marker
    
    plt.show()    

    return poly1d_fn

def plot_a(data):
    arr_referenc =  np.arange(len(data))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 6)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.plot(arr_referenc,data, color='#00429d', linewidth=0.5)
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Time [frames]')
    plt.ylabel('Wavespeed [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(125))
    #ax.yaxis.set_minor_locator(MultipleLocator(.1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    #ax.set_ylim(ymax=20)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def grayscaleplot_dataset(dataset):
    
    arr_referenc =  np.arange(len(dataset[0]))
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    for i in range(len(dataset)):
        ax.plot(arr_referenc,dataset[i], color='#00429d')
    ax.legend()
    
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('Pixel')
    plt.ylabel('Grayvalue')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    


#%%

### --- Main ---- ###


#%%

def pattern2D(data, threshold, background, fps, y1_cut, cutwidth):
    
    
    final = []
    trigger = 0
    
    bg = background[0]
    bg = bg[y1_cut:y1_cut+cutwidth,:]
    
    for i in data:
        if trigger == 0:
            tofilter = np.subtract(i[y1_cut:y1_cut+cutwidth,:],(bg*0.75))
            final = tofilter
            #final = nd.gaussian_filter(tofilter,3)
            trigger = 1;
        else:
            tofilter = np.subtract(i[y1_cut:y1_cut+cutwidth,:],(bg*0.75))
            #final = np.concatenate((tofilter, final))
            final = np.concatenate((final, tofilter))
            #final = np.concatenate((final,nd.gaussian_filter(tofilter,3)))
                  
    img_binary = final > threshold
    
    ## plot ##   
    fig = plt.figure(dpi=300) # create a 5 x 5 figure
    ax = fig.add_subplot(111)

    #adds a title and axes labels
    #ax.set_title("x [mm]", fontsize='55')
    #ax.set_xlabel("2D wave pattern at "+str(fps)+" frames per second", fontsize='55')

    #change axis direction
    #ax.invert_yaxis()
    #ax.xaxis.tick_top()
    #ax.xaxis.set_label_position('top')
    #ax.xaxis.set
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    #Edit tick
    #ax.tick_params(bottom=True, top=False, length=7, width=2) #, labelleft=False, labeltop=False
    #ax.set_xticks(np.arange(0, 24, step=2.4))

    #labeling ticks
    #tickvalues_x = np.arange(0, 24, step=2.4)
    #tickvalues_x = tickvalues_x.round(1)
    #ax.set_xticklabels(tickvalues_x, fontsize='33')
    #tickvalues_y = np.arange(len(data), 0, step=-5)
    #ax.set_yticklabels(tickvalues_y, fontsize='28')
    
    ax.imshow(img_binary, cmap="hot")
    plt.show()
    return final

#Create and Adjust 2D wavepattern

pattern = pattern2D(forward_frames, 7, background, 60, 400, 15)

#%%

def fft_pattern2D(pattern):
    image = pattern
    
    pixel_size_mm = 0.0147 
    dt = 1/60
    
    time = np.shape(pattern)[0]/30 #* 1/80 #time in s
    
    ky = np.fft.fftshift(np.fft.fftfreq(int(np.shape(pattern)[0]), d=1/80))
    kx = np.fft.fftshift(np.fft.fftfreq(np.shape(pattern)[1], d=pixel_size_mm))
    
    # Convert the image to grayscale if necessary
    if len(image.shape) > 2:
        image = np.mean(image, axis=2)
    
    # Perform the 2D Fourier transform
    f = np.fft.fftshift(np.fft.fft2(image))
    
    # Compute the magnitude squared of the Fourier coefficients
    fk_psd = np.abs(f) ** 2
    
    gradient = np.gradient(f[:int(np.shape(pattern)[0]/2-2),int(np.shape(pattern)[1]/2):])
    
    result = np.multiply(gradient[0],gradient[1])
    
    result_f = np.abs(result) ** 2
    
    # Display the F-K PSD
    fig, axs = plt.subplots(figsize=(5, 5),dpi=500)
    im = axs.imshow(np.log10(result_f), cmap='hot', extent=[0, kx.max()-10, 0, ky.max()], vmin = 14, vmax= 18)
    axs.set_xlabel('wave number, $mm^{-1}$')
    axs.set_ylabel('frequency, $s^{-1}$')
    cax = fig.add_axes([0.67, 0.3, 0.025, 0.45])
    axs.text(16, 20.5, "$log(I^2(k,\omega/2\pi))$", rotation=90, fontsize=12, fontweight="bold",
            verticalalignment = 'center')
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    #Find Max Power Spectrum
    
    binary_img = np.log10(result_f) > 15
    binary_img_reshape = binary_img[0:-150,10:150]
    
    # Get the coordinates of non-zero (white) pixels
    nonzero_indices = np.transpose(np.nonzero(binary_img_reshape))

    # Split the coordinates into x and y arrays
    x = nonzero_indices[:, 1]
    y = nonzero_indices[:, 0]

    # Reshape x and y for linear regression
    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)
 
    model = LinearRegression()
    model.fit(x_reshaped, y_reshaped)
    
    # Get the slope and intercept of the linear fit
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    
    print(intercept)
    print(slope)
    
    #fig, axs = plt.subplots()
    #plt.imshow(binary_img_reshape, cmap='gray', aspect='auto')
    #plt.axline((10, intercept), slope=slope, color='red', linestyle='--', label='Linear Maximum')
    #plt.title('Binary Image')
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

def create_density_map(matrix):
    density_map = []
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        row_density = []
        for j in range(cols):
            if matrix[i][j] == 1:
                density = 0
                for x in range(max(0, i-3), min(rows, i+4)):
                    for y in range(max(0, j-3), min(cols, j+4)):
                        if matrix[x][y] == 1:
                            density += 1
                row_density.append(density)
            else:
                row_density.append(0)
        density_map.append(row_density)
        
    # Plot the density map using Matplotlib
    plt.rcParams["figure.figsize"] = (15,15)
    plt.imshow(np.array(density_map), cmap='hot', interpolation='nearest')
    plt.title('Density Map')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Amplitude', rotation=270, labelpad=15)
    plt.show()    
    
    return density_map

#Create 2D-FFT of wave pattern 
#Calculate Amplitude based on pixel density in 7x7 matrix.
#density_map = create_density_map(pattern)
fft_pattern2D(pattern)

#%%
# Begin wave greyscale analysis
print("Shape of image series imported: " + str(backward_frames[0].shape))

#%%

def gsc_wave_analysis(data, stepsize, peak_distance, peak_height, threshold, pixelsize, exptime, background, y1_cut, cutwidth, x1_cut):
    
    peaklist2 = []

    column = 0
    
    #bg = background[y1_cut:y1_cut+cutwidth,x1_cut:]

    fig = plt.figure(figsize = (10,10), dpi=100) # create a 5 x 5 figure
    ax = fig.gca()    

    for i in data:
        #frame_croped = np.subtract(i[y1_cut:y1_cut+cutwidth,x1_cut:],(bg*.8)) > threshold            # >4 noise reduction! Adjustable &&& cutsize 400 pixel!
        frame_croped = i[y1_cut:y1_cut+cutwidth,x1_cut:900] > threshold
        data = lbw.smooth(lbw.smooth(lbw.grayscale_h2(frame_croped),2),2)
        peaks, _ = find_peaks(data, distance=peak_distance, height=peak_height)
        #
        plt.plot(data, linewidth=0.5)           #, color='#00429d'
        plt.plot(peaks, data[peaks], "x")
        #
        if not len(peaklist2):
            peaklist2 = peaks.reshape((-1, 1))
        else:
            column += 1
            temp = np.zeros(peaklist2.shape[0])
            for i in range(len(peaks)):
                trigger = False
                for n in range(peaklist2.shape[0]):
                    if abs(peaks[i]-peaklist2[n,column-1]) < 100 and trigger == False:
                        temp[n] = peaks[i]
                        trigger = True
                    elif n == peaklist2.shape[0] and trigger == False:
                        temp.append(peaks[i])
            if temp.shape[0] > peaklist2.shape[0]:  #ad zeroes to make array fit
                for i in range(peaklist2.shape[1]):
                    peaklist2[:,i] = (np.zeros(abs(temp.shape[0]-peaklist2.shape[0]))).append(peaklist2[:,i])
            peaklist2 = np.column_stack((peaklist2,temp))
    
    
    
    plt.xlabel('Pixel')
    plt.ylabel('Grayvalue')

    #removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 
    
    ax.grid(color='grey', linestyle='-', linewidth=0.2, alpha=0.5)
    ax.set_xlim(xmin=0)
    
    plt.show()
    
    conv_value = pixelsize/exptime
    speed_list = [[] for _ in range(peaklist2.shape[0])]
    #stepsize = 3
    row = column = 0
    #print(peaklist2)
    while row < peaklist2.shape[0]:
        column = 0
        temp_res = 0
        steps = 0
        trigger = True
        while column < peaklist2.shape[1]:
            #if row >= 2 and row <= 4:
            if peaklist2[row,column] != 0 and trigger == True:
                trigger = False
                temp_res = peaklist2[row,column]
                steps = 1
            elif peaklist2[row,column] != 0 and steps == stepsize:
                temp_res = (peaklist2[row,column-1]-temp_res)/steps
                speed_list[row].append(temp_res*conv_value)
                column = column - steps
                trigger = True
            elif peaklist2[row,column] == 0 and trigger == False:
                temp_res = 0
                steps = 0
                trigger = True
            elif peaklist2[row,column] != 0 and column == peaklist2.shape[1]-1:
                temp_res = (peaklist2[row,column-1]-temp_res)/steps
                speed_list[row].append(temp_res*conv_value)
            elif trigger == False:
                steps += 1
            column += 1
        row += 1
        
    new_speed_list = []
    if len(speed_list[0]) > len(speed_list[1]):
        x = len(speed_list[0])
    else:
        x = len(speed_list[1])
        
    for n in range(x):
        temp = 0
        correction = 0
        for i in speed_list:
            if n < len(i):
                temp += i[n]
            else:
                correction += 1
        new_speed_list.append(temp/(len(speed_list)-correction))
    
    peaks_r ,peaks_c = peaklist2.shape
    list_wavelen = []
    for c in range(peaks_c):
        temp = 0
        count = 0
        for r in range(peaks_r-2):
            if peaklist2[r+2,c] != 0 and peaklist2[r,c] != 0:
                #print(str(peaklist2[r,c])+' - '+str(peaklist2[r+1,c]))
                temp += abs((peaklist2[r,c]-peaklist2[r+2,c]))*pixelsize/2
                count+=1
                #list_wavelen.append((peaklist2[r,c]-peaklist2[r+1,c])*pixelsize)
        if count != 0:
            list_wavelen.append(temp/(count))
        else:
            list_wavelen.append(0)
    
    
    return new_speed_list, list_wavelen

#The STEPSIZE is the parameter that specifies the time interval, in frames, over which the phase speed is measured. Recommended > 3.
#!Stop-Start>=STEPSIZE!

#Pahse speed in mm/s
#Wave length in mm

start = 0
stop = 30
print('start: ' + str(start) + ' stop: '+str(stop))

stepsize = 5
peak_distance_max = 100
peak_height_min = 0.01
threshold = 12
pixelsize = 0.0147 #mm
exptime = 1/60 #s = 1/frames per second

sl, wl = gsc_wave_analysis(forward_frames[start:], stepsize, peak_distance_max, peak_height_min, threshold, pixelsize, exptime, background[0], 0, 300, 0)
#sl2, wl2 = gsc_wave_analysis(backward_frames[start:stop], stepsize, peak_distance_max, peak_height_min, threshold, pixelsize, exptime, background[0], 300, 300, 0)

#%%
#At this point the data is stored or/and added to the corresponding List (e.g. for different pressures), adjust!
speed_list20 = sl
wavelen_list20 = wl
#%%
speed_list20 = np.append(speed_list20,sl)
wavelen_list20 = np.append(wavelen_list20,wl)
#%%
#grayscaleplot(slf15)
#grayscaleplot(wavelen_list15)
#
with open(folder+ '_speedlist_forward.txt', 'w') as filehandle:
    json.dump(sl, filehandle)

with open(folder+ '_wavelenlist_forward.txt', 'w') as filehandle:
    json.dump(wl, filehandle)   #.tolist()
#%% Gaussian Filter to smooth the data, adjust sigma!
slf15_gauss = gaussian_filter1d(speed_list15_new, sigma=3)
slf20_gauss = gaussian_filter1d(speed_list20, sigma=4)
slf25_gauss = gaussian_filter1d(speed_list25_new, sigma=2)
slf30_gauss = gaussian_filter1d(speed_list30, sigma=4)
#%% ERROR
error_15 = abs(np.subtract(speed_list15_new,slf15_gauss))
error_20 = abs(np.subtract(speed_list20,slf20_gauss))
error_25 = abs(np.subtract(speed_list25_new,slf25_gauss))
error_30 = abs(np.subtract(speed_list30,slf30_gauss))
#%%
wl15 = gaussian_filter1d(wavelen_list15, sigma=1)
wl20 = gaussian_filter1d(wavelen_list20, sigma=1)
wl25 = gaussian_filter1d(wavelen_list25, sigma=1)
wl30 = gaussian_filter1d(wavelen_list30, sigma=1)
#%% ERROR
wl_error_15 = abs(np.subtract(wl15,wavelen_list15))
wl_error_20 = abs(np.subtract(wl20,wavelen_list20))
wl_error_25 = abs(np.subtract(wl25,wavelen_list25))
wl_error_30 = abs(np.subtract(wl30,wavelen_list30))
#%%
#### Add Group velocity #####
slf15 = np.add(slf15_gauss, 87.6)
slf20 = np.add(slf20_gauss, 76.9)
slf25 = np.add(slf25_gauss, 59.3)
slf30 = np.add(slf30_gauss, 56.7)
#%%
# Open a file in write mode ('w')
with open('slf20.txt', 'w') as file:
    # Write each element of the array to the file
    for element in slf15:
        file.write(str(element) + '\n')
#%% PLOT
#bigplot_wavelen(wl15, wl20, wl25, wl30)
#bigplot_speed(speed_list, speed_list20, speed_list25, speed_list30)
bigplot_speed(slf15, slf20, slf25, slf30, error_15, error_20, error_25, error_30)
#%%
### Calculate Statistical Error
#print(np.average(speed_list30))
print(np.average(wl15),np.average(wl20),np.average(wl25),np.average(wl30))

s = 0   ## standardabweichung ##
for i in range(len(wl30)):
    s += (wl30[i] - np.average(wl30))**2
s = np.sqrt((1/(len(wl30)))*s)
dx = s/np.sqrt(len(wl30))
print(dx)

c = c-bg
#%%
## plot ##   
fig = plt.figure(figsize = (10,30)) # create a 5 x 5 figure
ax = fig.add_subplot(111)
ax.imshow(c, cmap="gray")
plt.show()
#%%

nx = c.shape[0]
ny = c.shape[1]
x = np.arange(0,nx)
y = np.arange(0,ny)
X,Y = np.meshgrid(x,y)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# Plot a 3D surface
ax.plot_surface(X, Y, c)

plt.show()

#End

































