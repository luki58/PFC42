# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 17:05:52 2024

@author: Lukas
"""

import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import codecs, json
import pandas
#%% PLOTS
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

def plot_2025(data, data2, title):
    arr_referenc =  np.arange(len(data))
    arr_referenc2 =  np.arange(len(data2))
 
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.scatter(arr_referenc,data, marker='^', color='#00429d', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc2,data2, marker='s', color='#00cc00', linewidth=.7, s=5, facecolors='none')

    ax.legend(['20 Pa', '25 Pa'], loc='upper right', prop={'size': 9})
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('T [frames]')
    plt.ylabel('$C_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    
    
def plot_30(data, data2, title):
        arr_referenc =  np.arange(len(data))
        arr_referenc2 =  np.arange(len(data2))
     
        fig, ax = plt.subplots(dpi=600)
        fig.set_size_inches(6, 3)
        #x_peak = find_peaks(data,height=2.4)
        #print(x_peak[0][0])
        #fig.savefig('test2png.png', dpi=100)
        #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
        ax.scatter(arr_referenc,data, marker='^', color='#00429d', linewidth=.7, s=5, facecolors='none')
        ax.scatter(arr_referenc2,data2, marker='s', color='#00cc00', linewidth=.7, s=5, facecolors='none')

        ax.legend(['30 Pa, forward', '30 Pa, backward'], loc='upper right', prop={'size': 9})
        
        #adds a title and axes labels
        ax.set_title(title)
        plt.xlabel('T [frames]')
        plt.ylabel('$C_{DAW}$ [mm]')
     
        #removing top and right borders
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False) 
        
        #Edit tick 
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        #add vertical lines
        #ax.axvline(left, linestyle='dashed', color='b');
        #ax.axvline(right, linestyle='dashed', color='b');

        #adds major gridlines
        ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
        
        #ax limit
        ax.set_xlim(xmin=0)
        
        #legend
        #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
        
        plt.show()    

def bigplot_2(data, data2, title, legend):
    arr_referenc =  np.arange(len(data))
    arr_referenc2 =  np.arange(len(data2))
    
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.scatter(arr_referenc,data, marker='^', color='#00429d', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc2,data2, marker='s', color='#00cc00', linewidth=.7, s=5, facecolors='none')
    ax.legend(legend, loc='upper right', prop={'size': 9})
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('T [frames]')
    plt.ylabel('$C_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def bigplot_3(data, data2, data3, title):
    arr_referenc =  np.arange(len(data))
    arr_referenc2 =  np.arange(len(data2))
    arr_referenc3 =  np.arange(len(data3))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.scatter(arr_referenc,data, marker='^', color='#00429d', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc2,data2, marker='s', color='#00cc00', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc3,data3, marker='o', color='#ff8000', linewidth=.7, s=5, facecolors='none')
    ax.legend(['20 Pa', '25 Pa', '30 Pa'], loc='upper right', prop={'size': 9})
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('T [frames]')
    plt.ylabel('$C_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def bigplot_4(data, data2, data3, data4, title, legend):
    arr_referenc =  np.arange(len(data))
    arr_referenc2 =  np.arange(len(data2))
    arr_referenc3 =  np.arange(len(data3))
    arr_referenc4 =  np.arange(len(data4))
    
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.scatter(arr_referenc,data, marker='^', color='#00429d', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc2,data2, marker='s', color='#00cc00', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc3,data3, marker='o', color='#ff8000', linewidth=.7, s=5, facecolors='none')
    ax.scatter(arr_referenc4,data4, marker='x', color='#ff0000', linewidth=.7, s=5)
    ax.legend(legend, loc='upper right', prop={'size': 9})
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('T [frames]')
    plt.ylabel('$C_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def vgroup_p40(data1, error1, data2, error2, data3, error3, data4, error4, data5, error5, title, legend):
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.errorbar(30, data1, yerr=error1, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(50,data2, yerr=error2, marker='s', color='#00cc00', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(70,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(90,data4, yerr=error4, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(100,data5, yerr=error5, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='upper left', prop={'size': 9})
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$C_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def v_group_5(theory, data1, error1, data2, error2, data3, error3, data4, error4, data5, error5, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    arr_referenc =  np.arange(10, 105, 5)
    ax.plot(arr_referenc , theory*1000, c= "red", linestyle=':', linewidth=.9)
    ax.errorbar(30, data1, yerr=error1, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(40,data2, yerr=error2, marker='s', color='#00cc00', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(60,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(80,data4, yerr=error4, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(100,data5, yerr=error5, marker='o', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.08, 0.60, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$v_{group}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def v_group_5_30pa(theory, data1, error1, data2, error2, data3, error3, data4, error4, data5, error5, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    arr_referenc =  np.arange(10, 105, 5)
    ax.plot(arr_referenc , theory*1000, c= "red", linestyle=':', linewidth=.9)
    ax.errorbar(40, data1, yerr=error1, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(50,data2, yerr=error2, marker='s', color='#00cc00', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(60,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(80,data4, yerr=error4, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(100,data5, yerr=error5, marker='o', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.08, 0.60, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$v_{group}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def c_daw_4(theory, data1, error1, data3, error3, data4, error4, data5, error5, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    arr_referenc =  np.arange(10, 105, 5)
    ax.plot(arr_referenc , theory, c= "red", linestyle=':', linewidth=.9)
    ax.errorbar(40, data1, yerr=error1, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(60,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(80,data4, yerr=error4, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(100,data5, yerr=error5, marker='o', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.08, 0.60, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$C_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()

def lambda_daw_5(theory, data1, error1, data2, error2, data3, error3, data4, error4, data5, error5, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    #arr_referenc =  np.arange(10, 105, 5)
    #ax.plot(arr_referenc , theory, c= "red", linestyle=':', linewidth=.9)
    ax.errorbar(30, data1, yerr=error1, fmt='^', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(40, data2, yerr=error2, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(60,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(80,data4, yerr=error4, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(100,data5, yerr=error5, marker='o', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='upper left', prop={'size': 9}, bbox_to_anchor=(0.08, 0.90, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$\lambda_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=25)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()

def lambda_daw_5_2(theory, data1, error1, data2, error2, data3, error3, data4, error4, data5, error5, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    #arr_referenc =  np.arange(10, 105, 5)
    #ax.plot(arr_referenc , theory, c= "red", linestyle=':', linewidth=.9)
    ax.errorbar(40, data1, yerr=error1, fmt='^', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(50, data2, yerr=error2, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(60,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(80,data4, yerr=error4, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(100,data5, yerr=error5, marker='o', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='lower right', prop={'size': 9}, bbox_to_anchor=(0.08, 0.10, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$\lambda_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=25)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()

def lambda_daw_4(data1, error1, data2, error2, data3, error3, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    #arr_referenc =  np.arange(10, 105, 5)
    #ax.plot(arr_referenc , theory, c= "red", linestyle=':', linewidth=.9)
    #ax.errorbar(15, data, yerr=error, fmt='s', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(20, data1, yerr=error1, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(25,data2, yerr=error2, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(30,data3, yerr=error3, marker='x', color='#000000', linewidth=.9, markersize=3, capsize=1)
    #ax.legend(legend, loc='upper left', prop={'size': 9}, bbox_to_anchor=(0.08, 0.90, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('Pressure')
    plt.ylabel('$\lambda_{DAW}$ [mm]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(.5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=15)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()
   
def c_daw_4_30pa(theory, data1, error1, data3, error3, data4, error4, data5, error5, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    arr_referenc =  np.arange(10, 105, 5)
    ax.plot(arr_referenc , theory, c= "red", linestyle=':', linewidth=.9)
    ax.errorbar(40, data1, yerr=error1, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(60,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(80,data4, yerr=error4, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.errorbar(100,data5, yerr=error5, marker='o', color='#000000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.08, 0.60, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$C_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()

def c_daw_3(theory, data1, error1, data3, error3, data4, error4, title, legend):
  
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    arr_referenc =  np.arange(10, 105, 5)
    ax.plot(arr_referenc , theory, c= "red", linestyle=':', linewidth=.9)
    ax.errorbar(40, data1, yerr=error1, fmt='^', color='#00429d', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(60,data3, yerr=error3, marker='o', color='#ff8000', linewidth=.9, markersize=3, capsize=1, mfc='w')
    ax.errorbar(80,data4, yerr=error4, marker='x', color='#ff0000', linewidth=.9, markersize=3, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.08, 0.60, .90, .01))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('E-field [%]')
    plt.ylabel('$C_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()   

def bigploterror_3(data15, error15, data20, error20, data25, error25, data30, error30, data30_pfc, error30_pfc, data40, error40, title, legend):
    #arr_referenc =  np.arange(len(data))
    #arr_referenc2 =  np.arange(len(data2))
    #arr_referenc3 =  np.arange(len(data3))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    
    ax.errorbar(15, data15, yerr=error15, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(20, data20, yerr=error20, fmt='s', color='#00cc00', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(25, data25, yerr=error25, fmt='s', color='#00cc00', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(30, data30, yerr=error30, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(30, data30_pfc, yerr=error30_pfc, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(40, data40, yerr=error40, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9})#, bbox_to_anchor=(0.04, 0.85, .25, .102))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('Run')
    plt.ylabel('$v_{group}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show()
    
def bigploterror_3_cdaw(data, error, data2, error2, data3, error3, title, legend):
    #arr_referenc =  np.arange(len(data))
    #arr_referenc2 =  np.arange(len(data2))
    #arr_referenc3 =  np.arange(len(data3))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    
    ax.errorbar(1, data, yerr=error, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(2, data2, yerr=error2, fmt='s', color='#00cc00', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(3, data3, yerr=error3, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.04, 0.85, .25, .102))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('Run')
    plt.ylabel('$C_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show()    

def bigploterror_4_cdaw(data, error, data2, error2, data3, error3, data4, error4, title, legend):
    #arr_referenc =  np.arange(len(data))
    #arr_referenc2 =  np.arange(len(data2))
    #arr_referenc3 =  np.arange(len(data3))
    #arr_referenc4 =  np.arange(len(data4))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    
    ax.errorbar(1, data, yerr=error, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(2, data2, yerr=error2, fmt='s', color='#00cc00', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(3, data3, yerr=error3, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(4, data4, yerr=error4, fmt='x',color='#ff0000', markersize=2, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.04, 0.45, .95, .102))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('Run')
    plt.ylabel('$C_{daw}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show()    

def bigploterror_4(data, error, data2, error2, data3, error3, data4, error4, title, legend):
    #arr_referenc =  np.arange(len(data))
    #arr_referenc2 =  np.arange(len(data2))
    #arr_referenc3 =  np.arange(len(data3))
    #arr_referenc4 =  np.arange(len(data4))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    
    ax.errorbar(1, data, yerr=error, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(2, data2, yerr=error2, fmt='s', color='#00cc00', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(3, data3, yerr=error3, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(4, data4, yerr=error4, fmt='x',color='#ff0000', markersize=2, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.04, 0.85, .85, .102))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('Run')
    plt.ylabel('$v_{group}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show()    

def bigploterror_6_3ov3(data, error, data2, error2, data3, error3, data4, error4, data5, error5, data6, error6, title, legend):
    #arr_referenc =  np.arange(len(data))
    #arr_referenc2 =  np.arange(len(data2))
    #arr_referenc3 =  np.arange(len(data3))
    #arr_referenc4 =  np.arange(len(data4))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    
    ax.errorbar(1, data, yerr=error, fmt='^', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(1, data2, yerr=error2, fmt='x', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(2, data3, yerr=error3, fmt='^', color='#00cc00', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(2, data4, yerr=error4, fmt='x', color='#00cc00', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(3, data5, yerr=error5, fmt='^',color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(3, data6, yerr=error6, fmt='x',color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.04, 0.85, .85, .102))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('Run')
    plt.ylabel('$v_{group}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show() 

def bigploterror_12(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, data, error, data2, error2, data3, error3, data4, error4, data5, error5, data6, error6, data7, error7, data8, error8, data9, error9, data10, error10, data11, error11, data12, error12, xlabel, legend):
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.errorbar(x1, data, yerr=error, fmt='x', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x2, data2, yerr=error2, fmt='x', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x3, data3, yerr=error3, fmt='x', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x4, data4, yerr=error4, fmt='x', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x5, data5, yerr=error5, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x6, data6, yerr=error6, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x7, data7, yerr=error7, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x8, data8, yerr=error8, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x9, data9, yerr=error9, fmt='^', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x10, data10, yerr=error10, fmt='^', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x11, data11, yerr=error11, fmt='^', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x12, data12, yerr=error12, fmt='^', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.04, 0.85, .85, .102))
    
    #adds a title and axes labels
    #ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel('$v_{group}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show() 
    
def bigploterror_12_2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, data, error, data2, error2, data3, error3, data4, error4, data5, error5, data6, error6, data7, error7, data8, error8, data9, error9, data10, error10, data11, error11, data12, error12, xlabel, legend):
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.errorbar(x1, data, yerr=error, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x2, data2, yerr=error2, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x3, data3, yerr=error3, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x4, data4, yerr=error4, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x5, data5, yerr=error5, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x6, data6, yerr=error6, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x7, data7, yerr=error7, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x8, data8, yerr=error8, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x9, data9, yerr=error9, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x10, data10, yerr=error10, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x11, data11, yerr=error11, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x12, data12, yerr=error12, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.04, 0.85, .85, .102))
    
    #adds a title and axes labels
    #ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel('$c_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show()
    
def bigploterror_12_3(x1, dx1, x2, dx2, x3, dx3, x4, dx4, x5, dx5, x6, dx6, x7, dx7, x8, dx8, x9, dx9, x10, dx10, x11, dx11, x12, dx12, data, error, data2, error2, data3, error3, data4, error4, data5, error5, data6, error6, data7, error7, data8, error8, data9, error9, data10, error10, data11, error11, data12, error12, xlabel, legend):
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.errorbar(x1, data, yerr=error, xerr=dx1, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x2, data2, yerr=error2, xerr=dx2, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x3, data3, yerr=error3, xerr=dx3, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x4, data4, yerr=error4, xerr=dx4, fmt='x', color='#000000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x5, data5, yerr=error5, xerr=dx5, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x6, data6, yerr=error6, xerr=dx6, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x7, data7, yerr=error7, xerr=dx7, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x8, data8, yerr=error8, xerr=dx8, fmt='s', color='#00429d', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x9, data9, yerr=error9, xerr=dx9, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x10, data10, yerr=error10, xerr=dx10, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x11, data11, yerr=error11, xerr=dx11, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.errorbar(x12, data12, yerr=error12, xerr=dx12, fmt='^', color='#ff8000', markersize=3, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.04, 0.85, .85, .102))
    
    #adds a title and axes labels
    #ax.set_title(title)
    plt.xlabel(xlabel)
    plt.ylabel('$c_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show()
#%%  
def cdaw_bigploterror_6_3ov3_theory(data, error, theory, z, title, legend):
    #arr_referenc =  np.arange(len(data))
    #arr_referenc2 =  np.arange(len(data2))
    #arr_referenc3 =  np.arange(len(data3))
    #arr_referenc4 =  np.arange(len(data4))
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    #
    p = [15, 20, 25, 30, 40]
    #
    ax.errorbar(p, data, yerr=error, fmt='x', color='#48A2F1', markersize=2, linewidth=1, capsize=2)   
    ax.scatter(p, theory, marker='o', color='#000000', linewidth=1, s=30, facecolors='none')#, facecolors='none'
    #
    ax.legend(legend, loc='upper right')
    #
    #adds a title and axes labels
    #ax.set_title(title)
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('$C_{Daw}$ [mm/s]')
    # Create a second y-axis sharing the same x-axis
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(p, z, color='#9A0CCA', linewidth=.5, linestyle='dashed')
    #
    ax2.legend(["Charge potential"], bbox_to_anchor=(0.15, 0.65, .85, 0.1))#, prop={'size': 9})#, bbox_to_anchor=(0.08, 0.85, .85, .102))
    #
    plt.ylabel('z')
    #
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #adds major gridlines
    #ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.2', color='gray')
    ax.grid(which='minor', linestyle='--', linewidth='0.1', color='gray')
    
    #ax2.set_ylim(ymax=.4)
    ax.set_ylim(ymax=80)
    
    plt.show() 
    
def group_bigploterror_6_3ov3_theory(v_group, v_group_pfc, theory, havnes, title, legend):
    
    fig, ax = plt.subplots(dpi=500)
    fig.set_size_inches(6, 3)
    #
    p = [15, 20, 25, 30, 40]
    p2 = [20, 25, 30]
    #
    ax.errorbar(p2, v_group[0], yerr=v_group[1], fmt='x', color='#48A2F1', markersize=2, linewidth=1, capsize=2) 
    ax.errorbar(p, v_group_pfc[0], yerr=v_group_pfc[1], fmt='^', color='#D81B1B', markersize=2, linewidth=1, capsize=2)        
    ax.scatter(p, theory, marker='o', color='#000000', linewidth=1, s=30, facecolors='none')#, facecolors='none'
    #
    ax.legend(['Theory','ISS Data','Pfc Data'])
    #
    plt.ylabel('$v_{group}$ [mm/s]')
    #
    plt.xlabel('Pressure [Pa]')
    #
    # Create a second y-axis sharing the same x-axis
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(p, havnes, color='#9A0CCA', linewidth=.5, linestyle='dashed')
    #
    ax2.legend(["Havnes-Parameter"], bbox_to_anchor=(0.15, 0.6, .85, 0.1))#, prop={'size': 9})#, bbox_to_anchor=(0.08, 0.85, .85, .102))
    #
    plt.ylabel('P')
    #
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #adds major gridlines
    #ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.2', color='gray')
    ax.grid(which='minor', linestyle='--', linewidth='0.1', color='gray')
    
    # Create combined legend
    #lines_labels = [data, theory, plot]
    #labels = [line.get_label() for line in lines_labels]
    #ax2.legend(lines_labels, labels, loc='upper right')
    
    ax.set_title(title)
    #ax limit
    ax.set_ylim(ymin=30, ymax=100)
    ax2.set_ylim(ymax=2.632)
    
    plt.show() 

def group_efieldploterror_6_3ov3_theory(ef15, data15, error15, theory15, ef20, data, error, theory20, ef25, data2, error2, theory25, ef30, data3, error3, theory30, ef40, data4, error4, theory40, havnes, title, legend):
    
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    #
    x_arr_referenc =  np.divide([abs(ef15)/15, abs(ef20)/20, abs(ef25)/25, abs(ef30)/30, abs(ef40)/40],100)
    #
    ax.errorbar(x_arr_referenc[0], data15, yerr=error15, fmt='x', color='#ff8000', markersize=2, linewidth=1, capsize=2)
    ax.errorbar(x_arr_referenc[1], data, yerr=error, fmt='x', color='#ff8000', markersize=2, linewidth=1, capsize=2)        
    ax.errorbar(x_arr_referenc[2], data2, yerr=error2, fmt='x', color='#ff8000', markersize=2, linewidth=1, capsize=2)    
    ax.errorbar(x_arr_referenc[3], data3, yerr=error3, fmt='x',color='#ff8000', markersize=2, linewidth=1, capsize=2)
    ax.errorbar(x_arr_referenc[4], data4, yerr=error4, fmt='x',color='#ff8000', markersize=2, linewidth=1, capsize=2)
    #
    ax.scatter(x_arr_referenc[0], theory15, marker='o' , color='#000000', linewidth=1, s=30, facecolors='none')#, facecolors='none'
    ax.scatter(x_arr_referenc[1], theory20, marker='o' , color='#000000', linewidth=1, s=30, facecolors='none')
    ax.scatter(x_arr_referenc[2], theory25, marker='o', color='#000000', linewidth=1, s=30, facecolors='none')
    ax.scatter(x_arr_referenc[3], theory30, marker='o', color='#000000', linewidth=1, s=30, facecolors='none')
    ax.scatter(x_arr_referenc[4], theory40, marker='o', color='#000000', linewidth=1, s=30, facecolors='none')
    #
    #ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.08, 0.85, .85, .102))
    #adds a title and axes labels
    #
    plt.ylabel('$v_{group}$ [mm/s]')
    #
    plt.xlabel('E/p [V/cm/Pa]')
    #
    # Create a second y-axis sharing the same x-axis
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(x_arr_referenc, havnes, color='#00429d', linewidth=.5, linestyle='dashed')
    #
    #ax.legend(['Data','Theory'])
    ax2.legend(["Harvnes-Parameter"], loc='upper left')#, prop={'size': 9})#, bbox_to_anchor=(0.08, 0.85, .85, .102))
    #
    plt.ylabel('P')
    #
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #adds major gridlines
    #ax.minorticks_on()
    ax.grid(which='major', linestyle='--', linewidth='0.2', color='gray')
    ax.grid(which='minor', linestyle='--', linewidth='0.1', color='gray')
    
    # Create combined legend
    #lines_labels = [data, theory, plot]
    #labels = [line.get_label() for line in lines_labels]
    #ax2.legend(lines_labels, labels, loc='upper right')
    
    ax.set_title(title)
    #ax limit
    ax.set_ylim(ymin=30, ymax=100)
    #ax2.set_ylim(ymax=2.55)
    
    plt.show()     

def bigploterror_6(data, error, data2, error2, data3, error3, data4, error4, data5, error5, data6, error6, title, legend):
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.errorbar(1, data, yerr=error, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(2, data2, yerr=error2, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(3, data3, yerr=error3, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(4, data4, yerr=error4, fmt='o',color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(5, data5, yerr=error3, fmt='x', color='#ff0000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(6, data6, yerr=error4, fmt='x',color='#ff0000', markersize=2, linewidth=1, capsize=1)
    ax.legend(legend, loc='upper right', prop={'size': 9}, bbox_to_anchor=(0.05, 0.5, .9, .1))
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('Run')
    plt.ylabel('$v_{group}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    plt.show() 
    
def bigplot_speed(data, data2, data3, data4, error_15, error_20, error_25, error_30):
    arr_referenc =  np.arange(len(data))
    arr_referenc2 =  np.arange(len(data2))
    arr_referenc3 =  np.arange(len(data3))
    arr_referenc4 =  np.arange(len(data4))
    fig, ax = plt.subplots(dpi=1200)
    fig.set_size_inches(6, 6)
    #x_peak = find_peaks(data,height=2.4)
    #print(x_peak[0][0])
    #fig.savefig('test2png.png', dpi=100)
    #color pattern: '#00429d' blue, '#b03a2e' red, '#1e8449' green
    ax.errorbar(arr_referenc, data, yerr=error_15, fmt='^', color='#00429d', markersize=2, linewidth=1, capsize=1)                          # , linewidth=1.25)
    ax.errorbar(arr_referenc2, data2, yerr=error_20, fmt='s', color='#00cc00', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(arr_referenc3, data3, yerr=error_25, fmt='o', color='#ff8000', markersize=2, linewidth=1, capsize=1)
    ax.errorbar(arr_referenc4, data4, yerr=error_30, fmt='x',color='#ff0000', markersize=2, linewidth=1, capsize=1)
    ax.legend(['15 Pa','20 Pa', '25 Pa', '30 Pa'], loc='upper right', prop={'size': 9})
        
    #adds a title and axes labels
    #ax.set_title('')
    plt.xlabel('T [s]')
    plt.ylabel('$C_{DAW}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(25))
    ax.yaxis.set_major_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axhline(63.76, linestyle='--', color='#00429d', linewidth=1);
    #ax.axhline(60.94, linestyle='-.', color='#00cc00', linewidth=1);
    #ax.axhline(60.2, linestyle=':', color='#ff8000', linewidth=1);
    #ax.axhline(58.4, linestyle=':', color='#ff0000', linewidth=1);

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=0)
    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '0.0'
    labels[1] = '0.25'
    labels[2] = '0.5'
    labels[3] = '0.75'
    labels[4] = '1.0'
    labels[5] = '1.25'

    ax.set_xticklabels(labels)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def allplot_noerror(data, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, title, legend):
    
    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    #DC
    #20
    ax.scatter(20,data, marker='o', color='#000000', linewidth=1, s=8, facecolors='none') #30
    ax.scatter(20,data2, marker='s', color='#8f8f8f', linewidth=1, s=8, facecolors='none') #40
    ax.scatter(20,data3, marker='^', color='#e8b31a', linewidth=1, s=8, facecolors='none') #60
    ax.scatter(20,data4, marker='s', color='#df8c10', linewidth=1, s=8) #80
    ax.scatter(20,data5, marker='x', color='#d4660a', linewidth=1, s=8) #100
    #25
    ax.scatter(25,data6, marker='o', color='#000000', linewidth=1, s=8, facecolors='none')
    ax.scatter(25,data7, marker='s', color='#8f8f8f', linewidth=1, s=8, facecolors='none') #40
    ax.scatter(25,data8, marker='^', color='#e8b31a', linewidth=1, s=8, facecolors='none') #60
    ax.scatter(25,data9, marker='s', color='#df8c10', linewidth=1, s=8) #80
    ax.scatter(25,data10, marker='x', color='#d4660a', linewidth=1, s=8) #100
    #30
    ax.scatter(30,data11, marker='s', color='#8f8f8f', linewidth=1, s=8, facecolors='none') #40
    ax.scatter(30,data12, marker='^', color='#020080', linewidth=1, s=8, facecolors='none') #50
    ax.scatter(30,data13, marker='^', color='#e8b31a', linewidth=1, s=8, facecolors='none') #60
    ax.scatter(30,data14, marker='s', color='#df8c10', linewidth=1, s=8) #80
    ax.scatter(30,data15, marker='x', color='#d4660a', linewidth=1, s=8) #100
    #
    #ax.legend(legend, loc='upper right', prop={'size': 9})
    
    #adds a title and axes labels
    ax.set_title(title)
    plt.xlabel('P [Pa]')
    plt.ylabel('$v_{group}$ [mm/s]')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    ax.set_xlim(xmin=15, xmax=35)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def dispersion_relation(k_20, k_25, k_30, k_35, k_40, w_20, w_25, w_30, w_35, w_40):

    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.scatter(w_20, k_20, marker='^', color='#00429d', linewidth=2, s=20, facecolors='none')
    ax.scatter(w_25, k_25, marker='o', color='#8f8f8f', linewidth=2, s=20, facecolors='none')
    ax.scatter(w_30, k_30, marker='x', color='#ff8000', linewidth=2, s=20)
    ax.scatter(w_35, k_35, marker='s', color='#000000', linewidth=2, s=20)
    #ax.scatter(w_40, k_40, marker='s', color='#000000', linewidth=2, s=20)
    ax.legend(['100','40', '60', '80', '30'], loc='lower right', prop={'size': 9})

    #adds a title and axes labels
    ax.set_title('Dispersion Relation')
    plt.ylabel('$\omega$/$\omega_{pd}$')
    plt.xlabel('k$\lambda_{D}$')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def dispersion_relation_all(w_20, w_25, w_30, w_15, w_40, k_20, k_25, k_30, k_15, k_40):

    fig, ax = plt.subplots(dpi=600)
    fig.set_size_inches(6, 3)
    
    ax.scatter(w_15, k_15, marker='s', color='#D81B1B', linewidth=2, s=20, facecolors='none')
    ax.scatter(w_20, k_20, marker='^', color='#48A2F1', linewidth=2, s=20, facecolors='none')
    ax.scatter(w_25, k_25, marker='o', color='#FFC107', linewidth=2, s=20, facecolors='none')
    ax.scatter(w_30, k_30, marker='x', color='#004D40', linewidth=2, s=20)
    ax.scatter(w_40, k_40, marker='d', color='#9A0CCA', linewidth=2, s=20, facecolors='none')
    ax.legend(['15 Pa', '20 Pa','25 Pa', '30 Pa', '40 Pa'], loc='lower right', prop={'size': 9})

    #adds a title and axes labels
    ax.set_title('Dispersion Relation')
    plt.ylabel('$\omega$/$\omega_{pd}$')
    plt.xlabel('k$\lambda_{D}$')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

def charge_plot(Z_d, Z_d_0, z_error, z_external, k):

    fig, ax = plt.subplots(dpi=800)
    fig.set_size_inches(6, 3)
    
    #D81B1B, 48A2F1, FFC107, 004D40, 9A0CCA
    #
    ax.errorbar(k, Z_d , yerr=z_error, fmt='d',color='#D81B1B', markersize=3, linewidth=1, capsize=3)
    #
    ax.errorbar(k, Z_d_0 , yerr=z_error, fmt='^',color='#48A2F1', markersize=3, linewidth=1, capsize=3)
    #
    #external data
    #
    ax.scatter(z_external[0][0], z_external[0][1], marker='o', color='#FFC107', linewidth=1.5, s=30, facecolors='none')
    ax.scatter(z_external[1][0], z_external[1][1], marker='s', color='#004D40', linewidth=1.5, s=30, facecolors='none')
    ax.scatter(z_external[2][0][:6], z_external[2][1][:6], marker='x', color='#9A0CCA', linewidth=1.5, s=30)
    ax.scatter(z_external[3][0], z_external[3][1], marker='h', color='#000000', linewidth=1.5, s=30, facecolors='none')
    #
    ax.legend(['Fortov2004', 'Khrapak2003', 'Khrapak2005', 'Yaroshenko2004', 'Exp. z depleted', 'Exp. z'], loc='upper right', prop={'size': 7.5})
    #
    #adds a title and axes labels
    ax.set_title('Charge evolution')
    plt.ylabel('$z = Ze^2/aT_e$')
    plt.xlabel('$\lambda / l_i$')
 
    #removing top and right borders
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False) 
    
    #Edit tick 
    #ax.xaxis.set_minor_locator(MultipleLocator(10))
    #ax.yaxis.set_minor_locator(MultipleLocator(1))

    #add vertical lines
    #ax.axvline(left, linestyle='dashed', color='b');
    #ax.axvline(right, linestyle='dashed', color='b');

    #adds major gridlines
    ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
    
    #ax limit
    #ax.set_xlim(xmin=0)
    ax.set_ylim(ymin = 0, ymax = 1.15)
    
    #legend
    #ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)
    
    plt.show()    

#%% Cloud speed DC100 LOAD

#   MAIN   #

###################
### Cloud speed ###
###################

trials = 20

cloud_speed_20pa_103901 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_103901_20pa_headspeed.json'))
cloud_speed_20pa_103947 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_103947_20pa_headspeed.json'))
cloud_speed_20pa_104625 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104625_20pa_headspeed.json'))
cloud_speed_20pa_104809 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104809_20pa_headspeed.json'))
cloud_speed_25pa_110808 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_110808_25pa_headspeed.json'))
cloud_speed_25pa_111329 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_111329_25pa_headspeed.json'))
cloud_speed_30pa_105732 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_105732_30pa_headspeed.json'))
cloud_speed_30pa_110231 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_110231_30pa_headspeed.json'))

#PFC42 Data:  VM2-AVI-240606-090245_30pa_0p5mA_tr100_headspeed
cloud_speed_15pa = json.load(open('resultspfc42/Parabola#19-15p-t100_headspeed.json'))
excel_data_df = pandas.read_excel('resultspfc42/Book-20pa-trial100-head.xlsx')
cloud_speed_20pa = excel_data_df['headspeed'].tolist()
excel_data_df = pandas.read_excel('resultspfc42/Book-25pa-trial100-head.xlsx')
cloud_speed_25pa = excel_data_df['headspeed'].tolist()
cloud_speed_30pa = json.load(open('resultspfc42/VM2-AVI-240606-090245_30pa_0p5mA_tr100_headspeed.json'))
cloud_speed_40pa = json.load(open('resultspfc42/Parabola#0-40pa-t100_headspeed.json'))


# compare group velocities #
data_15 = np.average(cloud_speed_15pa['velocities'][:trials])     #This converts to numpy
error_15 = np.average(cloud_speed_15pa['errors'][:trials]) + np.std(cloud_speed_15pa['velocities'][:trials])/np.sqrt(trials)

data_20 = np.average(cloud_speed_20pa_103901['velocities'][:trials])     #This converts to numpy
error_20 = np.average(cloud_speed_20pa_103901['errors'][:trials]) + np.std(cloud_speed_20pa_103901['velocities'][:trials])/np.sqrt(trials)   #This converts to numpy
data_20_2 = np.average(cloud_speed_20pa_103947['velocities'][:trials])     #This converts to numpy
error_20_2 = np.average(cloud_speed_20pa_103947['errors'][:trials]) + np.std(cloud_speed_20pa_103947['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy
data_20_3 = np.average(cloud_speed_20pa_104625['velocities'][:trials])     #This converts to numpy
error_20_3 = np.average(cloud_speed_20pa_104625['errors'][:trials]) + np.std(cloud_speed_20pa_104625['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy
data_20_4 = np.average(cloud_speed_20pa_104809['velocities'][:trials])     #This converts to numpy
error_20_4 = np.average(cloud_speed_20pa_104809['errors'][:trials]) + np.std(cloud_speed_20pa_104809['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy

#bigploterror_4(data_20, error_20, data_20_2, error_20_2, data_20_3, error_20_3, data_20_4, error_20_4, 'Compare group velocities 20pa', ['20 Pa 10:39:01', '20 Pa 10:39:47', '20 Pa 10:46:25', '20 Pa 10:48:09'])

data_25 = np.average(cloud_speed_25pa_110808['velocities'][:trials])     #This converts to numpy
error_25 = np.average(cloud_speed_25pa_110808['errors'][:trials]) + np.std(cloud_speed_25pa_110808['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy
data_25_2 = np.average(cloud_speed_25pa_111329['velocities'][:trials])     #This converts to numpy
error_25_2 = np.average(cloud_speed_25pa_111329['errors'][:trials]) + np.std(cloud_speed_25pa_111329['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy
data_30 = np.average(cloud_speed_30pa_105732['velocities'][:trials])     #This converts to numpy
error_30 = np.average(cloud_speed_30pa_105732['errors'][:trials]) + np.std(cloud_speed_30pa_105732['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy
data_30_2 = np.average(cloud_speed_30pa_110231['velocities'][:trials])     #This converts to numpy
error_30_2 = np.average(cloud_speed_30pa_110231['errors'][:trials]) + np.std(cloud_speed_30pa_110231['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy

#pfc42
data_20_pfc = np.average(cloud_speed_20pa)     #This converts to numpy
error_20_pfc = np.std(cloud_speed_20pa)/np.sqrt(len(cloud_speed_20pa))
data_25_pfc = np.average(cloud_speed_25pa)     #This converts to numpy
error_25_pfc = np.std(cloud_speed_25pa)/np.sqrt(len(cloud_speed_25pa))
data_30_pfc = np.average(cloud_speed_30pa['velocities'][:trials])     #This converts to numpy
error_30_pfc = np.average(cloud_speed_30pa['errors'][:trials]) + np.std(cloud_speed_30pa['velocities'][:trials])/np.sqrt(trials)
data_40 = np.average(cloud_speed_40pa['velocities'][:trials])     #This converts to numpy
error_40 = np.average(cloud_speed_40pa['errors'][:trials]) + np.std(cloud_speed_40pa['velocities'][:trials])/np.sqrt(trials)      #This converts to numpy

#bigploterror_4(data_25, error_25, data_25_2, error_25_2, data_30, error_30, data_30_2, error_30_2, 'Compare group velocities 25-30pa', ['25 Pa 11:08:08', '25 Pa 11:13:29', '30 Pa 10:57:32', '30 Pa 11:02:31'])

data_15_average = data_15
error_15_average = error_15

data_20_average = np.average([data_20, data_20_2, data_20_4])
error_20_average = np.std([data_20, data_20_2]) + np.average([error_20, error_20_2, error_20_4])/np.sqrt(len([error_20, error_20_2, error_20_4]))

data_25_average = np.average([data_25, data_25_2])
error_25_average = np.std([error_25, error_25_2]) + np.average([error_25, error_25_2])/np.sqrt(len([error_25, error_25_2]))

data_30_average = np.average([data_30, data_30_2])
error_30_average = np.std([error_30, error_30_2]) + np.average([error_30, error_30_2])/np.sqrt(len([error_30, error_30_2]))

data_40_average = data_40
error_40_average = error_40

bigploterror_3(data_15_average, error_15_average, data_20_average, error_20_average, data_25_average, error_25_average, data_30_average, error_30_average, data_30_pfc, error_30_pfc, data_40_average, error_40_average, ' ',['20Pa', '25Pa', '30Pa'])

#%% PLOT

# Import Math #
obj_text_f = codecs.open('resultsC17/theory/theo_ dustspeed_neutralandiondrag_dc100_z_adjusted_depleted.txt', 'r', encoding='utf-8').read()
parameters = json.load(open('resultsC17/parameters/system-parameter-C15-230125.json'))
theory = np.array(json.loads(obj_text_f)) #This reads json to list
theo_plot = theory[:,0]
z = theory[:,1]
#
havnes = [parameters["15pa"]["havnes"], parameters["20pa"]["havnes"], parameters["25pa"]["havnes"], parameters["30pa"]["havnes"], parameters["40pa"]["havnes"]]
ef = [parameters["15pa"]["e-field-vm"], parameters["20pa"]["e-field-vm"], parameters["25pa"]["e-field-vm"], parameters["30pa"]["e-field-vm"], parameters["40pa"]["e-field-vm"]]
Zd_a = [parameters["15pa"]["Z_d"], parameters["20pa"]["Z_d"], parameters["25pa"]["Z_d"], parameters["30pa"]["Z_d"], parameters["40pa"]["Z_d"]]
#
v_g_100_pfc = [[data_15_average, data_20_pfc, data_25_pfc, data_30_pfc, data_40_average],[error_15_average, error_20_pfc, error_25_pfc, error_30_pfc, error_40_average]]
v_g_100 = [[data_20_average, data_25_average, data_30_average], [error_20_average, error_25_average, error_30_average]]
#
group_bigploterror_6_3ov3_theory(v_g_100, v_g_100_pfc, theo_plot*1000, havnes,' ', ['20pa theory z='+str(.225), '25pa theory z='+str(.226), '30pa theory z=' +str(.229),'20pa exp-F', '25pa exp-F', '30pa exp-F'])
#group_efieldploterror_6_3ov3_theory(ef[0], data_15_average, error_15_average, theo_plot[0]*1000, ef[1], data_20_average, error_20_average, theo_plot[1]*1000, ef[2], data_25_average, error_25_average, theo_plot[2]*1000, ef[3], data_30_average, error_30_average, theo_plot[3]*1000, ef[4], data_40_average, error_40_average, theo_plot[4]*1000, havnes,' ', ['20pa theory z='+str(.225), '25pa theory z='+str(.226), '30pa theory z=' +str(.229),'20pa exp-F', '25pa exp-F', '30pa exp-F'])


#%% Phase speeds DC100 - LOAD

##########################
### Phase speeds DC100 ###
##########################

# load Parabola#19-15pa_speedlist_forward PFC #

obj_text_f = codecs.open('resultspfc42/phase-speeds/Parabola#19-15pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
f15 = json.loads(obj_text_f) #This reads json to list
speedlist_15pa_forward = -np.array(f15) + np.average(cloud_speed_15pa['velocities'][:trials])

# load VM2_AVI_230125_103901_20pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_103901_20pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_103901_20pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
b20 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_103901_20pa_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_103901['velocities'][:trials])     #This converts to numpy
VM2_AVI_230125_103901_20pa_speedlist_backward = -np.array(b20) + np.average(cloud_speed_20pa_103901['velocities'][:trials])     #This converts to numpy

# load VM2_AVI_230125_103947_20pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_103947_20pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_103947_20pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
b20 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_103947_20pa_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_103947['velocities'][:trials])     #This converts to numpy
VM2_AVI_230125_103947_20pa_speedlist_backward = -np.array(b20) + np.average(cloud_speed_20pa_103947['velocities'][:trials])     #This converts to numpy

# load VM2_AVI_230125_104625_20pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_104625_20pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_104625_20pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
b20 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_104625_20pa_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_104625['velocities'][:trials])     #This converts to numpy
VM2_AVI_230125_104625_20pa_speedlist_backward = -np.array(b20) + np.average(cloud_speed_20pa_104625['velocities'][:trials])     #This converts to numpy

# load VM2_AVI_230125_104625_20pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_104809_20pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_104809_20pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
b20 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_104809_20pa_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_104809['velocities'][:trials])     #This converts to numpy
VM2_AVI_230125_104809_20pa_speedlist_backward = -np.array(b20) + np.average(cloud_speed_20pa_104809['velocities'][:trials])     #This converts to numpy

# load VM2_AVI_230125_110808_25pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_110808_25pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_110808_25pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f25 = json.loads(obj_text_f) #This reads json to list
b25 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_110808_25pa_speedlist_forward = -np.array(f25) + np.average(cloud_speed_25pa_110808['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_110808_25pa_speedlist_backward = np.array(b25) + np.average(cloud_speed_25pa_110808['velocities'][:trials])      #This converts to numpy

# load VM2_AVI_230125_111329_25pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_111329_25pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_111329_25pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f25 = json.loads(obj_text_f) #This reads json to list
b25 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_111329_25pa_speedlist_forward = -np.array(f25) + np.average(cloud_speed_25pa_111329['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_111329_25pa_speedlist_backward = np.array(b25) + np.average(cloud_speed_25pa_111329['velocities'][:trials])      #This converts to numpy

# load VM2_AVI_230125_105732_30pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_105732_30pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_105732_30pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f30 = json.loads(obj_text_f) #This reads json to list
b30 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_105732_30pa_speedlist_forward = -np.array(f30) + np.average(cloud_speed_30pa_105732['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_105732_30pa_speedlist_backward = np.array(b30) + np.average(cloud_speed_30pa_105732['velocities'][:trials])      #This converts to numpy

# load VM2_AVI_230125_110231_30pa #

obj_text_f = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_110231_30pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/VM2_AVI_230125_110231_30pa_speedlist_backward.txt', 'r', encoding='utf-8').read()
f30 = json.loads(obj_text_f) #This reads json to list
b30 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_110231_30pa_speedlist_forward = -np.array(f30) + np.average(cloud_speed_30pa_110231['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_110231_30pa_speedlist_backward = np.array(b30) + np.average(cloud_speed_30pa_110231['velocities'][:trials])      #This converts to numpy

# load Parabola#0-40pa_speedlist_forward PFC #

obj_text_f = codecs.open('resultspfc42/phase-speeds/Parabola#0-40pa_speedlist_forward.txt', 'r', encoding='utf-8').read()
f40 = json.loads(obj_text_f) #This reads json to list
speedlist_40pa_forward = -np.array(f40[10:]) + np.average(cloud_speed_40pa['velocities'][:trials])

#%% PLOT

#bigplot_4(VM2_AVI_230125_103901_20pa_speedlist_forward, VM2_AVI_230125_103947_20pa_speedlist_forward, VM2_AVI_230125_104625_20pa_speedlist_forward, VM2_AVI_230125_104809_20pa_speedlist_forward, 'Forward', ['20 Pa 10:39:01', '20 Pa 10:39:47', '20 Pa 10:46:25', '20 Pa 10:48:09'])
#bigplot_4(VM2_AVI_230125_103901_20pa_speedlist_backward, VM2_AVI_230125_103947_20pa_speedlist_backward, VM2_AVI_230125_104625_20pa_speedlist_backward, VM2_AVI_230125_104809_20pa_speedlist_backward, 'Backward', ['20 Pa 10:39:01', '20 Pa 10:39:47', '20 Pa 10:46:25', '20 Pa 10:48:09'])

#bigplot_4(VM2_AVI_230125_110808_25pa_speedlist_forward, VM2_AVI_230125_111329_25pa_speedlist_forward, VM2_AVI_230125_110808_25pa_speedlist_backward, VM2_AVI_230125_111329_25pa_speedlist_backward, ' ', ['25 Pa 11:13:29 F', '25 Pa 11:08:08 F', '25 Pa 11:13:29 B', '25 Pa 11:08:08 B'])

#bigplot_4(VM2_AVI_230125_105732_30pa_speedlist_forward, VM2_AVI_230125_110231_30pa_speedlist_forward, VM2_AVI_230125_105732_30pa_speedlist_backward, VM2_AVI_230125_110231_30pa_speedlist_backward, ' ', ['30 Pa 10:57:32 F', '30 Pa 11:02:31 F', '30 Pa 10:57:32 B', '30 Pa 11:02:31 B'])

#title_f = 't1 = 2ms, t3 = 2ms; I1 = 0.5mA, I3 = -0.5mA'
#title_b = 't1 = 0ms, t3 = 2ms; I1 = 0.5mA, I3 = -0.5mA'
#bigplot_3(VM2_AVI_230125_103901_20pa_speedlist_forward, VM2_AVI_230125_111329_25pa_speedlist_forward, VM2_AVI_230125_110231_30pa_speedlist_forward, title_f)
#bigplot_3(VM2_AVI_230125_103901_20pa_speedlist_backward, VM2_AVI_230125_111329_25pa_speedlist_backward, VM2_AVI_230125_110231_30pa_speedlist_backward, title_b)


# Calculate Avergae and Error, Compare Plot 1DC #
a = speedlist_15pa_forward
speedlist_average_15pa_forward = np.average(a)
error_speedlist_average_15pa_forward = np.std(a)/np.sqrt(len(a)) + error_15
a = [np.average(VM2_AVI_230125_103901_20pa_speedlist_forward[:len(VM2_AVI_230125_103901_20pa_speedlist_forward)]), np.average(VM2_AVI_230125_103947_20pa_speedlist_forward[:len(VM2_AVI_230125_103901_20pa_speedlist_forward)]), np.average(VM2_AVI_230125_104625_20pa_speedlist_forward[:len(VM2_AVI_230125_103901_20pa_speedlist_forward)]), np.average(VM2_AVI_230125_104809_20pa_speedlist_forward[:len(VM2_AVI_230125_103901_20pa_speedlist_forward)])]
speedlist_average_20pa_forward = np.average(a)
error_speedlist_average_20pa_forward = np.std(a)/np.sqrt(len(a))
a = [np.average(VM2_AVI_230125_103901_20pa_speedlist_backward[:150]),np.average(VM2_AVI_230125_103947_20pa_speedlist_backward[:150]), np.average(VM2_AVI_230125_104809_20pa_speedlist_backward[:150])]
speedlist_average_20pa_backward = np.average(a)
error_speedlist_average_20pa_backward = np.std(a)/np.sqrt(len(a))
a = [np.average(VM2_AVI_230125_110808_25pa_speedlist_forward[:len(VM2_AVI_230125_110808_25pa_speedlist_forward)]),np.average(VM2_AVI_230125_111329_25pa_speedlist_forward[:len(VM2_AVI_230125_110808_25pa_speedlist_forward)])]
speedlist_average_25pa_forward = np.average(a)
error_speedlist_average_25pa_forward = np.std(a)/np.sqrt(len(a))+ error_25_average
a = [np.average(VM2_AVI_230125_110808_25pa_speedlist_backward[:100]),np.average(VM2_AVI_230125_111329_25pa_speedlist_backward[:100])]
speedlist_average_25pa_backward = np.average(a)
error_speedlist_average_25pa_backward = np.std(a)/np.sqrt(len(a))
a = [np.average(VM2_AVI_230125_105732_30pa_speedlist_forward[:len(VM2_AVI_230125_110231_30pa_speedlist_forward)]),np.average(VM2_AVI_230125_110231_30pa_speedlist_forward[:len(VM2_AVI_230125_110231_30pa_speedlist_forward)])]
speedlist_average_30pa_forward = np.average(a)
error_speedlist_average_30pa_forward = np.std(a) + error_30_average
a = [np.average(VM2_AVI_230125_105732_30pa_speedlist_backward[:100]),np.average(VM2_AVI_230125_110231_30pa_speedlist_backward[:100])]
speedlist_average_30pa_backward = np.average(a) 
error_speedlist_average_30pa_backward = np.std(a)/np.sqrt(len(a)) + error_20
a = speedlist_15pa_forward
speedlist_average_15pa_forward = np.average(a)
error_speedlist_average_15pa_forward = np.std(a)/np.sqrt(len(a)) + error_15
a = speedlist_40pa_forward
speedlist_average_40pa_forward = np.average(a)
error_speedlist_average_40pa_forward = np.std(a)/np.sqrt(len(a)) + error_40

#bigploterror_6_3ov3(speedlist_average_20pa_forward, error_speedlist_average_20pa_forward, speedlist_average_20pa_backward, error_speedlist_average_20pa_backward, speedlist_average_25pa_forward, error_speedlist_average_25pa_forward, speedlist_average_25pa_backward, error_speedlist_average_25pa_backward, speedlist_average_30pa_forward, error_speedlist_average_30pa_forward, speedlist_average_30pa_backward, error_speedlist_average_25pa_backward, ' ', ['20pa F', '20pa B', '25pa F', '25pa B', '30pa F', '30pa B'])

# Import Math #
obj_text_f = codecs.open('resultsC17/theory/theo_cdaw_dc100_z0.236_depl.txt', 'r', encoding='utf-8').read()
theory = np.array(json.loads(obj_text_f)) #This reads json to list
theo_plot = theory[:,0]
z = theory[:,1]
#
data = [speedlist_average_15pa_forward, speedlist_average_20pa_forward, speedlist_average_25pa_forward, speedlist_average_30pa_forward, speedlist_average_40pa_forward]
error = [error_speedlist_average_15pa_forward, error_speedlist_average_20pa_forward, error_speedlist_average_25pa_forward, error_speedlist_average_30pa_forward, error_speedlist_average_40pa_forward]

cdaw_bigploterror_6_3ov3_theory(data, error, theo_plot, z, ' ', ['from Equation (x)', 'Exp. data'])

#%% Group speeds DCXXX - LOAD

trials = 20

##########################
### Group speeds DCXXX ###
##########################

#cloud_speed_20pa_t13 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104518_20pa_t13_headspeed.json'))   #No wave appearing
cloud_speed_20pa_t13_2 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104901_20pa_t13_headspeed.json'))
cloud_speed_20pa_t14 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104331_20pa_t14_headspeed.json'))
cloud_speed_20pa_t16 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104154_20pa_t16_headspeed.json'))
cloud_speed_20pa_t16_2 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104227_20pa_t16_headspeed.json'))
cloud_speed_20pa_t18 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_104101_20pa_t18_headspeed.json'))
cloud_speed_25pa_t13 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_111238_25pa_t13_headspeed.json'))
cloud_speed_25pa_t14 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_111156_25pa_t14_headspeed.json'))
cloud_speed_25pa_t16 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_111106_25pa_t16_headspeed.json'))
cloud_speed_25pa_t16_2 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_111106_25pa_t16_headspeed.json'))
#cloud_speed_25pa_t18 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_110918_25pa_t18_headspeed.json'))
cloud_speed_25pa_t18_2 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_110948_25pa_t18_headspeed.json'))
cloud_speed_30pa_t14 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_110058_30pa_t14_headspeed.json'))
cloud_speed_30pa_t15 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_110133_30pa_t15_headspeed.json'))
cloud_speed_30pa_t16 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_110007_30pa_t16_headspeed.json'))
cloud_speed_30pa_t18 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_105845_30pa_t18_headspeed.json'))
cloud_speed_30pa_t18_2 = json.load(open('resultsC17/group-vel/VM2_AVI_230125_105926_30pa_t18_headspeed.json'))

#PFC42 VM2-AVI-240606-090844_30pa_0p5mA_tr20_headspeed
cloud_speed_30pa_t12 = json.load(open('resultspfc42/VM2-AVI-240606-090844_30pa_0p5mA_tr20_headspeed.json'))
cloud_speed_30pa_t13_2 = json.load(open('resultspfc42/Parabola#8-30pa-t30_headspeed.json'))
cloud_speed_30pa_t13 = json.load(open('resultspfc42/VM2-AVI-240606-085944_40pa_0p5mA_tr30_headspeed.json')) 
cloud_speed_30pa_t135 = json.load(open('resultspfc42/VM2-AVI-240606-090844_30pa_0p5mA_tr35_headspeed.json')) 
cloud_speed_30pa_t15_2 = json.load(open('resultspfc42/VM2-AVI-240606-090545_30pa_0p5mA_tr50_headspeed.json')) 
cloud_speed_30pa_t17 = json.load(open('resultspfc42/VM2-AVI-240606-090245_30pa_0p5mA_tr70_headspeed.json')) 
cloud_speed_30pa_t19 = json.load(open('resultspfc42/VM2-AVI-240606-090245_30pa_0p5mA_tr90_headspeed.json')) 
cloud_speed_40pa_t13 = json.load(open('resultspfc42/Parabola#0-40pa-t70_headspeed.json'))
cloud_speed_40pa_t15 = json.load(open('resultspfc42/VM2-AVI-240606-085944_40pa_0p5mA_tr30_headspeed.json'))
cloud_speed_40pa_t17 = json.load(open('resultspfc42/Parabola#0-40pa-t70_headspeed.json'))  
cloud_speed_40pa_t19 = json.load(open('resultspfc42/Parabola#0-40pa-t90_headspeed.json')) 
    

# PLOT Cloud speeds #
data20_t13 = np.average(cloud_speed_20pa_t13_2['velocities'][:trials])#np.average([np.average(cloud_speed_20pa_t13['velocities'][:trials]), np.average(cloud_speed_20pa_t13_2['velocities'][:trials])])     #This converts to numpy
error20_t13 = np.average(cloud_speed_20pa_t13_2['errors'][:trials]) + np.std(cloud_speed_20pa_t13_2['velocities'][:trials])
data20_t14 = np.average(cloud_speed_20pa_t14['velocities'][:trials])     #This converts to numpy
error20_t14 = np.average(cloud_speed_20pa_t14['errors'][:trials]) + np.std(cloud_speed_20pa_t14['velocities'][:trials])     #This converts to numpy
data20_t16 = np.average([np.average(cloud_speed_20pa_t16['velocities'][:trials]), np.average(cloud_speed_20pa_t16_2['velocities'][:trials])])      #This converts to numpy
error20_t16 = np.average([np.average(cloud_speed_20pa_t16['errors'][:trials]) + np.std(cloud_speed_20pa_t16['velocities'][:trials]), np.average(cloud_speed_20pa_t16_2['errors'][:trials]) + np.std(cloud_speed_20pa_t16_2['velocities'][:trials])]) + np.std([np.average(cloud_speed_20pa_t16['errors'][:trials]) + np.std(cloud_speed_20pa_t16['velocities'][:trials]), np.average(cloud_speed_20pa_t16_2['errors'][:trials]) + np.std(cloud_speed_20pa_t16_2['velocities'][:trials])])   #This converts to numpy
data20_t18 = np.average(cloud_speed_20pa_t18['velocities'][:trials])     #This converts to numpy
error20_t18 = np.average(cloud_speed_20pa_t18['errors'][:trials]) + np.std(cloud_speed_20pa_t18['velocities'][:trials])     #This converts to numpy
#
data25_t13 = np.average(cloud_speed_25pa_t13['velocities'][:trials])     #This converts to numpy
error25_t13 = np.average(cloud_speed_25pa_t13['errors'][:trials]) + np.std(cloud_speed_25pa_t13['velocities'][:trials])     #This converts to numpy
data25_t14 = np.average(cloud_speed_25pa_t14['velocities'][:trials])     #This converts to numpy
error25_t14 = np.average(cloud_speed_25pa_t14['errors'][:trials]) + np.std(cloud_speed_25pa_t14['velocities'][:trials])     #This converts to numpy
data25_t16 = np.average([np.average(cloud_speed_25pa_t16['velocities'][:trials]), np.average(cloud_speed_25pa_t16_2['velocities'][:trials])])     #This converts to numpy
error25_t16 = np.average([np.average(cloud_speed_25pa_t16['errors'][:trials]) + np.std(cloud_speed_25pa_t16['velocities'][:trials]), np.average(cloud_speed_25pa_t16_2['errors'][:trials]) + np.std(cloud_speed_25pa_t16_2['velocities'][:trials])]) + np.std([np.average(cloud_speed_25pa_t16['errors'][:trials]) + np.std(cloud_speed_25pa_t16['velocities'][:trials]), np.average(cloud_speed_25pa_t16_2['errors'][:trials]) + np.std(cloud_speed_25pa_t16_2['velocities'][:trials])])     #This converts to numpy
data25_t18 = np.average(cloud_speed_25pa_t18_2['velocities'][:trials])# , np.average(cloud_speed_25pa_t18_2['velocities'][:trials])])     #This converts to numpy
error25_t18 = np.average(cloud_speed_25pa_t18_2['errors'][:trials] + np.std(cloud_speed_25pa_t18_2['velocities'][:trials])) #, np.average(cloud_speed_25pa_t18_2['errors'][:trials]) + np.std(cloud_speed_25pa_t18_2['velocities'][:trials])]) + np.std([np.average(cloud_speed_25pa_t18['errors'][:trials]) + np.std(cloud_speed_25pa_t18['velocities'][:trials]), np.average(cloud_speed_25pa_t18_2['errors'][:trials]) + np.std(cloud_speed_25pa_t18_2['velocities'][:trials])])     #This converts to numpy
#
data30_t14 = np.average(cloud_speed_30pa_t14['velocities'][:trials])     #This converts to numpy
error30_t14 = np.average(cloud_speed_30pa_t14['errors'][:trials]) + np.std(cloud_speed_30pa_t14['velocities'][:trials])     #This converts to numpy
data30_t15 = np.average(cloud_speed_30pa_t15['velocities'][:trials])     #This converts to numpy
error30_t15 = np.average(cloud_speed_30pa_t15['errors'][:trials]) + np.std(cloud_speed_30pa_t15['velocities'][:trials])     #This converts to numpy
data30_t16 = np.average(cloud_speed_30pa_t16['velocities'][:trials])     #This converts to numpy
error30_t16 = np.average(cloud_speed_30pa_t16['errors'][:trials]) + np.std(cloud_speed_30pa_t16['velocities'][:trials])     #This converts to numpy
data30_t18 = np.average([np.average(cloud_speed_30pa_t18['velocities'][:trials]), np.average(cloud_speed_30pa_t18_2['velocities'][:trials])])     #This converts to numpy
error30_t18 = np.average([np.average(cloud_speed_30pa_t18['errors'][:trials]) + np.std(cloud_speed_30pa_t18['velocities'][:trials]), np.average(cloud_speed_30pa_t18_2['errors'][:trials]) + np.std(cloud_speed_30pa_t18_2['velocities'][:trials])])    #This converts to numpy
#PFC42
#
data30_t12_pfc = np.average(cloud_speed_30pa_t12['velocities'][:trials])    #This converts to numpy
error30_t12_pfc = np.average(cloud_speed_30pa_t12['errors'][:trials]) + np.std(cloud_speed_30pa_t12['velocities'][:trials])      #This converts to numpy
data30_t13_pfc = np.average(cloud_speed_30pa_t13_2['velocities'][:trials])    #This converts to numpy
error30_t13_pfc = np.average(cloud_speed_30pa_t13_2['errors'][:trials]) + np.std(cloud_speed_30pa_t13_2['velocities'][:trials])      #This converts to numpy
data30_t135_pfc = np.average(cloud_speed_30pa_t135['velocities'][:trials])    #This converts to numpy
error30_t135_pfc = np.average(cloud_speed_30pa_t135['errors'][:trials]) + np.std(cloud_speed_30pa_t135['velocities'][:trials])      #This converts to numpy
data30_t15_pfc = np.average(cloud_speed_30pa_t15_2['velocities'][:trials])    #This converts to numpy
error30_t15_pfc = np.average(cloud_speed_30pa_t15_2['errors'][:trials]) + np.std(cloud_speed_30pa_t15_2['velocities'][:trials])      #This converts to numpy
data30_t17_pfc = np.average(cloud_speed_30pa_t17['velocities'][:trials])    #This converts to numpy
error30_t17_pfc = np.average(cloud_speed_30pa_t17['errors'][:trials]) + np.std(cloud_speed_30pa_t17['velocities'][:trials])      #This converts to numpy
data30_t19_pfc = np.average(cloud_speed_30pa_t19['velocities'][:trials])    #This converts to numpy
error30_t19_pfc = np.average(cloud_speed_30pa_t19['errors'][:trials]) + np.std(cloud_speed_30pa_t19['velocities'][:trials])      #This converts to numpy

#
data40_t13 = np.average(cloud_speed_40pa_t13['velocities'][:trials])     #This converts to numpy
error40_t13 = np.average(cloud_speed_40pa_t13['errors'][:trials]) + np.std(cloud_speed_40pa_t13['velocities'][:trials])     #This converts to numpy
data40_t15 = np.average(cloud_speed_40pa_t15['velocities'][:trials])
error40_t15 = np.average(cloud_speed_40pa_t15['errors'][:trials]) + np.std(cloud_speed_40pa_t15['velocities'][:trials])     #This converts to numpy
data40_t17 = np.average(cloud_speed_40pa_t17['velocities'][:trials])     #This converts to numpy
error40_t17 = np.average(cloud_speed_40pa_t17['errors'][:trials]) + np.std(cloud_speed_40pa_t17['velocities'][:trials])
data40_t19 = np.average(cloud_speed_40pa_t19['velocities'][:trials])     #This converts to numpy
error40_t19 = np.average(cloud_speed_40pa_t19['errors'][:trials]) + np.std(cloud_speed_40pa_t19['velocities'][:trials])     #This converts to numpy

bigploterror_6(data30_t12_pfc, error30_t12_pfc, abs(data30_t13_pfc), error30_t13_pfc, data30_t135_pfc, error30_t135_pfc, abs(data30_t15_pfc), error30_t15_pfc, data30_t17_pfc, error30_t17_pfc, data30_t19_pfc, error30_t19_pfc, 'Compare group velocities 25-30pa RDC', ['20 Pa t13', '20 Pa t14', '25 Pa t13', '25 Pa t14', '30 Pa t14', '30 Pa t15'])
#bigploterror_6(data20_t13, error20_t13, data20_t14, error20_t14, data25_t13, error25_t13, data25_t14, error25_t14, data30_t14, error30_t14, data30_t15, error30_t15, 'Compare group velocities 25-30pa RDC', ['20 Pa t13', '20 Pa t14', '25 Pa t13', '25 Pa t14', '30 Pa t14', '30 Pa t15'])
#bigploterror_6_3ov3(data20_t14, error20_t14, data_20_average, error_20_average, data25_t14, error25_t14, data_25_average, error_25_average, data30_t14, error30_t14, data_30_average, error_30_average, 't_1 = 1.4 ms', ['20 Pa', '20 Pa 100DC', '25 Pa', '25 Pa 100DC', '30 Pa', '30 Pa 100DC'])
#allplot_noerror(data20_t13, data20_t14, data20_t16, data20_t18, data_20_average, data25_t13, data25_t14, data25_t16, data25_t18, data_25_average, data30_t14, data30_t15, data30_t16, data30_t18, data_30_average, '', [''])#data_20_average

#%% PLOT
# Import Math #
theory_cloud_speed_20pa = np.array(json.load(open('resultsC17/theory/theo_v_group_20pa_ef-reduce.txt')))
theory_cloud_speed_25pa = np.array(json.load(open('resultsC17/theory/theo_v_group_25pa_ef-reduce.txt')))
theory_cloud_speed_30pa = np.array(json.load(open('resultsC17/theory/theo_v_group_30pa_ef-reduce.txt')))
theory_cloud_speed_40pa = np.array(json.load(open('resultsC17/theory/theo_v_group_30pa_ef-reduce.txt')))
# PLOT
v_group_5(theory_cloud_speed_20pa, data20_t13, error20_t13, data20_t14, error20_t14, data20_t16, error20_t16, data20_t18, error20_t18, data_20_average, error_20_average , '', ['Theory 20pa', 'E30%', 'E40%', 'E60%', 'E80%', 'E100%'])
v_group_5(theory_cloud_speed_25pa, data25_t13, error25_t13, data25_t14, error25_t14, data25_t16, error20_t16, data25_t18, error25_t18, data_25_average, error_25_average, '', ['Theory 25pa', 'E30%', 'E40%', 'E60%', 'E80%', 'E100%'])
v_group_5_30pa(theory_cloud_speed_30pa, data30_t14, error30_t14, data30_t15, error30_t15, data30_t16, error30_t16, data30_t18, error30_t18, data_30_average, error_30_average, '', ['Theory 30pa', 'E40%', 'E50%', 'E60%', 'E80%', 'E100%'])
vgroup_p40(data40_t13, error40_t13, abs(data40_t15), error40_t15, data40_t17, error40_t17, data40_t19, error40_t19, data_40_average, error_40_average, '', ['E30%', 'E50%', 'E70%', 'E90%', 'E100%'])
#
cs_deviation_20 = 100.0 -(100.0/data_20_average)*data20_t14
cs_deviation_25 = 100.0 -(100.0/data_25_average)*data25_t14
cs_deviation_30 = 100.0 -(100.0/data_30_average)*data30_t14
#%% PIV Cloud speeds

########################
### PIV Cloud speeds ###
########################

obj_text = codecs.open('resultsC17/PIV/20pa_t13_frame_0472_0473.txt', 'r', encoding='utf-8').read()
a = np.array(json.loads(obj_text)) #This reads json to list
obj_text = codecs.open('resultsC17/PIV/20pa_t13_frame_0472_0473_R.txt', 'r', encoding='utf-8').read()
b = np.array(json.loads(obj_text)) #This reads json to list
pa20_t13_frame_0472_0473_weights = np.average(a, weights = b)      #This converts to numpy

obj_text = codecs.open('resultsC17/PIV/20pa_t13_frame_0473_0474.txt', 'r', encoding='utf-8').read()
a = np.array(json.loads(obj_text)) #This reads json to list
obj_text = codecs.open('resultsC17/PIV/20pa_t13_frame_0473_0474_R.txt', 'r', encoding='utf-8').read()
b = np.array(json.loads(obj_text)) #This reads json to list
pa20_t13_frame_0473_0474_weights = np.average(a, weights = b)      #This converts to numpy

obj_text = codecs.open('resultsC17/PIV/20pa_t13_head_frame_0316_0317.txt', 'r', encoding='utf-8').read()
a = np.array(json.loads(obj_text)) #This reads json to list
obj_text = codecs.open('resultsC17/PIV/20pa_t13_head_frame_0316_0317_R.txt', 'r', encoding='utf-8').read()
b = np.array(json.loads(obj_text)) #This reads json to list
pa20_t13_head_frame_0316_317_weights = np.average(a, weights = b)      #This converts to numpy

p20_t13_backward = (pa20_t13_frame_0472_0473_weights + pa20_t13_frame_0473_0474_weights) /2

p20_t13_deviation = 100.0 - (100.0/p20_t13_backward)*pa20_t13_head_frame_0316_317_weights

obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0949_0950.txt', 'r', encoding='utf-8').read()
a = np.array(json.loads(obj_text)) #This reads json to list
obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0949_0950_R.txt', 'r', encoding='utf-8').read()
b = np.array(json.loads(obj_text)) #This reads json to list
pa25_t13_frame_0949_0950_weights = np.average(a, weights = b)      #This converts to numpy

obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0950_0951.txt', 'r', encoding='utf-8').read()
a = np.array(json.loads(obj_text)) #This reads json to list
obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0950_0951_R.txt', 'r', encoding='utf-8').read()
b = np.array(json.loads(obj_text)) #This reads json to list
pa25_t13_frame_0950_0951_weights = np.average(a, weights = b)      #This converts to numpy

obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0775_0776.txt', 'r', encoding='utf-8').read()
a = np.array(json.loads(obj_text)) #This reads json to list
obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0775_0776_R.txt', 'r', encoding='utf-8').read()
b = np.array(json.loads(obj_text)) #This reads json to list
pa25_t13_frame_0775_0776_weights = np.average(a, weights = b)      #This converts to numpy

obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0776_0777.txt', 'r', encoding='utf-8').read()
a = np.array(json.loads(obj_text)) #This reads json to list
obj_text = codecs.open('resultsC17/PIV/VM2_AVI_230125_111238_25pa_t13_frame_0776_0777_R.txt', 'r', encoding='utf-8').read()
b = np.array(json.loads(obj_text)) #This reads json to list
pa25_t13_frame_0776_0777_weights = np.average(a, weights = b)      #This converts to numpy

pa25_t13_backward = (pa25_t13_frame_0949_0950_weights + pa25_t13_frame_0950_0951_weights) /2

pa25_t13_forward = (pa25_t13_frame_0775_0776_weights + pa25_t13_frame_0776_0777_weights) /2

p25_t13_deviation = 100.0 - (100.0/pa25_t13_backward)*pa25_t13_forward

#%% Phase speeds DCXXX

##########################
### Phase speeds DCXXX ###
##########################

# load VM2_AVI_230125_104901_20pa_t13 #
obj_text_b = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_104901_20pa_t13_speedlist_backward.txt', 'r', encoding='utf-8').read()
b20 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_104901_20pa_t13_speedlist_backward = np.array(b20) + np.average(cloud_speed_20pa_t13_2['velocities'][:trials])    #This converts to numpy
# load VM2_AVI_230125_104331_20pa_t14 #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_104331_20pa_t14_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_104331_20pa_t14_speedlist_backward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
b20 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_104331_20pa_t14_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_t14['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_104331_20pa_t14_speedlist_backward = np.array(b20) + np.average(cloud_speed_20pa_t14['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_104101_20pa_t18_speedlist_forward #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_104101_20pa_t18_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_104101_20pa_t18_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_t18['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_104331_20pa_t16 #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_104154_20pa_t16_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_104154_20pa_t16_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_t16['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_104227_20pa_t16 #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_104227_20pa_t16_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_104227_20pa_t16_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_t16_2['velocities'][:trials])      #This converts to numpy


# load VM2_AVI_230125_111238_25pa_t13 #
obj_text_b = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_111238_25pa_t13_speedlist_backward.txt', 'r', encoding='utf-8').read()
b25 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_111238_25pa_t13_speedlist_backward = -np.array(b25) + np.average(cloud_speed_25pa_t13['velocities'][:trials])     #This converts to numpy
# load VM2_AVI_230125_111238_25pa_t14 #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_111238_25pa_t14_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_111238_25pa_t14_speedlist_backward.txt', 'r', encoding='utf-8').read()
f25 = json.loads(obj_text_f) #This reads json to list
b25 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_111238_25pa_t14_speedlist_forward = -np.array(f25) + np.average(cloud_speed_25pa_t14['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_111238_25pa_t14_speedlist_backward = np.array(b25) + np.average(cloud_speed_25pa_t14['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_111106_25pa_t16_speedlist_forward #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_111106_25pa_t16_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_111106_25pa_t16_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_t16['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_110948_25pa_t18_speedlist_forward #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_110948_25pa_t18_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_110948_25pa_t18_speedlist_forward = -np.array(f20) + np.average(cloud_speed_20pa_t18['velocities'][:trials])      #This converts to numpy

# load VM2_AVI_230125_110058_30pa_t14 #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_110058_30pa_t14_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_110058_30pa_t14_speedlist_backward.txt', 'r', encoding='utf-8').read()
f30 = json.loads(obj_text_f) #This reads json to list
b30 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_110058_30pa_t14_speedlist_forward = -np.array(f30) + np.average(cloud_speed_30pa_t14['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_110058_30pa_t14_speedlist_backward = np.array(b30) + np.average(cloud_speed_30pa_t14['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_110133_30pa_t15 #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_110133_30pa_t15_speedlist_forward.txt', 'r', encoding='utf-8').read()
obj_text_b = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_110133_30pa_t15_speedlist_backward.txt', 'r', encoding='utf-8').read()
f30 = json.loads(obj_text_f) #This reads json to list
b30 = json.loads(obj_text_b) #This reads json to list
VM2_AVI_230125_110133_30pa_t15_speedlist_forward = -np.array(f30) + np.average(cloud_speed_30pa_t15['velocities'][:trials])      #This converts to numpy
VM2_AVI_230125_110133_30pa_t15_speedlist_backward = np.array(b30) + np.average(cloud_speed_30pa_t15['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_110007_30pa_t16_speedlist_forward #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_110007_30pa_t16_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_110007_30pa_t16_speedlist_forward = -np.array(f20) + np.average(cloud_speed_30pa_t16['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_105926_30pa_t18_speedlist_forward #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_105926_30pa_t18_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_105926_30pa_t18_speedlist_forward = -np.array(f20) + np.average(cloud_speed_30pa_t18_2['velocities'][:trials])      #This converts to numpy
# load VM2_AVI_230125_105845_30pa_t18_speedlist_forward #
obj_text_f = codecs.open('resultsC17/phase-speeds/RDC/VM2_AVI_230125_105845_30pa_t18_speedlist_forward.txt', 'r', encoding='utf-8').read()
f20 = json.loads(obj_text_f) #This reads json to list
VM2_AVI_230125_105845_30pa_t18_speedlist_forward = -np.array(f20) + np.average(cloud_speed_30pa_t18['velocities'][:trials])      #This converts to numpy


title_f = 't1 = 1.4ms, t3 = 2ms; I1 = 0.5mA, I3 = -0.5mA'
title_b = 't1 = 0ms, t3 = 1.4ms; I1 = 0.5mA, I3 = -0.5mA'

#bigplot_3(VM2_AVI_230125_104331_20pa_t14_speedlist_forward, VM2_AVI_230125_111238_25pa_t14_speedlist_forward, VM2_AVI_230125_110058_30pa_t14_speedlist_forward, title_f)
#bigplot_3(VM2_AVI_230125_104331_20pa_t14_speedlist_backward, VM2_AVI_230125_111238_25pa_t14_speedlist_backward, VM2_AVI_230125_110058_30pa_t14_speedlist_backward, title_b)

#plot_2025(VM2_AVI_230125_104901_20pa_t13_speedlist_backward, VM2_AVI_230125_111238_25pa_t13_speedlist_backward[:100], '20 Pa - 25Pa, t1 = 1.3ms')
#plot_30(VM2_AVI_230125_110133_30pa_t15_speedlist_forward, VM2_AVI_230125_110133_30pa_t15_speedlist_backward[:200], '30Pa, t1 = 1.5ms')


# Calculate and Compare Average DC t14 #
# 20 t14 #
speedlist_average_20pa_forward_t14 = np.average(VM2_AVI_230125_104331_20pa_t14_speedlist_forward)
error_speedlist_average_20pa_forward_t14 = np.std(VM2_AVI_230125_104331_20pa_t14_speedlist_forward) + error20_t14
speedlist_average_20pa_backward_t14 = np.average(VM2_AVI_230125_104331_20pa_t14_speedlist_backward[:150])
error_speedlist_average_20pa_backward_t14 = np.std(VM2_AVI_230125_104331_20pa_t14_speedlist_backward[:150])
# 20 t13 #
speedlist_average_20pa_backward_t13 = np.average(VM2_AVI_230125_104901_20pa_t13_speedlist_backward)
error_speedlist_average_20pa_backward_t13 = np.std(VM2_AVI_230125_104901_20pa_t13_speedlist_backward) + error20_t13
# 20 t18 #
speedlist_average_20pa_forward_t18 = np.average(VM2_AVI_230125_104101_20pa_t18_speedlist_forward)
error_speedlist_average_20pa_forward_t18 = np.std(VM2_AVI_230125_104101_20pa_t18_speedlist_forward) + error20_t18
# 20 t16 #
speedlist_average_20pa_forward_t16 = np.average([np.average(VM2_AVI_230125_104227_20pa_t16_speedlist_forward),np.average(VM2_AVI_230125_104154_20pa_t16_speedlist_forward)])
error_speedlist_average_20pa_forward_t16 = np.std(VM2_AVI_230125_104227_20pa_t16_speedlist_forward) + np.std(VM2_AVI_230125_104227_20pa_t16_speedlist_forward) + error20_t16

# 25 t14 #
speedlist_average_25pa_forward_t14 = np.average(VM2_AVI_230125_111238_25pa_t14_speedlist_forward[:50])
error_speedlist_average_25pa_forward_t14 = np.std(VM2_AVI_230125_111238_25pa_t14_speedlist_forward[:50]) + error25_t14
speedlist_average_25pa_backward_t14 = np.average(VM2_AVI_230125_111238_25pa_t14_speedlist_backward[:200])
error_speedlist_average_25pa_backward_t14 = np.std(VM2_AVI_230125_111238_25pa_t14_speedlist_backward[:200])
# 25 t16 #
speedlist_average_25pa_forward_t16 = np.average(VM2_AVI_230125_111106_25pa_t16_speedlist_forward)
error_speedlist_average_25pa_forward_t16 = np.std(VM2_AVI_230125_111106_25pa_t16_speedlist_forward) + error25_t16
# 25 t18 #
speedlist_average_25pa_forward_t18 = np.average(VM2_AVI_230125_110948_25pa_t18_speedlist_forward[:50])
error_speedlist_average_25pa_forward_t18 = np.std(VM2_AVI_230125_110948_25pa_t18_speedlist_forward[:50]) + error25_t18

# 30 t14 #
speedlist_average_30pa_forward_t14 = np.average(VM2_AVI_230125_110058_30pa_t14_speedlist_forward)
error_speedlist_average_30pa_forward_t14 = np.std(VM2_AVI_230125_110058_30pa_t14_speedlist_forward) + error30_t14
speedlist_average_30pa_backward_t14 = np.average(VM2_AVI_230125_110058_30pa_t14_speedlist_backward[:200])
error_speedlist_average_30pa_backward_t14 = np.std(VM2_AVI_230125_110058_30pa_t14_speedlist_backward[:200])
# 30 t18 #
speedlist_average_30pa_forward_t18 = np.average([np.average(VM2_AVI_230125_105845_30pa_t18_speedlist_forward), np.average(VM2_AVI_230125_105926_30pa_t18_speedlist_forward)])
error_speedlist_average_30pa_forward_t18 = np.average([np.std(VM2_AVI_230125_105845_30pa_t18_speedlist_forward), np.std(VM2_AVI_230125_105926_30pa_t18_speedlist_forward)]) + error30_t18
# 30 t16 #
speedlist_average_30pa_forward_t16 = np.average(VM2_AVI_230125_110007_30pa_t16_speedlist_forward)
error_speedlist_average_30pa_forward_t16 = np.std(VM2_AVI_230125_110007_30pa_t16_speedlist_forward) + error30_t16

#bigploterror_6_3ov3(speedlist_average_20pa_forward_t14, error_speedlist_average_20pa_forward_t14, speedlist_average_20pa_backward_t14, error_speedlist_average_20pa_backward_t14, speedlist_average_25pa_forward_t14, error_speedlist_average_25pa_forward_t14, speedlist_average_25pa_backward_t14, error_speedlist_average_25pa_backward_t14, speedlist_average_30pa_forward_t14, error_speedlist_average_30pa_forward_t14, speedlist_average_30pa_backward_t14, error_speedlist_average_30pa_backward_t14, 'Reduced Duty Cycle t1 = 1.4ms',  ['20pa F', '20pa B', '25pa F', '25pa B', '30pa F', '30pa B'])
#bigploterror_4_cdaw(speedlist_average_20pa_forward_t14, error_speedlist_average_20pa_forward_t14, speedlist_average_20pa_forward_t16, error_speedlist_average_20pa_forward_t16, speedlist_average_20pa_forward_t18, error_speedlist_average_20pa_forward_t18, speedlist_average_20pa_forward, error_speedlist_average_20pa_forward, '20pa', ['E-field 40%', 'E-field 60%', 'E-field 80%', 'E-field 100%'])
#allplot_noerror()

# theory compare 20 pa e-field drop
theory_physe_speed_20pa = np.array(json.load(open('resultsC17/theory/theo_c_daw_20pa_ef-reduce.txt')))
c_daw_4(theory_physe_speed_20pa, speedlist_average_20pa_forward_t14, error_speedlist_average_20pa_forward_t14, speedlist_average_20pa_forward_t16, error_speedlist_average_20pa_forward_t16, speedlist_average_20pa_forward_t18, error_speedlist_average_20pa_forward_t18, speedlist_average_20pa_forward, error_speedlist_average_20pa_forward, '20pa', ['Theory 20pa','E-field 40%', 'E-field 60%', 'E-field 80%', 'E-field 100%'])

# theory compare 25 pa e-field drop
theory_physe_speed_25pa = np.array(json.load(open('resultsC17/theory/theo_c_daw_25pa_ef-reduce.txt')))
c_daw_4(theory_physe_speed_25pa, speedlist_average_25pa_forward_t14, error_speedlist_average_25pa_forward_t14, speedlist_average_25pa_forward_t16, error_speedlist_average_25pa_forward_t16, speedlist_average_25pa_forward_t18, error_speedlist_average_25pa_forward_t18, speedlist_average_25pa_forward, error_speedlist_average_25pa_forward, '25pa', ['Theory 25pa','E-field 40%', 'E-field 60%', 'E-field 80%', 'E-field 100%'])

# theory compare 30 pa e-field drop
theory_physe_speed_30pa = np.array(json.load(open('resultsC17/theory/theo_c_daw_30pa_ef-reduce.txt')))
c_daw_4_30pa(theory_physe_speed_30pa, speedlist_average_30pa_forward_t14, error_speedlist_average_30pa_forward_t14, speedlist_average_30pa_forward_t16, error_speedlist_average_30pa_forward_t16, speedlist_average_30pa_forward_t18, error_speedlist_average_30pa_forward_t18, speedlist_average_30pa_forward, error_speedlist_average_30pa_forward, '30pa', ['Theory 30pa','E-field 40%', 'E-field 60%', 'E-field 80%', 'E-field 100%'])

#%% Dispersion Relation

###########################
### Dispersion Relation ###
###########################

# Open and Read JSON file with system data
f = open ('resultsC17/parameters/system-parameter-C15-230125.json', "r")
data_read = json.loads(f.read())
#
# Wave Length Read In 15 #
a = np.array(json.load(open('resultspfc42/wavelength/Parabola#19-15pa_wavelenlist_forward.txt')))
wavelen_15pa = np.average(a[a != 0])
wavelen_error_15pa = np.std(a[a != 0])/np.sqrt(len(a[a != 0]))
#
# Wave Length Read In 20 #
a = np.array(json.load(open('resultsC17/wavelength/VM2_AVI_230125_103901_20pa_wavelenlist_forward.txt')))
b = np.array(json.load(open('resultsC17/wavelength/VM2_AVI_230125_103947_20pa_wavelenlist_forward.txt')))
c = np.array(json.load(open('resultsC17/wavelength/VM2_AVI_230125_104625_20pa_wavelenlist_forward.txt')))
d = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_104101_20pa_t18_wavelenlist_forward.txt')))
e = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_104154_20pa_t16_wavelenlist_forward.txt')))
f = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_104227_20pa_t16_wavelenlist_forward.txt')))
g = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_104331_20pa_t14_wavelenlist_backward.txt')))
h = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_104331_20pa_t14_wavelenlist_forward.txt')))
i = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_104901_20pa_t13_wavelenlist_backward.txt')))
# 20
wavelen_20pa = np.average([np.average(a), np.average(b), np.average(c)]) #, np.average(np.array(json.load(open('wavelength/VM2_AVI_230125_104809_20pa_wavelenlist_forward.txt'))))]
wavelen_error_20pa = np.average([np.std(a)/np.sqrt(len(a)), np.std(b)/np.sqrt(len(b)), np.std(c)/np.sqrt(len(c))]) #, np.average(np.array(json.load(open('wavelength/VM2_AVI_230125_104809_20pa_wavelenlist_forward.txt'))))]
# 20t18
wavelen_20pa_t18 = np.average(d[d != 0])
wavelen_error_20pa_t18 = np.std(d[d != 0])/np.sqrt(len(d[d != 0]))
# 20t16
wavelen_20pa_t16 = np.average([np.average(e[e != 0]), np.average(f[f != 0])])
wavelen_error_20pa_t16 = np.average([np.std(e[e != 0])/np.sqrt(len(e[e != 0])), np.std(f[f != 0])/np.sqrt(len(f[f != 0]))])
# 20t14
wavelen_20pa_t14 = np.average([np.average(g[g != 0]), np.average(h[h != 0])])
wavelen_error_20pa_t14 = np.average([np.std(g[g != 0])/np.sqrt(len(g[g != 0])), np.std(h[h != 0])/np.sqrt(len(h[h != 0]))])
# 20t13
wavelen_20pa_t13 = np.average(i[i != 0])
wavelen_error_20pa_t13 = np.std(i[i != 0])/np.sqrt(len(i[i != 0]))
#
# Wave Length Read In 25 #
a = np.array(json.load(open('resultsC17/wavelength/VM2_AVI_230125_110808_25pa_wavelenlist_forward.txt')))
b = np.array(json.load(open('resultsC17/wavelength/VM2_AVI_230125_111329_25pa_wavelenlist_forward.txt')))
d = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_110948_25pa_t18_wavelenlist_forward.txt')))
e = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_111106_25pa_t16_wavelenlist_forward.txt')))
f = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_111238_25pa_t14_wavelenlist_forward.txt')))
g = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_111238_25pa_t13_wavelenlist_backward.txt'))) #backward
# 25
wavelen_25pa = np.average([np.average(a), np.average(b)])
wavelen_error_25pa = np.average([np.std(a)/np.sqrt(len(a)), np.std(b)/np.sqrt(len(b))]) #, np.average(np.array(json.load(open('wavelength/VM2_AVI_230125_104809_20pa_wavelenlist_forward.txt'))))]
# 25t18
wavelen_25pa_t18 = np.average(d[d != 0])
wavelen_error_25pa_t18 = np.std(d[d != 0])/np.sqrt(len(d[d != 0]))
# 25t16
wavelen_25pa_t16 = np.average(e[e != 0])
wavelen_error_25pa_t16 = np.std(e[e != 0])/np.sqrt(len(e[e != 0]))
# 25t14
wavelen_25pa_t14 = np.average(f[f != 0])
wavelen_error_25pa_t14 = np.std(f[f != 0])/np.sqrt(len(f[f != 0]))
# 25t13 Backward
wavelen_25pa_t13_b = np.average(g[g != 0])
wavelen_error_25pa_t13_b = np.std(g[g != 0])/np.sqrt(len(g[g != 0]))
#
# Wave Length Read In 30 #
a = np.array(json.load(open('resultsC17/wavelength/VM2_AVI_230125_105732_30pa_wavelenlist_forward.txt')))
b = np.array(json.load(open('resultsC17/wavelength/VM2_AVI_230125_110231_30pa_wavelenlist_forward.txt')))
d = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_105845_30pa_t18_wavelenlist_forward.txt')))
e = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_105926_30pa_t18_wavelenlist_forward.txt')))
f = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_110007_30pa_t16_wavelenlist_forward.txt')))
g = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_110058_30pa_t14_wavelenlist_forward.txt')))
h = np.array(json.load(open('resultsC17/wavelength/RDC/VM2_AVI_230125_110133_30pa_t15_wavelenlist_forward.txt')))
# 30
wavelen_30pa = np.average([np.average(a), np.average(b)])
wavelen_error_30pa = np.average([np.std(a)/np.sqrt(len(a)), np.std(b)/np.sqrt(len(b))]) #, np.average(np.array(json.load(open('wavelength/VM2_AVI_230125_104809_20pa_wavelenlist_forward.txt'))))]
# 30t18
wavelen_30pa_t18 = np.average([np.average(d[d != 0]), np.average(e[e != 0])])
wavelen_error_30pa_t18 = np.average([np.std(d[d != 0])/np.sqrt(len(d[d != 0])), np.std(e[e != 0])/np.sqrt(len(e[e != 0]))])
# 30t16
wavelen_30pa_t16 = np.average(f[f != 0])
wavelen_error_30pa_t16 = np.std(f[f != 0])/np.sqrt(len(f[f != 0]))
# 30t14
wavelen_30pa_t14 = np.average(g[g != 0])
wavelen_error_30pa_t14 = np.std(g[g != 0])/np.sqrt(len(g[g != 0]))
# 30t15
wavelen_30pa_t15 = np.average(h[h != 0])
wavelen_error_30pa_t15 = np.std(h[h != 0])/np.sqrt(len(h[h != 0]))
#
# Wave Length Read In 40 #
a = np.array(json.load(open('resultspfc42/wavelength/Parabola#0-40pa_wavelenlist_forward.txt')))
wavelen_40pa = np.average(a[a != 0])
wavelen_error_40pa = np.std(a[a != 0])/np.sqrt(len(a[a != 0]))

# Plot Wavelength
#All
lambda_daw_4(wavelen_20pa, wavelen_error_20pa, wavelen_25pa, wavelen_error_25pa, wavelen_30pa, wavelen_error_30pa, '', [])
#30
lambda_daw_5_2(0, wavelen_30pa_t14, wavelen_error_30pa_t14, wavelen_30pa_t15, wavelen_error_30pa_t15, wavelen_30pa_t16, wavelen_error_30pa_t16, wavelen_30pa_t18,  wavelen_error_30pa_t18, wavelen_30pa, wavelen_error_30pa, '30 Pa', ['E-field 40%', 'E-field 50%', 'E-field 60%', 'E-field 80%', 'E-field 100%'])
#25
lambda_daw_5(0, wavelen_25pa_t13_b, wavelen_error_25pa_t13_b, wavelen_25pa_t14, wavelen_error_25pa_t14, wavelen_25pa_t16, wavelen_error_25pa_t16, wavelen_25pa_t18,  wavelen_error_25pa_t18, wavelen_25pa, wavelen_error_25pa, '25 Pa', ['E-field 30% (b)', 'E-field 40%', 'E-field 60%', 'E-field 80%', 'E-field 100%'])
#20
lambda_daw_5(0, wavelen_20pa_t13, wavelen_error_20pa_t13, wavelen_20pa_t14, wavelen_error_20pa_t14, wavelen_20pa_t16, wavelen_error_20pa_t16, wavelen_20pa_t18,  wavelen_error_20pa_t18, wavelen_20pa, wavelen_error_20pa, '20 Pa', ['E-field 30%', 'E-field 40%', 'E-field 60%', 'E-field 80%', 'E-field 100%'])


# Calculate wave vector k #
# 15
k_15pa = (2*np.pi) / (wavelen_15pa*0.001) #in 1/m
k_15pa_error = ((2*np.pi) / (wavelen_15pa*0.001)**2) * wavelen_error_15pa*0.001
# 20
k_20pa = (2*np.pi) / (wavelen_20pa*0.001) #in 1/m
k_20pa_error = ((2*np.pi) / (wavelen_20pa*0.001)**2) * wavelen_error_20pa*0.001
k_20pa_t18 = (2*np.pi) / (wavelen_20pa_t18*0.001)
k_20pa_error_t18 = ((2*np.pi) / (wavelen_20pa_t18*0.001)**2) * wavelen_error_20pa_t18*0.001
k_20pa_t16 = (2*np.pi) / (wavelen_20pa_t16*0.001)
k_20pa_error_t16 = ((2*np.pi) / (wavelen_20pa_t16*0.001)**2) * wavelen_error_20pa_t16*0.001
k_20pa_t14 = (2*np.pi) / (wavelen_20pa_t14*0.001)
k_20pa_error_t14 = ((2*np.pi) / (wavelen_20pa_t14*0.001)**2) * wavelen_error_20pa_t14*0.001
k_20pa_t13 = (2*np.pi) / (wavelen_20pa_t13*0.001) #backward!
k_20pa_error_t13 = ((2*np.pi) / (wavelen_20pa_t13*0.001)**2) * wavelen_error_20pa_t13*0.001
# 25
k_25pa = (2*np.pi) / (wavelen_25pa*0.001)
k_25pa_error = ((2*np.pi) / (wavelen_25pa*0.001)**2) * wavelen_error_25pa*0.001
k_25pa_t18 = (2*np.pi) / (wavelen_25pa_t18*0.001)
k_25pa_error_t18 = ((2*np.pi) / (wavelen_25pa_t18*0.001)**2) * wavelen_error_25pa_t18*0.001
k_25pa_t16 = (2*np.pi) / (wavelen_25pa_t16*0.001)
k_25pa_error_t16 = ((2*np.pi) / (wavelen_25pa_t16*0.001)**2) * wavelen_error_25pa_t16*0.001
k_25pa_t14 = (2*np.pi) / (wavelen_25pa_t14*0.001)
k_25pa_error_t14 = ((2*np.pi) / (wavelen_25pa_t14*0.001)**2) * wavelen_error_25pa_t14*0.001
# 30
k_30pa = (2*np.pi) / (wavelen_30pa*0.001)
k_30pa_error = ((2*np.pi) / (wavelen_30pa*0.001)**2) * wavelen_error_30pa*0.001
k_30pa_t18 = (2*np.pi) / (wavelen_30pa_t18*0.001)
k_30pa_error_t18 = ((2*np.pi) / (wavelen_30pa_t18*0.001)**2) * wavelen_error_30pa_t18*0.001
k_30pa_t16 = (2*np.pi) / (wavelen_30pa_t16*0.001)
k_30pa_error_t16 = ((2*np.pi) / (wavelen_30pa_t16*0.001)**2) * wavelen_error_30pa_t16*0.001
k_30pa_t14 = (2*np.pi) / (wavelen_30pa_t14*0.001)
k_30pa_error_t14 = ((2*np.pi) / (wavelen_30pa_t14*0.001)**2) * wavelen_error_30pa_t14*0.001
# 40
k_40pa = (2*np.pi) / (wavelen_40pa*0.001) #in 1/m
k_40pa_error = ((2*np.pi) / (wavelen_40pa*0.001)**2) * wavelen_error_40pa*0.001

# Calculate Frequeny (omega) #
# 15
omega_15pa = (speedlist_average_15pa_forward*0.001) * k_15pa # w = C_daw * k 
# 20
omega_20pa = (speedlist_average_20pa_forward*0.001) * k_20pa # w = C_daw * k 
omega_20pa_t18 = (speedlist_average_20pa_forward_t18*0.001) * k_20pa_t18
omega_20pa_t16 = (speedlist_average_20pa_forward_t16*0.001) * k_20pa_t16
omega_20pa_t14 = (speedlist_average_20pa_forward_t14*0.001) * k_20pa_t14
# 25
omega_25pa = (speedlist_average_25pa_forward*0.001) * k_25pa
omega_25pa_t18 = (speedlist_average_25pa_forward_t18*0.001) * k_25pa_t18
omega_25pa_t16 = (speedlist_average_25pa_forward_t16*0.001) * k_25pa_t16
omega_25pa_t14 = (speedlist_average_25pa_forward_t14*0.001) * k_25pa_t14
# 30
omega_30pa = (speedlist_average_30pa_forward*0.001) * k_30pa
omega_30pa_t18 = (speedlist_average_30pa_forward_t18*0.001) * k_30pa_t18
omega_30pa_t16 = (speedlist_average_30pa_forward_t16*0.001) * k_30pa_t16
omega_30pa_t14 = (speedlist_average_30pa_forward_t14*0.001) * k_30pa_t14
# 40
omega_40pa = (speedlist_average_40pa_forward*0.001) * k_40pa # w = C_daw * k 
#
bigploterror_12(k_20pa, k_20pa_t18, k_20pa_t16, k_20pa_t14, k_25pa, k_25pa_t18, k_25pa_t16, k_25pa_t14, k_30pa, k_30pa_t18, k_30pa_t16, k_30pa_t14, data_20_average, error_20_average, data20_t18, error20_t18, data20_t16, error20_t16, data20_t14, error20_t14, data_25_average, error_25_average, data25_t18, error25_t18, data25_t16, error25_t16, data25_t14, error25_t14, data_30_average, error_30_average, data30_t18, error30_t18, data30_t16, error30_t16, data30_t14, error30_t14, 'k [1/m]', [])
bigploterror_12_3(k_20pa, k_20pa_error, k_20pa_t18, k_20pa_error_t18, k_20pa_t16, k_20pa_error_t16, k_20pa_t14, k_20pa_error_t14, k_25pa, k_25pa_error, k_25pa_t18, k_25pa_error_t18, k_25pa_t16, k_25pa_error_t16, k_25pa_t14, k_25pa_error_t14, k_30pa, k_30pa_error, k_30pa_t18, k_30pa_error_t18, k_30pa_t16, k_30pa_error_t16, k_30pa_t14, k_30pa_error_t14, speedlist_average_20pa_forward, error_speedlist_average_20pa_forward, speedlist_average_20pa_forward_t18, error_speedlist_average_20pa_forward_t18, speedlist_average_20pa_forward_t16, error_speedlist_average_20pa_forward_t16, speedlist_average_20pa_forward_t14, error_speedlist_average_20pa_forward_t14, speedlist_average_25pa_forward, error_speedlist_average_25pa_forward, speedlist_average_25pa_forward_t18, error_speedlist_average_25pa_forward_t18, speedlist_average_25pa_forward_t16, error_speedlist_average_25pa_forward_t16, speedlist_average_25pa_forward_t14, error_speedlist_average_25pa_forward_t14, speedlist_average_30pa_forward, error_speedlist_average_30pa_forward, speedlist_average_30pa_forward_t18, error_speedlist_average_30pa_forward_t18, speedlist_average_30pa_forward_t16, error_speedlist_average_30pa_forward_t16, speedlist_average_30pa_forward_t14, error_speedlist_average_30pa_forward_t14, 'k [1/m]', [])
#
bigploterror_12(omega_20pa, omega_20pa_t18, omega_20pa_t16, omega_20pa_t14, omega_25pa, omega_25pa_t18, omega_25pa_t16, omega_25pa_t14, omega_30pa, omega_30pa_t18, omega_30pa_t16, omega_30pa_t14, data_20_average, error_20_average, data20_t18, error20_t18, data20_t16, error20_t16, data20_t14, error20_t14, data_25_average, error_25_average, data25_t18, error25_t18, data25_t16, error25_t16, data25_t14, error25_t14, data_30_average, error_30_average, data30_t18, error30_t18, data30_t16, error30_t16, data30_t14, error30_t14, 'omega', [])
bigploterror_12_2(omega_20pa, omega_20pa_t18, omega_20pa_t16, omega_20pa_t14, omega_25pa, omega_25pa_t18, omega_25pa_t16, omega_25pa_t14, omega_30pa, omega_30pa_t18, omega_30pa_t16, omega_30pa_t14, speedlist_average_20pa_forward, error_speedlist_average_20pa_forward, speedlist_average_20pa_forward_t18, error_speedlist_average_20pa_forward_t18, speedlist_average_20pa_forward_t16, error_speedlist_average_20pa_forward_t16, speedlist_average_20pa_forward_t14, error_speedlist_average_20pa_forward_t14, speedlist_average_25pa_forward, error_speedlist_average_25pa_forward, speedlist_average_25pa_forward_t18, error_speedlist_average_25pa_forward_t18, speedlist_average_25pa_forward_t16, error_speedlist_average_25pa_forward_t16, speedlist_average_25pa_forward_t14, error_speedlist_average_25pa_forward_t14, speedlist_average_30pa_forward, error_speedlist_average_30pa_forward, speedlist_average_30pa_forward_t18, error_speedlist_average_30pa_forward_t18, speedlist_average_30pa_forward_t16, error_speedlist_average_30pa_forward_t16, speedlist_average_30pa_forward_t14, error_speedlist_average_30pa_forward_t14, 'omega', [])
#
#dispersion_relation(omega_20pa/data_read["20pa"]["w_pd"], omega_20pa_t18/data_read["20pa"]["w_pd"], omega_20pa_t16/data_read["20pa"]["w_pd"], omega_20pa_t14/data_read["20pa"]["w_pd"], 0, k_20pa*data_read["20pa"]["debye_Di"], k_20pa_t18*data_read["20pa"]["debye_Di"], k_20pa_t16*data_read["20pa"]["debye_Di"], k_20pa_t14*data_read["20pa"]["debye_Di"], 0)
#dispersion_relation(omega_25pa/data_read["25pa"]["w_pd"], omega_25pa_t18/data_read["25pa"]["w_pd"], omega_25pa_t16/data_read["25pa"]["w_pd"], omega_25pa_t14/data_read["25pa"]["w_pd"], 0, k_25pa*data_read["25pa"]["debye_Di"], k_25pa_t18*data_read["25pa"]["debye_Di"], k_25pa_t16*data_read["25pa"]["debye_Di"], k_25pa_t14*data_read["25pa"]["debye_Di"], 0)
#dispersion_relation(omega_30pa/data_read["30pa"]["w_pd"], omega_30pa_t18/data_read["30pa"]["w_pd"], omega_30pa_t16/data_read["30pa"]["w_pd"], omega_30pa_t14/data_read["30pa"]["w_pd"], 0, k_30pa*data_read["30pa"]["debye_Di"], k_30pa_t18*data_read["30pa"]["debye_Di"], k_30pa_t16*data_read["30pa"]["debye_Di"], k_30pa_t14*data_read["30pa"]["debye_Di"], 0)
#%% Charge PLOT
#prep
w_15 = [omega_15pa/data_read["15pa"]["w_pd"]]
w_20 = [omega_20pa/data_read["20pa"]["w_pd"], omega_20pa_t18/data_read["20pa"]["w_pd"], omega_20pa_t16/data_read["20pa"]["w_pd"], omega_20pa_t14/data_read["20pa"]["w_pd"]]
w_25 = [omega_25pa/data_read["25pa"]["w_pd"], omega_25pa_t18/data_read["25pa"]["w_pd"], omega_25pa_t16/data_read["25pa"]["w_pd"], omega_25pa_t14/data_read["25pa"]["w_pd"]]
w_30 = [omega_30pa/data_read["30pa"]["w_pd"], omega_30pa_t18/data_read["30pa"]["w_pd"], omega_30pa_t16/data_read["30pa"]["w_pd"], omega_30pa_t14/data_read["30pa"]["w_pd"]]
w_40 = [omega_40pa/data_read["40pa"]["w_pd"]]
#
k_15 = [k_15pa*data_read["15pa"]["debye_Di"]]
k_20 = [k_20pa*data_read["20pa"]["debye_Di"], k_20pa_t18*data_read["20pa"]["debye_Di"], k_20pa_t16*data_read["20pa"]["debye_Di"], k_20pa_t14*data_read["20pa"]["debye_Di"]]
k_25 = [k_25pa*data_read["25pa"]["debye_Di"], k_25pa_t18*data_read["25pa"]["debye_Di"], k_25pa_t16*data_read["25pa"]["debye_Di"], k_25pa_t14*data_read["25pa"]["debye_Di"]]
k_30 = [k_30pa*data_read["30pa"]["debye_Di"], k_30pa_t18*data_read["30pa"]["debye_Di"], k_30pa_t16*data_read["30pa"]["debye_Di"], k_30pa_t14*data_read["30pa"]["debye_Di"]]
k_40 = [k_40pa*data_read["40pa"]["debye_Di"]]
#
#dispersion_relation_all(w_20, w_25, w_30, w_15, w_40, k_20, k_25, k_30, k_15, k_40)
#
# Charge
#
a = json.load(open('ext-data/Fortov2004.json'))
b = json.load(open('ext-data/Khrapak2003.json'))
c = json.load(open('ext-data/Khrapak2005.json'))
d = json.load(open('ext-data/Yaroshenko2004.json'))
z_external = [[a['x'],a['y']], [b['x'],b['y']], [c['x'],c['y']], [d['x'],d['y']]]
z_d = [parameters["15pa"]["z_depl"], parameters["20pa"]["z_depl"], parameters["25pa"]["z_depl"], parameters["30pa"]["z_depl"], parameters["40pa"]["z_depl"]]
z_0 = [parameters["15pa"]["z"], parameters["20pa"]["z"], parameters["25pa"]["z"], parameters["30pa"]["z"], parameters["40pa"]["z"]]
z_error = [parameters["15pa"]["z_error"], parameters["20pa"]["z_error"], parameters["25pa"]["z_error"], parameters["30pa"]["z_error"], parameters["40pa"]["z_error"]]
ion_mean = [1/(parameters["15pa"]["n_0"]*parameters["neon"]["cross-section"]),1/(parameters["20pa"]["n_0"]*parameters["neon"]["cross-section"]),1/(parameters["25pa"]["n_0"]*parameters["neon"]["cross-section"]),1/(parameters["30pa"]["n_0"]*parameters["neon"]["cross-section"]),1/(parameters["40pa"]["n_0"]*parameters["neon"]["cross-section"])]
dx_ion_mean = [4.99514846e+20/(parameters["15pa"]["n_0"]**2*parameters["neon"]["cross-section"]), 4.99514846e+20/(parameters["20pa"]["n_0"]**2*parameters["neon"]["cross-section"]), 4.99514846e+20/(parameters["25pa"]["n_0"]**2*parameters["neon"]["cross-section"]), 4.99514846e+20/(parameters["30pa"]["n_0"]**2*parameters["neon"]["cross-section"]),4.99514846e+20/(parameters["40pa"]["n_0"]**2*parameters["neon"]["cross-section"])]
collisionality = [parameters["15pa"]["debye_D"]/ion_mean[0], parameters["20pa"]["debye_D"]/ion_mean[1], parameters["25pa"]["debye_D"]/ion_mean[2], parameters["30pa"]["debye_D"]/ion_mean[3], parameters["40pa"]["debye_D"]/ion_mean[4]]
#
charge_plot(z, z_0, z_error, z_external, collisionality)

#%%

### todo ###

#theory compare 30% efield pressure drop
#theory compare 40% efield pressure drop
#theory compare 60% efield pressure drop
#theory compare 80% efield pressure drop
#theory compare 100% efield pressure drop

#%% Save Data
# Write System Parameters Json #
data = {
                "15pa" :{
                    "v_group_100" : data_15_average,
                    "v_group_100_error" : error_15_average,
                    "k_100" : k_15pa,
                    "k_100_error" : k_15pa_error,
                    "c_daw_100" : speedlist_average_15pa_forward,
                    "c_daw_100_error" : error_speedlist_average_15pa_forward,
            },
                "20pa" : {
                    "v_group_100" : data_20_average,
                    "v_group_100_error" : error_20_average,
                    "v_group_30"  : data20_t13,
                    "v_group_30_error"  : error20_t13,
                    "v_group_40"  : data20_t14,
                    "v_group_40_error"  : error20_t14,
                    "v_group_60"  : data20_t16,
                    "v_group_60_error"  : error20_t16,
                    "v_group_80"  : data20_t18,
                    "v_group_80_error"  : error20_t18,
                    "c_daw_100" : speedlist_average_20pa_forward,
                    "c_daw_100_error" : error_speedlist_average_20pa_forward,
                    "c_daw_40"  : speedlist_average_20pa_forward_t14,
                    "c_daw_40_error"  : error_speedlist_average_20pa_forward_t14,
                    "c_daw_60"  : speedlist_average_20pa_forward_t16,
                    "c_daw_60_error"  : error_speedlist_average_20pa_forward_t16,
                    "c_daw_80"  : speedlist_average_20pa_forward_t18,
                    "c_daw_80_error"  : error_speedlist_average_20pa_forward_t18,
                    "k_100" : k_20pa,
                    "k_100_error" : k_20pa_error,
                    "k_80" : k_20pa_t18,
                    "k_80_error" : k_20pa_error_t18,
                    "k_60" : k_20pa_t16,
                    "k_60_error" : k_20pa_error_t16,
                    "k_40" : k_20pa_t14,
                    "k_40_error" : k_20pa_error_t14,
                    "k_30" : k_20pa_t13,
                    "k_30_error" : k_20pa_error_t13
                    
            },
                "25pa" : {
                    "v_group_100" : data_25_average,
                    "v_group_100_error" : error_25_average,
                    "v_group_30"  : data25_t13,
                    "v_group_30_error"  : error25_t13,
                    "v_group_40"  : data25_t14,
                    "v_group_40_error"  : error25_t14,
                    "v_group_60"  : data25_t16,
                    "v_group_60_error"  : error25_t16,
                    "v_group_80"  : data25_t18,
                    "v_group_80_error"  : error25_t18,
                    "c_daw_100" : speedlist_average_25pa_forward,
                    "c_daw_100_error" : error_speedlist_average_25pa_forward,
                    "c_daw_40"  : speedlist_average_25pa_forward_t14,
                    "c_daw_40_error"  : error_speedlist_average_25pa_forward_t14,
                    "c_daw_60"  : speedlist_average_25pa_forward_t16,
                    "c_daw_60_error"  : error_speedlist_average_25pa_forward_t16,
                    "c_daw_80"  : speedlist_average_25pa_forward_t18,
                    "c_daw_80_error"  : error_speedlist_average_25pa_forward_t18,
                    "k_100" : k_25pa,
                    "k_100_error" : k_25pa_error,
                    "k_80" : k_25pa_t18,
                    "k_80_error" : k_25pa_error_t18,
                    "k_60" : k_25pa_t16,
                    "k_60_error" : k_25pa_error_t16,
                    "k_40" : k_25pa_t14,
                    "k_40_error" : k_25pa_error_t14,
            },
                "30pa" : {
                    "v_group_100" : data_30_average,
                    "v_group_100_error" : error_30_average,
                    "v_group_40"  : data30_t14,
                    "v_group_40_error"  : error30_t14,
                    "v_group_50"  : data30_t15,
                    "v_group_50_error"  : error30_t15,
                    "v_group_60"  : data30_t16,
                    "v_group_60_error"  : error30_t16,
                    "v_group_80"  : data30_t18,
                    "v_group_80_error"  : error30_t18,
                    "c_daw_100" : speedlist_average_30pa_forward,
                    "c_daw_100_error" : error_speedlist_average_30pa_forward,
                    "c_daw_40"  : speedlist_average_30pa_forward_t14,
                    "c_daw_40_error"  : error_speedlist_average_30pa_forward_t14,
                    "c_daw_60"  : speedlist_average_30pa_forward_t16,
                    "c_daw_60_error"  : error_speedlist_average_30pa_forward_t16,
                    "c_daw_80"  : speedlist_average_30pa_forward_t18,
                    "c_daw_80_error"  : error_speedlist_average_30pa_forward_t18,
                    "k_100" : k_30pa,
                    "k_100_error" : k_30pa_error,
                    "k_80" : k_30pa_t18,
                    "k_80_error" : k_30pa_error_t18,
                    "k_60" : k_30pa_t16,
                    "k_60_error" : k_30pa_error_t16,
                    "k_40" : k_30pa_t14,
                    "k_40_error" : k_30pa_error_t14,
            },
                "40pa" : {
                    "v_group_100" : data_40_average,
                    "v_group_100_error" : error_40_average,
                    "k_100" : k_40pa,
                    "k_100_error" : k_40pa_error,
                    "c_daw_100" : speedlist_average_40pa_forward,
                    "c_daw_100_error" : error_speedlist_average_40pa_forward,
            }
        }
with open('resultsC17/finaldatastack.json', 'w') as filehandle:
    json.dump(data, filehandle)


