# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:50:49 2024

@author: Lukas
"""

import pims
import numpy as np

# Function to calculate absolute brightness
def calculate_brightness(image):
    # Convert image to grayscale if needed (assumes 2D or 3D with channels)
    if image.ndim == 3:  # If image has color channels
        image = np.mean(image, axis=-1)  # Convert to grayscale by averaging channels
    return np.sum(image)  # Sum of pixel values gives absolute brightness

# Function to process all images and calculate brightness series and average brightness
def process_image_series(folder_path):
    # Use pims to open all BMP images in the folder
    images = pims.ImageSequence(folder_path + "/*.bmp")
    
    brightness_series = []
    
    # Iterate over each image and calculate brightness
    for image in images:
        brightness = calculate_brightness(image)
        brightness_series.append(brightness)
    
    # Calculate average brightness over the series
    average_brightness = np.mean(brightness_series)
    
    return brightness_series, average_brightness

# Example usage
file_name = '/Parabola#16-20pa-100trial70'
#folder_path = 'D:\\C15_Boundary_Layer_Data\C#15\DC100' + file_name + '\Forward'  # Change this to your folder path
folder_path = 'D://PFC42-D3-raw' + file_name + '/100/wave'  # Change this to your folder path
output_file = 'brightness' + file_name + '.txt' 
brightness_series, average_brightness = process_image_series(folder_path)

np.savetxt(output_file, [average_brightness], fmt='%f')

# Print the results
print("Brightness series:", brightness_series)
print("Average brightness:", average_brightness)
#%%

brightness_values_20 = [

    10542119.173913,  # VM2_AVI_230125_103901_20pa.txt

    10648718.651163,  # VM2_AVI_230125_103947_20pa.txt

    9945435.980000,   # VM2_AVI_230125_104625_20pa.txt

    10706953.968254,  # VM2_AVI_230125_104809_20pa.txt

    10976736.235294,  # VM2_AVI_230125_110808_25pa.txt

    10524720.906667   # VM2_AVI_230125_111329_25pa.txt

]

brightness_values_25 = [

    10360413.568345,  # VM2_AVI_230124_123220_25pa.txt
    
    10976736.235294,  # VM2_AVI_230125_110808_25pa.txt

    10524720.906667   # VM2_AVI_230125_111329_25pa.txt
]

brightness_values_30 = [

    10546160.425000,  # VM2_AVI_230125_105732_30pa.txt

    10598369.230769,  # VM2_AVI_230125_110231_30pa.txt
]

deviation_20_30 = abs(np.average(brightness_values_20) - np.average(brightness_values_30))

brightness_pfc = [

    5189529.554455,  # Parabola#0-40pa-100trial70.txt

    4687006.277228,  # Parabola#6-30pa-100trial70.txt

    4037776.296703,  # Parabola#10-25pa-100trial70.txt

    5425786.881188,  # Parabola#16-20pa-100trial70.txt

    5125554.957746   # Parabola#19-15pa-100trial70.txt

]

deviation_pfc_40_25 = (100/brightness_pfc[0])*(brightness_pfc[0]-brightness_pfc[2])
