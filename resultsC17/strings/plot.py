import json
import matplotlib.pyplot as plt
import numpy

# Load the data from the JSON file
file_path_20pa_t20 = 'VM1_AVI_230125_104431_20pa_t12/velocities.npy'
file_path_20pa_t30 = 'VM1_AVI_230125_104518_20pa_t13/velocities.npy'
#
string_path_20pa_t20 = 'VM1_AVI_230125_104431_20pa_t12/percentages.npy'
string_path_20pa_t30 = 'VM1_AVI_230125_104518_20pa_t13/percentages.npy'

pixelsize = 0.0147
fps = 60
time_per_frame = 1000 / fps  # Time per frame in milliseconds
#
vel_20pa_t20 = numpy.multiply(numpy.load(file_path_20pa_t20)[50:530,0], pixelsize*fps)
vel_20pa_t30 = numpy.multiply(numpy.load(file_path_20pa_t30)[:,0], pixelsize*fps)
#
string_data_20pa_t20 = numpy.load(string_path_20pa_t20)[50:530,1]
string_data_20pa_t30 = numpy.load(string_path_20pa_t30)[:,1]
#
x_20pa_t20 = numpy.arange(len(vel_20pa_t20))
x_20pa_t30 = numpy.arange(len(vel_20pa_t30))
#
x_string_20pa_t20 = numpy.arange(len(string_data_20pa_t20))
x_string_20pa_t30 = numpy.arange(len(string_data_20pa_t30))
# Assume the length of your longest dataset to determine the range of the x-axis
max_frames = max(len(vel_20pa_t20), len(vel_20pa_t30), len(string_data_20pa_t20), len(string_data_20pa_t30))


# Plot the data
plt.figure(figsize=(10, 5),dpi=600)
plt.plot(x_20pa_t20, vel_20pa_t20, label='Velocity - 40% dc', color='#48A2F1', marker='x', markersize=4 ,markevery=2, linewidth=.9)
plt.plot(x_20pa_t30, vel_20pa_t30, label='Velocity - 35% dc', color='#48A2F1', marker='o', mfc='w', markersize=4 ,markevery=2, linewidth=.9)
plt.plot(x_string_20pa_t20, string_data_20pa_t20, label='String - 40% dc', color='#D81B1B', marker='x', markersize=4 ,markevery=2, linewidth=.9)
plt.plot(x_string_20pa_t30, string_data_20pa_t30, label='String - 35% dc', color='#D81B1B', marker='o', mfc='w', markersize=4 ,markevery=2, linewidth=.9)
#
plt.xlabel('Time [s]')
plt.ylabel('v [mm/s] & String [%]')
plt.legend(loc="upper left")

# Determine tick positions and labels, setting the interval at 0.5 seconds (fps/2 frames)
tick_positions = numpy.arange(0, max_frames, fps / 2)  # Tick every 0.5 seconds
tick_labels = [f"{x / fps:.1f}" for x in tick_positions]  # Convert frame number to seconds, 1 decimal place

plt.xticks(tick_positions, tick_labels)
plt.grid(True)

# Set x-axis limits
plt.xlim(0,max_frames-1)  # Set max limit to correspond to the maximum time in milliseconds

plt.tight_layout()
plt.show()