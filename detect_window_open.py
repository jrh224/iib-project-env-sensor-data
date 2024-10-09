## 9th Oct:
## a) Plots occupancy (Re) along with vertical lines indicating rapid temp drops
## b) Plots temp along with vertical lines indicating rapid temp drops


from matplotlib import pyplot as plt
from SensorData import SensorData

filename = '15s 1week RoomA.csv'
sensor_data = SensorData(filename=filename)

filtered_data = sensor_data.filter_by_date(days=1)

temp_data = filtered_data.filter_by_reading_type('T')
controller_data = filtered_data.filter_by_reading_type('C')

## temp_data.plot('Temp data plotted every 15s, for RoomA', 'Time', 'Temp')
# controller_data.plot('Controller duty plotted every 15s, for RoomA', 'Time', 'Controller Duty')

smoothed_temp_data = temp_data.smooth_data(window_size=50)

gradient_of_smoothed_temp = smoothed_temp_data.get_gradient()
# gradient_of_smoothed_temp.plot('Gradient of smoothed temp data plotted every 15s, for RoomA', 'Time', 'Temp')

steepest_section_centers = gradient_of_smoothed_temp.get_window_open_events()

# Iterate over the steepest_section_centers and plot vertical lines where window open events occur
for i, time in enumerate(steepest_section_centers):
    if i == 0:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1, label='Rapid temperature drop')
    else:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1)

# Now plot occupancy
re_data = filtered_data.filter_by_reading_type('Re')
smoothed_re_data = re_data.smooth_data(window_size=100)
smoothed_re_data.plot('Smoothed Re data plotted every 15s, for RoomA', 'Time', 'Chance of occupancy')

plt.show(block=True)


# Now plot temperature
plt.clf()

for i, time in enumerate(steepest_section_centers):
    if i == 0:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1, label='Rapid temperature drop')
    else:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1)
smoothed_temp_data.plot('Smoothed temp data plotted every 15s, for RoomA', 'Time', 'Temp')

plt.show()