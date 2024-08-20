from matplotlib import pyplot as plt
from SensorData import SensorData

filename = '15s 1week RoomA.csv'
sensor_data = SensorData(filename=filename)

filtered_data = sensor_data.filter_by_date(days=1)

temp_data = filtered_data.filter_by_reading_type('T')
controller_data = filtered_data.filter_by_reading_type('C')

temp_data.plot('Temp data plotted every 15s, for RoomA', 'Time', 'Temp')
# controller_data.plot('Controller duty plotted every 15s, for RoomA', 'Time', 'Controller Duty')

smoothed_temp_data = temp_data.smooth_data(window_size=50)
# smoothed_temp_data.plot('Smoothed temp data plotted every 15s, for RoomA', 'Time', 'Temp')

gradient_of_smoothed_temp = smoothed_temp_data.get_gradient()
# gradient_of_smoothed_temp.plot('Gradient of smoothed temp data plotted every 15s, for RoomA', 'Time', 'Temp')

peaks_of_smoothed_temp = gradient_of_smoothed_temp.get_significant_values()
peaks_of_smoothed_temp.plot_vertical_peaks()

plt.show()
