from matplotlib import pyplot as plt
from SensorData import SensorData


filename = '15s 1week RoomA.csv'
sensor_data = SensorData(filename=filename)

filtered_data = sensor_data.filter_by_date(days=1)
re_data = filtered_data.filter_by_reading_type('Re')

smoothed_temp_data = re_data.smooth_data(window_size=100)
smoothed_temp_data.plot('Smoothed Re data plotted every 15s, for RoomA', 'Time', 'Temp')

plt.show()