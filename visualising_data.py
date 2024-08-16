# from utilities import *
from SensorData import SensorData

sensor_data = SensorData(filename='15s 1week RoomA.csv')

filtered_data = sensor_data.filter_by_reading_type('T').filter_by_date(days=1)

filtered_data.plot('Temp data plotted every 15s, for RoomA', 'Time', 'Temp')
