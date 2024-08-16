from SensorData import SensorData

filename = '15s 1week RoomA.csv'
sensor_data = SensorData(filename=filename)

filtered_data = sensor_data.filter_by_date(days=1)

temp_data = filtered_data.filter_by_reading_type('T')
controller_data = filtered_data.filter_by_reading_type('C')

temp_data.plot('Temp data plotted every 15s, for RoomA', 'Time', 'Temp')
controller_data.plot('Controller duty plotted every 15s, for RoomA', 'Time', 'Controller Duty')
