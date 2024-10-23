## 20th Oct
## Design a manual algorithm get_occupancy()

from matplotlib import pyplot as plt
from CustomDataframe import CustomDataframe

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)
sensor_data.filter_by_date(days=1)

sensor_data.get_occupancy()

# sensor_data.plot_dual(column1="Re_smoothed", column2="Lux_smoothed", ylabel1="Re smoothed", ylabel2="Lux smoothed", title="Re and light plotted for 15s 1week RoomA.csv")

sensor_data.plot_dual(column1="Re_smoothed", column2="occupancy_factor", ylabel1="Re smoothed", ylabel2="Occupancy factor", title="Re and occupancy factor plotted for 15s 1week RoomA.csv")

