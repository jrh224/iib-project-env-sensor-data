## 20th Oct
## Design an

from matplotlib import pyplot as plt
from CustomDataframe import CustomDataframe

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)
sensor_data.filter_by_date(days=1)

sensor_data.smooth_data(column="Lux", window_size=100)
sensor_data.smooth_data(column="Re", window_size=100)

# sensor_data.plot(column="Re_smoothed", ylabel="Re")
# sensor_data.plot(column="Lux_smoothed", ylabel="Lux")

sensor_data.plot_dual(column1="Re_smoothed", column2="Lux_smoothed", ylabel1="Re smoothed", ylabel2="Lux smoothed", title="Re and light plotted for 15s 1week RoomA.csv")

