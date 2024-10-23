## 23rd Oct
## Design an unsupervised learning algorithm to identify the states

from matplotlib import pyplot as plt
from CustomDataframe import CustomDataframe

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)
sensor_data.filter_by_date(days=1)

window_size = 120

sensor_data.smooth_data(column="Re", window_size=window_size)
sensor_data.smooth_data(column="Lux", window_size=window_size)

sensor_data.cluster_timeseries(window_size=window_size, n_clusters=3)

sensor_data.plot_dual(column1="Lux_smoothed", column2="Re_smoothed", ylabel1="Lux smoothed", ylabel2="Re smoothed", show=False)

plt.title("Re and Lux plotted with clusters detected via unsupervised K-means clustering. Features: Re mean, Re median, Lux means")

sensor_data.plot_clusters()

plt.show()
