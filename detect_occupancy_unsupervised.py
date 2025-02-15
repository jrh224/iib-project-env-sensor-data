## 23rd Oct
## Design an unsupervised learning algorithm to identify the states

from matplotlib import pyplot as plt
import pandas as pd
from utils.CustomDataframe import CustomDataframe

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)


# sensor_data.filter_by_date(start_date='17/01/2024', days=1)
sensor_data.filter_by_date(days=1)

window_size = 120

sensor_data.smooth_data(column="Re", window_size=window_size)
sensor_data.smooth_data(column="Rd", window_size=window_size)

sensor_data.cluster_timeseries(window_size=window_size, n_clusters=None)

sensor_data.plot_dual(column1="Rd_smoothed", column2="Re_smoothed", ylabel1="Rd smoothed", ylabel2="Re smoothed", show=False, fontsize=14, labelsize=14)

plt.title("Re and Rd plotted with clusters detected via unsupervised K-means clustering")

sensor_data.plot_clusters()

plt.tight_layout()
plt.show()

# sensor_data.plot_dual(column1="Me", ylabel1="Me values", column2="Md", ylabel2="Md values")
