## 3rd Nov
## Use a histogram to identify distances of notable locations in the room e.g. bed, desk etc


from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from CustomDataframe import CustomDataframe


filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)
sensor_data.filter_by_date(days=7)

sensor_data.smooth_data(column="Me")
sensor_data.smooth_data(column="Md")
sensor_data.smooth_data(column="Re")
sensor_data.smooth_data(column="Rd", window_size=50)

bins = 30
sig_peaks_x, sig_peaks_heights = sensor_data.get_frequent_values(column="Rd", bins=bins, threshold_factor=0.2)

plt.title("Histogram of Rd distances")
plt.xlabel("Rd value")
plt.ylabel("Frequency")
plt.hist(sensor_data.df["Rd"][~np.isnan(sensor_data.df["Rd"])], bins=bins, histtype='step')

# Plot vertical lines where the significant peaks in the histogram are
for i, x in enumerate(sig_peaks_x):
    if i == 0:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=1, label='Peak')
    else:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=1)

plt.show()

sensor_data.plot(column="Rd_smoothed", title="Rd smoothed", ylabel="Rd smoothed")