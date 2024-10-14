## 9th Oct: Check whether rapid temp drops always coincide with occupancy changes
## Particularly need to look at different seasons

from matplotlib import pyplot as plt
from CustomDataframe import CustomDataframe

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)

print(sensor_data.df["datetime"].min()) # This told us that the first reading is at 2024-01-15 18:29:00
print(sensor_data.df["datetime"].max()) # Last reading is at 2024-01-22 18:28:30

sensor_data.filter_by_date(start_date='15/01/2024', days=7)

sig_gradients = sensor_data.get_sig_gradients(column='T', max_clusters=50)

# Plot occupancy and temp decrease events together
for i, time in enumerate(sig_gradients):
    if i == 0:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1, label='Rapid temperature drop')
    else:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1)

sensor_data.smooth_data(column='Re', window_size=100)
sensor_data.plot(column='Re_smoothed', title='Smoothed Re data plotted every 15s, for RoomA', ylabel='Chance of occupancy')

plt.show()

# # Now plot temperature
# plt.clf()

# for i, time in enumerate(sig_gradients):
#     if i == 0:
#         plt.axvline(x=time, color='red', linestyle='--', linewidth=1, label='Rapid temperature drop')
#     else:
#         plt.axvline(x=time, color='red', linestyle='--', linewidth=1)

# smoothed_temp_data = temp_data.smooth_data()
# smoothed_temp_data.plot('Smoothed temp data plotted every 15s, for RoomA', 'Time', 'Temp')

# plt.show()

