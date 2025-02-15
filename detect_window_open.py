## 9th Oct:
## a) Plots occupancy (Re) along with vertical lines indicating rapid temp drops
## b) Plots temp along with vertical lines indicating rapid temp drops


from matplotlib import pyplot as plt
from utils.CustomDataframe import CustomDataframe

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)

sensor_data.filter_by_date(days=1)
steepest_sections = sensor_data.get_sig_gradients(column='T')

# Iterate over the steepest_section_centers and plot vertical lines where window open events occur
for i, time in enumerate(steepest_sections):
    if i == 0:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1, label='Rapid temperature drop')
    else:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1)

# Now plot occupancy
sensor_data.smooth_data('Re', window_size=100)
sensor_data.plot(column='Re_smoothed', title='Smoothed Re data plotted every 15s, for RoomA', ylabel='Chance of occupancy')

plt.show(block=True)


# Now plot temperature
plt.clf()

for i, time in enumerate(steepest_sections):
    if i == 0:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1, label='Rapid temperature drop')
    else:
        plt.axvline(x=time, color='red', linestyle='--', linewidth=1)

sensor_data.smooth_data(column='T')
sensor_data.plot(column='T_smoothed', title='Smoothed temp data plotted every 15s, for RoomA', ylabel='Temp')

plt.show()