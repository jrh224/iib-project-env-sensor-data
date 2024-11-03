## 9th Oct: Plots e.g. temperature data, raw

from matplotlib import pyplot as plt
from CustomDataframe import CustomDataframe

sensor_data = CustomDataframe(filename='15s 1week RoomA.csv')

sensor_data.filter_by_date(days=1)
sensor_data.plot(column='Me', title='Me plotted every 15s, for RoomA', ylabel='Temp C')

plt.show()
