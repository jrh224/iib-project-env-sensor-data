## 9th Oct: Plots e.g. temperature data, raw

from matplotlib import pyplot as plt
from CustomDataframe import CustomDataframe

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv')

print(sensor_data.df)

sensor_data.filter_by_date(days=7)
sensor_data.smooth_data(column='T')
sensor_data.plot(column='T_smoothed', title='Smoothed T plotted every 15s', ylabel='Temp C', fontsize=12)

# sensor_data.plot_dual(column1="Me", ylabel1="Me values", column2="Md", ylabel2="Md values")

plt.show()
