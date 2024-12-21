# 14th dec - Fitting an exponential function to a manually selected cooling curve in room HC/2

from matplotlib import pyplot as plt
import numpy as np
from CustomDataframe import CustomDataframe
from scipy.optimize import curve_fit
from datetime import datetime

sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2

sensor_data.filter_by_date(start_date="10/09/2024 14:30", end_date="10/10/2024 07:00")

sensor_data.smooth_data(column='T')

def model_func(t, A, K, C): # trying to fit the parameters A, K and C
    return A * np.exp(-K * t) + C

x_data = np.array([
    dt.timestamp() for dt in sensor_data.df[~np.isnan(sensor_data.df['T_smoothed'])]["datetime"].to_numpy()
])
# x_data = x_data - x_data[0] # Start from t = 0
y_data = sensor_data.df[~np.isnan(sensor_data.df['T_smoothed'])]["T_smoothed"].to_numpy()

print(x_data)
print(y_data)

optimised_parameters = curve_fit(model_func, x_data, y_data)[0]
print(optimised_parameters)

# Plot the results
plt.scatter(x_data, y_data, label="Data", color="blue", marker='.')
plt.plot(x_data, model_func(x_data, *optimised_parameters), label="Fitted Exponential Decay", color="red")
plt.xlabel('Time s')
plt.ylabel('Temp C')
plt.legend()
plt.show()

