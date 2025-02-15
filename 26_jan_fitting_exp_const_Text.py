# 14th dec - Fitting an exponential function to a manually selected cooling curve in room HC/2

from matplotlib import pyplot as plt
import numpy as np
from utils.CustomDataframe import CustomDataframe
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from datetime import datetime
from utils.openmeteocustom import get_external_temp
import pandas as pd


sensor_data = CustomDataframe(filename='wadham college/2024-11-30-12-31_influxdb_data-90ec.csv') # room HC/2


# 3 nice curves
# start_date="10/11/2024 13:00", end_date="10/12/2024 07:00"
# start_date="10/10/2024 13:00", end_date="10/11/2024 07:00"
# start_date="10/09/2024 14:30", end_date="10/10/2024 07:00"

start_date = pd.to_datetime("10/11/2024 13:00")
end_date = pd.to_datetime("10/12/2024 07:00")

sensor_data.filter_by_date(start_date=start_date, end_date=end_date)

# Going to try without smoothing data, since the data is going to be fit to a curve anyway
# sensor_data.smooth_data(column='T')

temp_df = get_external_temp(start_date=start_date.strftime('%Y-%m-%d'), end_date=end_date.strftime('%Y-%m-%d'))
temp_df['date'] = pd.to_datetime(temp_df['date'])  # Convert date column to datetime
temp_df['date'] = temp_df['date'].dt.tz_localize(None)

new_time_index = pd.date_range(start=start_date, end=end_date, freq='15S')  # 15-second intervals
new_temp_df = pd.DataFrame(new_time_index, columns=["datetime"])
new_temp_df['datetime'] = new_temp_df['datetime'].dt.tz_localize(None)

new_temp_df = pd.merge(new_temp_df, temp_df, left_on='datetime', right_on='date', how='left')
new_temp_df = new_temp_df.drop('date', axis=1)

# Perform linear interpolation to fill missing temperature values
new_temp_df['temperature_2m'] = new_temp_df['temperature_2m'].interpolate(method='linear')

sensor_data.df['datetime'] = sensor_data.df['datetime'].dt.tz_localize(None)
sensor_data.df = pd.merge(sensor_data.df, new_temp_df, left_on="datetime", right_on="datetime", how="left")

print(sensor_data.df)

T_in = sensor_data.df["T"].to_numpy()
T_ext = sensor_data.df["temperature_2m"].to_numpy()
T_ext_mean = T_ext.mean()
print(T_ext_mean)


# TRAIN A MODEL USING THESE VARIABLES

# Define the model function
# def model(t, RC, T_in0): # NEED TO PROPERLY DEFINE THIS MODEL. COULD ALSO ASSUME T_in0 EQUALS THE START VALUE IF NOT CONVERGING
#     # Numerically compute the integral
#     sum = T_in0 * np.exp(-t/RC)
#     u = 0 # represents the time value in the integral
#     while u <= t:
#         sum += (1/RC) * np.exp((u-t)/RC) * T_ext[u/15] * 15 # approximate with trapezium rule i.e. x15 seconds
#         u += 15
#     return sum

def model(t, RC):
    return T_ext_mean - (T_ext_mean-T_in[0]) * np.exp(-t/RC)

# CONTINUE HERE

t_array = sensor_data.df.index.to_numpy() * 15

# Fit the model to the data
initial_guess = [30000]  # Initial guesses for RC and T_in0
# popt, pcov = curve_fit(model, t_array, T_in, p0=initial_guess)
popt, pcov = curve_fit(model, t_array, T_in)

# Extract the fitted parameters
RC_fit = popt
print(f"Fitted RC (time constant): {RC_fit}")
# print(f"Fitted T_in0 (initial temperature): {T_in0_fit}")

# Predict T_in using the fitted parameters
t_array_extended = np.linspace(0, t_array[-1]*3, 100)
T_in_fit = model(t_array_extended, RC_fit)

# Plot the results
import matplotlib.pyplot as plt
plt.plot(t_array, T_in, 'o', label="Measured T_in")
plt.plot(t_array_extended, T_in_fit, '-', label=f"Fitted T_in (RC={RC_fit})")
# plt.plot(t_array, T_ext, 'o', label="External Temperature")
plt.xlabel("Time (s)")
plt.ylabel("T_in (°C)")
plt.legend()
plt.show()