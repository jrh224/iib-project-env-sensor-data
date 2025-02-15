# 26th November - First algorithm for pulling out responses and 

import numpy as np
from utils.CustomDataframe import CustomDataframe
from matplotlib import pyplot as plt

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)

sensor_data.filter_by_date(days=3)

all_responses = sensor_data.get_temp_response_series(column='T')

# Get mean and standard deviation of all_responses
mean_response, std_response = sensor_data.get_avg_temp_response_distrib(all_responses)

# Get shortest response
min_length = min(len(response) for response in all_responses)

# Generate time vector for plotting on same axis (assuming samples every 15s)
normalized_time = np.linspace(0, (min_length-1)*15, min_length)

for response in all_responses:
    plt.plot(normalized_time, response)

plt.plot(normalized_time, mean_response, label='Mean Response', color='blue')
plt.fill_between(normalized_time, mean_response - std_response, mean_response + std_response, color='blue', alpha=0.2, label='±1 Std Dev')
plt.xlabel("Normalized Time (s)")
plt.ylabel("Normalized Temperature (C)")
plt.title("Mean and Standard Deviation of Responses")
plt.legend()
plt.show()


# plt.clf()
# sensor_data.visualise_location_of_temp_responses(column='T')

