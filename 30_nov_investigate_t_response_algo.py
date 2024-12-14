# 30th Nov - Looking at longer timeframes - this time I include a loop to enable the clustering algorithm to operate effectively

import datetime
import numpy as np
from CustomDataframe import CustomDataframe
from matplotlib import cm, pyplot as plt

filename = '15s 1week RoomA.csv'
sensor_data = CustomDataframe(filename=filename)

all_responses = []
all_initial_temps = []
int_date = sensor_data.df['datetime'].min()
for i in range(1):
    start_date = int_date + i*datetime.timedelta(days=7)
    end_date = int_date+(i+1)*datetime.timedelta(days=7)
    segment_of_sensor_data = sensor_data.filter_by_date(start_date=start_date, end_date=end_date, in_place=False)
    responses, initial_temps = segment_of_sensor_data.get_temp_response_series(column='T', min_response_length=40, max_clusters=100) # 80 corresponds to 20 mins responses
    # segment_of_sensor_data.visualise_location_of_temp_responses(column='T')
    all_responses = all_responses + responses
    all_initial_temps = all_initial_temps + initial_temps

print(str(len(all_responses)) + " appropriate responses")
print(initial_temps)

# Get mean and standard deviation of all_responses
mean_response, std_response = sensor_data.get_avg_temp_response_distrib(all_responses)

# Get shortest response (should be the same as min_response_length)
min_length = min(len(response) for response in all_responses)

# Generate time vector for plotting on same axis (assuming samples every 15s)
normalized_time = np.linspace(0, (min_length-1)*15, min_length)

# Create normaliser based on min and max initial temps
norm = plt.Normalize(min(all_initial_temps), max(all_initial_temps))
# Use cool-warm colour map
cmap = cm.coolwarm
for i, response in enumerate(all_responses):
    color = cmap(norm(all_initial_temps[i]))
    plt.plot(normalized_time, response, color=color)

# Add a colorbar for reference
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label="Initialtemperature (C)")

plt.plot(normalized_time, mean_response, label='Mean Response', color='black')
plt.fill_between(normalized_time, mean_response - std_response, mean_response + std_response, color='grey', alpha=0.2, label='±1 Std Dev')
plt.xlabel("Normalized Time (s)")
plt.ylabel("Normalized Temperature (C)")
plt.title("Mean and Standard Deviation of Responses")
plt.legend()
plt.show()


# plt.clf()
# sensor_data.visualise_location_of_temp_responses(column='T')

