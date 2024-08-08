import matplotlib.pyplot as plt
import pandas as pd
import os

# Get filepath for CSV data required
root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
# print(os.listdir(filepath))
filename = 'Hourly 1 year RoomA.csv'
filepath = os.path.join(root_dir, "First_data_for_experimenting/", filename)

df = pd.read_csv(filepath,
                 parse_dates=['_start', '_stop', '_time']
                 )


print(df.dtypes)

# Filter the DataFrame where _field == 'Lux'
df_Lux = df[df['_field'] == 'Lux']

# Filter the DataFrame for the date range
start_date = '2023-04-10'
end_date = '2023-04-13'
df_Lux = df_Lux[(df_Lux['_time'] >= start_date) & (df_Lux['_time'] <= end_date)]

# Plot the filtered data
plt.scatter(df_Lux['_time'], df_Lux['_value'], label='Lux Field Data', s=10, alpha=0.7)
plt.title('Light data plotted hourly, for RoomA')
plt.xlabel('Time')
plt.ylabel('Lux')
plt.legend()
plt.grid(True)
plt.show()