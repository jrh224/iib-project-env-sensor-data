from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from CustomDataframe import CustomDataframe
from scipy.signal import find_peaks

sensor_data = CustomDataframe(filename="DataSorted_2024-08-08-09-09__SENS_36DC40metdata_combined.csv")

print(sensor_data.df["datetime"].min())

sampling_interval=900

fft_freqs, fft_values = sensor_data.get_fft(column="temp", smoothing_window=None, sampling_interval=sampling_interval, normalize=False) # temp is external, T is internal

frequency_cutoff = 1 / 43200  # 1 cycle per two days in Hz (don't want high frequencies which are negligible)

# Filter the frequencies and FFT values based on the cutoff
filtered_indices = np.where(fft_freqs <= frequency_cutoff)
filtered_freqs = fft_freqs[filtered_indices]
filtered_fft_values = fft_values[filtered_indices]

fft_values_mag = np.abs(filtered_fft_values) # Convert complex components into absolute for plotting

# Plotting
plt.plot(filtered_freqs, fft_values_mag)
plt.title("Frequency Domain of External Temperature")
plt.xlabel("Freq (hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Extract the peak
peak_indices = find_peaks(fft_values_mag, height=0.4)[0]
peak_freqs = [fft_freqs[i] for i in peak_indices]
print(peak_indices)
print(peak_freqs)

# Keep only the peak frequencies in the values
peak_fft_values = np.zeros_like(fft_values)
for i in peak_indices:
    peak_fft_values[i] = fft_values[i]
print(peak_fft_values)
reconstructed_signal = np.fft.irfft(filtered_fft_values)

# reconstructed_signal = np.fft.irfft(fft_values)[:800]

# Create the corresponding time array for plotting
num_samples = len(reconstructed_signal)
start_time = pd.Timestamp("2024-01-24 12:15:00+00:00") # Define the start time
time_array = pd.date_range(start=start_time, periods=num_samples, freq=f"{sampling_interval}S")
time_array = time_array[:800]

# Plot the reconstructed signal
plt.scatter(time_array, reconstructed_signal, s=10, alpha=0.7, label="Reconstructed temp data")
sensor_data.filter_by_date(days=7)
# sensor_data.plot(column="temp", show=True, label="Actual temp data")
plt.show()