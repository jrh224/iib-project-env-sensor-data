## 29th October
## Apply a fourier transform to temperature data to determine the frequencies involved

from matplotlib import pyplot as plt
import numpy as np
from CustomDataframe import CustomDataframe

# sensor_data = CustomDataframe(filename="DataSorted_2024-08-08-09-09__SENS_36DC40metdata_combined.csv")
sensor_data = CustomDataframe(filename='15s 1week RoomA.csv')
sampling_interval = 15

fft_freqs, fft_values  = sensor_data.get_fft(column="T", smoothing_window=None, sampling_interval=sampling_interval, normalize=True) # temp is external, T is internal

# frequency_cutoff = 1 / 43200  # 1 cycle per two days in Hz

# # Filter the frequencies and FFT values based on the cutoff
# filtered_indices = np.where(fft_freqs <= frequency_cutoff)
# filtered_freqs = fft_freqs[filtered_indices]
# filtered_fft_values = fft_values[filtered_indices]

print(fft_freqs)

# Plotting
plt.plot(fft_freqs, np.abs(fft_values))
plt.title("Frequency Domain of Internal Temperature")
plt.xlabel("Freq (hz)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()