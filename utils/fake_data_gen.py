from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import sys
import os

# Get the directory of the current file (fileB.py) and add it to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import openmeteocustom


def gen_sum_of_sines_with_exp_heating(hours=2160, length=25920, seed=42):
    t = np.linspace(0, hours, length) # Roughly based no 2160 hours in 3 months, with 12x as many datapoints
    x1 = 3 * np.sin((2 * np.pi * (t - 12)) / 24) + 7
    x2 = np.cos((2 * np.pi * (t+4)) / 4)

    # Step 1: Create the heater activation array
    x3 = np.zeros(length)
    # Randomly place heating events (ones)
    np.random.seed(seed)
    on_times = np.random.choice(np.arange(50, length - 50, 100), size=(length // 200), replace=False)
    durations = np.random.randint(2, 72, size=len(on_times))  # Random durations for heating, 10mins to 6 hrs

    for start, dur in zip(on_times, durations):
        x3[start:start + dur] = 1

    # Step 2: Define the exponential kernel for response
    tau = 4  # Time constant (time it takes to reach 63% of full value)
    t_kernel = np.arange(0, 5*tau)  # Kernel duration
    exp_kernel = np.exp(-t_kernel/tau)  # Exponential decay kernel
    exp_kernel /= exp_kernel.sum()  # Normalize

    # Step 3: Apply convolution to simulate temperature response
    scaling_factor = 3
    heating_increase = np.convolve(x3, exp_kernel, mode='same') * scaling_factor


    x4 = 3 * np.sin((2 * np.pi * (t+1)) / 5) + 2
    x5 = 2 * np.cos((2 * np.pi * (t-3)) / 2)
    x6 = 2 * np.cos((2 * np.pi * (t-3)) / 3)
    y = x1 + x2 + heating_increase + x4 + x5 + x6



    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    x3 = x3.reshape(-1, 1)
    x4 = x4.reshape(-1, 1)
    x5 = x5.reshape(-1, 1)
    x6 = x6.reshape(-1, 1)
    y = y.reshape(-1, 1)

    full_matrix = np.hstack((y, x1, x2, x3, x4, x5, x6))
    return full_matrix


def gen_sum_of_sines(hours=2160, length=25920):
    t = np.linspace(0, hours, length) # Roughly based no 2160 hours in 3 months, with 12x as many datapoints
    x1 = 3 * np.sin((2 * np.pi * (t - 12)) / 24) + 7
    x2 = np.cos((2 * np.pi * (t+4)) / 4)
    x3 = 0.5 * np.sin((2 * np.pi * (t-7)) / 9)
    x4 = 3 * np.sin((2 * np.pi * (t+1)) / 5) + 2
    x5 = 2 * np.cos((2 * np.pi * (t-3)) / 2)
    x6 = 2 * np.cos((2 * np.pi * (t-3)) / 3)
    y = x1 + x2 + x3 + x4 + x5 + x6

    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    x3 = x3.reshape(-1, 1)
    x4 = x4.reshape(-1, 1)
    x5 = x5.reshape(-1, 1)
    x6 = x6.reshape(-1, 1)
    y = y.reshape(-1, 1)

    full_matrix = np.hstack((y, x1, x2, x3, x4, x5, x6))
    return full_matrix



# Below is used for experiments from 9th march
def gen_sum_of_consts(hours=2160, length=25920, no_covariates=6, seed=42):
    np.random.seed(seed)
    AMPLITUDES = [-3, 7, 4, -1, 1, 6]
    full_matrix = np.zeros(shape=(length, no_covariates+1)) # Row 0 corresponds to y, the target
    for i in range(no_covariates):
        x = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x[start:start + dur] = 1

        # Calculate y_increase (DO NOT MAKE THIS RANDOM - OTHERWISE IT WILL GENERATE DIFFERENTLY FOR THE TEST DATA SET)
        y_increase = x * AMPLITUDES[i]
        
        full_matrix[:, i+1] = x # i+1 since first column is y target
        full_matrix[:, 0] += y_increase
    full_matrix[:, 0] += 15

    # plt.plot(full_matrix[:, 0])
    # plt.show()

    return full_matrix


def gen_sum_of_consts_w_lag(hours=2160, length=25920, no_covariates=6, seed=42):
    np.random.seed(seed)
    AMPLITUDES = [-3, 7, 4, -1, 1, 6]
    LAGS = [1, 2, 4, 3, 1, 3]
    full_matrix = np.zeros(shape=(length, no_covariates+1)) # Row 0 corresponds to y, the target
    for i in range(no_covariates):
        x = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x[start:start + dur] = 1

        # Calculate y_increase (DO NOT MAKE THIS RANDOM - OTHERWISE IT WILL GENERATE DIFFERENTLY FOR THE TEST DATA SET)
        y_increase = x * AMPLITUDES[i]
        # Shift y_increase along
        lag = LAGS[i]  # Adjust this to control how much lag you want
        y_increase = np.roll(y_increase, shift=lag)
        y_increase[:lag] = 0

        # plt.plot(y_increase, label="Y_increase")
        # plt.plot(x, label="Covariate xi")
        # plt.legend()
        # plt.show()

        full_matrix[:, i+1] = x # i+1 since first column is y target
        full_matrix[:, 0] += y_increase
    full_matrix[:, 0] += 15

    return full_matrix

def gen_sum_of_square_waves(hours=2160, length=25920, no_covariates=6, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, hours, length)
    AMPLITUDES = [-3, 7, 4, -1, 1, 6]
    PERIODS = np.array([10, 2, 7, 4, 8, 5]) / 5 # period corresponds to hours (since that is how t is defined). As written, period between 0.4 and 2 hours
    full_matrix = np.zeros(shape=(length, no_covariates+1)) # Row 0 corresponds to y, the target
    for i in range(no_covariates):
        x = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x[start:start + dur] = 1

        f = 1/PERIODS[i]
        y_increase = np.zeros_like(t) # Initialise output wave

        # Find where x transitions from 0 -> 1 (rising edges) since the square wave needs to start fresh for each of these
        rising_edges = np.where((x[:-1] == 0) & (x[1:] == 1))[0] + 1  # Add 1 to correct index shift ?

        for start in rising_edges:
            # Create time values local to this activation period
            local_t = np.arange(len(t) - start) / length * hours  # Convert index to time
            local_square_wave = signal.square(local_t * 2 * np.pi * f) * AMPLITUDES[i]
            
            # Apply until x turns off again
            end = start
            while end < length and x[end] == 1:
                y_increase[end] = local_square_wave[end - start]
                end += 1

        full_matrix[:, i+1] = x # i+1 since first column is y target
        full_matrix[:, 0] += y_increase

        # plt.plot(y_increase, label="Y_increase")
        # plt.plot(x, label="Covariate xi")
        # plt.legend()
        # plt.show()

    full_matrix[:, 0] += 15
        
    return full_matrix


def gen_sum_of_square_waves_smoothed(hours=2160, length=25920, no_covariates=6, sigma=5, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, hours, length)
    AMPLITUDES = [-3, 7, 4, -1, 1, 6]
    PERIODS = np.array([10, 2, 7, 4, 8, 5]) / 5 # period corresponds to hours (since that is how t is defined). As written, period between 0.4 and 2 hours
    full_matrix = np.zeros(shape=(length, no_covariates+1)) # Row 0 corresponds to y, the target
    for i in range(no_covariates):
        x = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x[start:start + dur] = 1

        f = 1/PERIODS[i]
        y_increase = np.zeros_like(t) # Initialise output wave

        # Find where x transitions from 0 -> 1 (rising edges) since the square wave needs to start fresh for each of these
        rising_edges = np.where((x[:-1] == 0) & (x[1:] == 1))[0] + 1  # Add 1 to correct index shift ?

        for start in rising_edges:
            # Create time values local to this activation period
            local_t = np.arange(len(t) - start) / length * hours  # Convert index to time
            local_square_wave = signal.square(local_t * 2 * np.pi * f) * AMPLITUDES[i]
            
            # Apply until x turns off again
            end = start
            while end < length and x[end] == 1:
                y_increase[end] = local_square_wave[end - start]
                end += 1

        from scipy.ndimage import gaussian_filter1d

        y_increase = gaussian_filter1d(y_increase, sigma=sigma)  # Adjust sigma for smoothness

        full_matrix[:, i+1] = x # i+1 since first column is y target
        full_matrix[:, 0] += y_increase

        # plt.plot(y_increase, label="Y_increase")
        # plt.plot(x, label="Covariate xi")
        # plt.legend()
        # plt.show()

    full_matrix[:, 0] += 15
        
    return full_matrix

# full_matrix = gen_sum_of_square_waves_smoothed(sigma=0.5)
# plt.plot(full_matrix[:, 1])
# plt.plot(full_matrix[:, 0])
# plt.show()
    

def gen_sum_of_sine_waves(hours=2160, length=25920, no_covariates=6, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, hours, length)
    AMPLITUDES = [-3, 7, 4, -1, 1, 6]
    PERIODS = np.array([10, 2, 7, 4, 8, 5]) / 5 # period corresponds to hours (since that is how t is defined). As written, period between 0.4 and 2 hours
    full_matrix = np.zeros(shape=(length, no_covariates+1)) # Row 0 corresponds to y, the target
    for i in range(no_covariates):
        x = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x[start:start + dur] = 1

        f = 1/PERIODS[i]
        y_increase = np.zeros_like(t) # Initialise output wave

        # Find where x transitions from 0 -> 1 (rising edges) since the square wave needs to start fresh for each of these
        rising_edges = np.where((x[:-1] == 0) & (x[1:] == 1))[0] + 1  # Add 1 to correct index shift ?

        for start in rising_edges:
            # Create time values local to this activation period
            local_t = np.arange(len(t) - start) / length * hours  # Convert index to time
            local_sine_wave = np.sin(local_t * 2 * np.pi * f) * AMPLITUDES[i]
            
            # Apply until x turns off again
            end = start
            while end < length and x[end] == 1:
                y_increase[end] = local_sine_wave[end - start]
                end += 1

        full_matrix[:, i+1] = x # i+1 since first column is y target
        full_matrix[:, 0] += y_increase

    return full_matrix

def gen_sum_of_sine_waves_rand_phase(hours=2160, length=25920, no_covariates=6, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, hours, length)
    AMPLITUDES = [-3, 7, 4, -1, 1, 6]
    PERIODS = np.array([10, 2, 7, 4, 8, 5]) / 5 # period corresponds to hours (since that is how t is defined). As written, period between 0.4 and 2 hours
    full_matrix = np.zeros(shape=(length, no_covariates+1)) # Row 0 corresponds to y, the target
    for i in range(no_covariates):
        x = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x[start:start + dur] = 1

        f = 1/PERIODS[i]
        y_increase = np.zeros_like(t) # Initialise output wave

        # Find where x transitions from 0 -> 1 (rising edges) since the square wave needs to start fresh for each of these
        rising_edges = np.where((x[:-1] == 0) & (x[1:] == 1))[0] + 1  # Add 1 to correct index shift ?

        for start in rising_edges:
            # Generate a random phase shift for the sine waves (this will depend on the seed, and so will be different in the train and test sets)
            random_phase = np.random.uniform(0, 2 * np.pi)

            # Create time values local to this activation period
            local_t = np.arange(len(t) - start) / length * hours  # Convert index to time
            local_sine_wave = np.sin(local_t * 2 * np.pi * f + random_phase) * AMPLITUDES[i]
            
            # Apply until x turns off again
            end = start
            while end < length and x[end] == 1:
                y_increase[end] = local_sine_wave[end - start]
                end += 1

        full_matrix[:, i+1] = x # i+1 since first column is y target
        full_matrix[:, 0] += y_increase

    return full_matrix




def gen_sum_of_exp(hours=2160, length=25920, no_covariates=6, seed=42):
    np.random.seed(seed)  # For reproducibility
    AMPLITUDES = [-3, 7, 4, -1, 1, 6]
    TAU = [1, 4, 2, 3 ,5, 1]
    full_matrix = np.zeros(shape=(length, no_covariates+1)) # to store each of the features
    for i in range(no_covariates):
        x_curr = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x_curr) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x_curr[start:start + dur] = 1

        # Define the exponential kernel for response
        tau = TAU[i]  # Time constant
        t_kernel = np.arange(0, 5*tau)  # Kernel duration
        exp_kernel = np.exp(t_kernel/tau)  # Exponential decay kernel
        exp_kernel /= exp_kernel.sum()  # Normalize
        scaling_factor = AMPLITUDES[i]

        # Step 3: Apply convolution to simulate temperature response
        temperature_increase = np.convolve(x_curr, exp_kernel, mode='same') * scaling_factor

        full_matrix[:, i+1] = x_curr
        full_matrix[:, 0] += temperature_increase

    full_matrix[:, 0] += 15
    return full_matrix

    
# full_matrix = gen_sum_of_sine_waves_rand_phase()
# plt.plot(full_matrix[:, 0])
# plt.plot(full_matrix[:, 1])
# plt.plot(full_matrix[:, 6])
# plt.show()




# RC synthetic data
def gen_r1c1_data(hours=2160, length=25920, tau=60, initial_temp=20, ext_temp_amp=5, seed=42):
    """
    Tau units is minutes (tau = 120 means room warms to 63% of the T_ext in 120 minutes)
    """
    np.random.seed(seed)  # For reproducibility
    t = np.linspace(0, hours, length)
    dt = 60 / (length / hours) # gives 5 minutes 
    ext_temp = 15 + ext_temp_amp * np.sin(2 * np.pi * t / (24 * 1/12))
    
    internal_temp = np.zeros(length)
    internal_temp[0] = initial_temp
    
    for t in range(1, length):
        internal_temp[t] = internal_temp[t-1] + (dt / tau) * (ext_temp[t-1] - internal_temp[t-1])
    
    return np.column_stack((internal_temp, ext_temp))

def gen_r2c2_data(hours=2160, length=25920, dt=60, tau1=60, tau2=120, initial_temp=20, ext_temp_amp=5, solar_amp=500, seed=42):
    np.random.seed(seed)  # For reproducibility
    t = np.linspace(0, hours, length)
    ext_temp = 15 + ext_temp_amp * np.sin(2 * np.pi * t / (24 * 1/12))
    solar_rad = solar_amp * np.maximum(0, np.sin(2 * np.pi * (t - 6 * 3600) / (24 * 1/12)))
    
    internal_temp = np.zeros(length)
    wall_temp = np.zeros(length)
    internal_temp[0] = initial_temp
    wall_temp[0] = initial_temp
    
    for t in range(1, length):
        wall_temp[t] = wall_temp[t-1] + (dt / tau2) * (ext_temp[t-1] - wall_temp[t-1])
        internal_temp[t] = internal_temp[t-1] + (dt / tau1) * (wall_temp[t-1] - internal_temp[t-1]) + (dt / tau1) * (solar_rad[t-1] / 100)
    
    return np.column_stack((internal_temp, ext_temp, solar_rad))

def gen_r2c2_w_regular_heating(hours=2160, length=25920, dt=60, 
                            C_wall=100000, C_room=3000, 
                            R_wall=0.1, R_room=0.05, R_heater=0.02, 
                            initial_temp=20, ext_temp_amp=5, solar_amp=500, 
                            heater_pow=2000, seed=42):
    np.random.seed(seed)  # For reproducibility
    t = np.linspace(0, hours, length) # units of t are hours
    ext_temp = 15 + ext_temp_amp * np.sin(2 * np.pi * (t-6) / (24))
    solar_rad = solar_amp * np.maximum(0, np.sin(2 * np.pi * ((t-6) - 6 * 3600) / (24)))

    # Heating signal (0 or 1)
    heater_signal = np.zeros(length)
    heater_on_times = [(6, 9), (18, 22)] # between these times
    for (start, end) in heater_on_times:
        heater_signal[(np.round(t) % 24 >= start) & (np.round(t) % 24 < end)] = 1  # Heater on
    
    # Initialize temperatures
    T_room = np.zeros(length)
    T_wall = np.zeros(length)
    T_room[0] = initial_temp
    T_wall[0] = initial_temp

    
    # Simulation loop (Euler integration)
    for t in range(1, length):
        # Heat exchange between external environment and wall
        dT_wall = (dt / C_wall) * (
            (ext_temp[t-1] - T_wall[t-1]) / R_wall -
            (T_wall[t-1] - T_room[t-1]) / R_room
        )
        
        # Heat exchange between wall and internal air, plus solar + heater inputs
        dT_room = (dt / C_room) * (
            (T_wall[t-1] - T_room[t-1]) / R_room +
            (solar_rad[t-1] / C_room) +
            (heater_signal[t-1] * heater_pow / C_room) / R_heater  # Heater as a heat flux
        )

        # Update temperatures
        T_wall[t] = T_wall[t-1] + dT_wall
        T_room[t] = T_room[t-1] + dT_room

    return np.column_stack((T_room, ext_temp, solar_rad, heater_signal))

def gen_r2c2_w_irregular_heating(hours=2160, length=25920, dt=60, 
                            C_wall=100000, C_room=3000, 
                            R_wall=0.1, R_room=0.05, R_heater=0.02, 
                            initial_temp=20, ext_temp_amp=5, solar_amp=500, 
                            heater_pow=2000, seed=42):
    np.random.seed(seed)  # For reproducibility
    t = np.linspace(0, hours, length) # units of t are hours
    ext_temp = 15 + ext_temp_amp * np.sin(2 * np.pi * (t-6) / (24))
    solar_rad = solar_amp * np.maximum(0, np.sin(2 * np.pi * ((t-6) - 6 * 3600) / (24)))

    # Heating signal (0 or 1)
    heater_signal = np.zeros(length)
    on_times = np.random.choice(np.arange(0, length, 5), size=len(heater_signal) // 200, replace=False)
    durations = np.random.randint(30, 100, size=len(on_times))
    for start, dur in zip(on_times, durations):
        heater_signal[start:start + dur] = 1

    # Initialize temperatures
    T_room = np.zeros(length)
    T_wall = np.zeros(length)
    T_room[0] = initial_temp
    T_wall[0] = initial_temp

    
    # Simulation loop (Euler integration)
    for t in range(1, length):
        # Heat exchange between external environment and wall
        dT_wall = (dt / C_wall) * (
            (ext_temp[t-1] - T_wall[t-1]) / R_wall -
            (T_wall[t-1] - T_room[t-1]) / R_room
        )
        
        # Heat exchange between wall and internal air, plus solar + heater inputs
        dT_room = (dt / C_room) * (
            (T_wall[t-1] - T_room[t-1]) / R_room +
            (solar_rad[t-1] / C_room) +
            (heater_signal[t-1] * heater_pow / C_room) / R_heater  # Heater as a heat flux
        )

        # Update temperatures
        T_wall[t] = T_wall[t-1] + dT_wall
        T_room[t] = T_room[t-1] + dT_room

    return np.column_stack((T_room, ext_temp, solar_rad, heater_signal))

def gen_r2c2_w_irregular_heating_real_meteo(hours=2160, length=25920, dt=300, 
                            C_wall=300000, C_room=500000, 
                            R_wall=0.2, R_room=0.1, 
                            initial_temp=20, solar_area=1, 
                            heater_pow=50, seed=42):
    np.random.seed(seed)  # For reproducibility
    t = np.linspace(0, hours, length) # units of t are hours

    start_date = pd.to_datetime("24/11/2024")
    end_date = pd.to_datetime("23/2/2025")
    LAT = 51.756
    LONG = 1.254
    hourly_df = openmeteocustom.get_temp_and_solar(start_date, end_date, LAT, LONG)
    hourly_df.set_index('date', inplace=True)
    highf_df = hourly_df.resample('5min').asfreq()
    highf_df.interpolate(method='linear', inplace=True)

    # plt.plot(hourly_df['temperature_2m'], label="External temp")
    # plt.plot(hourly_df['global_tilted_irradiance'], label="Solar irradiance")
    # plt.legend()
    # plt.show()
    # print(hourly_df)

    ext_temp = highf_df['temperature_2m'].to_numpy()[:length] # shorten so that they are 25920 data points long
    solar_rad = highf_df['global_tilted_irradiance'].to_numpy()[:length] * solar_area # take area into account

    # Heating signal (0 or 1)
    heater_signal = np.zeros(length)
    on_times = np.random.choice(np.arange(0, length, 5), size=len(heater_signal) // 200, replace=False)
    durations = np.random.randint(30, 100, size=len(on_times))
    for start, dur in zip(on_times, durations):
        heater_signal[start:start + dur] = 1

    # Initialize temperatures
    T_room = np.zeros(length)
    T_wall = np.zeros(length)
    T_room[0] = initial_temp
    T_wall[0] = initial_temp

    # Simulation loop (Euler integration)
    for t in range(1, length):
        # Heat flow from outside to wall
        Q_wall = (ext_temp[t-1] - T_wall[t-1]) / R_wall

        # Heat flow from wall to room
        Q_room = (T_wall[t-1] - T_room[t-1]) / R_room

        # Heater power (converted from Watts to Joules per step)
        Q_heater = heater_signal[t-1] * heater_pow  

        # Solar heat contribution
        Q_solar = solar_rad[t-1]  

        # Update wall temperature
        dT_wall = dt * (Q_wall - Q_room) / C_wall
        T_wall[t] = T_wall[t-1] + dT_wall

        # Update room temperature
        dT_room = dt * (Q_room + Q_solar + Q_heater) / C_room
        T_room[t] = T_room[t-1] + dT_room

        # # Prevent numerical instability (limit excessive changes per step)
        # if np.abs(dT_room) > 2:
        #     T_room[t] = T_room[t-1] + np.sign(dT_room) * 2 

    return np.column_stack((T_room, ext_temp, solar_rad, heater_signal))


def gen_r2c2_w_neighbour_room(hours=2160, length=25920, dt=300, 
                            C_wall=300000, C_room=500000, C_shared=150000,
                            R_wall=0.2, R_room=0.1, R_shared=0.1,
                            initial_temp=20, solar_area=1, 
                            heater_pow=50, seed=42):
    np.random.seed(seed)  # For reproducibility
    t = np.linspace(0, hours, length) # units of t are hours

    start_date = pd.to_datetime("24/11/2024")
    end_date = pd.to_datetime("23/2/2025")
    LAT = 51.756
    LONG = 1.254
    hourly_df = openmeteocustom.get_temp_and_solar(start_date, end_date, LAT, LONG)
    hourly_df.set_index('date', inplace=True)
    highf_df = hourly_df.resample('5min').asfreq()
    highf_df.interpolate(method='linear', inplace=True)

    # plt.plot(hourly_df['temperature_2m'], label="External temp")
    # plt.plot(hourly_df['global_tilted_irradiance'], label="Solar irradiance")
    # plt.legend()
    # plt.show()
    # print(hourly_df)

    ext_temp = highf_df['temperature_2m'].to_numpy()[:length] # shorten so that they are 25920 data points long
    solar_rad = highf_df['global_tilted_irradiance'].to_numpy()[:length] * solar_area # take area into account

    # Heating signal (0 or 1)
    heater_signal = np.zeros(length)
    on_times = np.random.choice(np.arange(0, length, 5), size=len(heater_signal) // 200, replace=False)
    durations = np.random.randint(30, 100, size=len(on_times))
    for start, dur in zip(on_times, durations):
        heater_signal[start:start + dur] = 1

    # Initialize temperatures
    T_room = np.zeros(length)
    T_wall = np.zeros(length)
    T_shared = np.zeros(length)
    T_room[0] = initial_temp
    T_wall[0] = initial_temp
    T_shared[0] = initial_temp

    T_neighbour = initial_temp # Initially model this as a constant

    # Simulation loop (Euler integration)
    for t in range(1, length):
        # Heat flow from outside to exterior wall
        Q_wall = (ext_temp[t-1] - T_wall[t-1]) / R_wall

        # Heat flow from exterior wall to room
        Q_room = (T_wall[t-1] - T_room[t-1]) / R_room

        # Heat flow from room to shared wall
        Q_shared_in = (T_room[t-1] - T_shared[t-1]) / R_shared

        # Heat flow from shared wall to neighboring room (fixed temperature)
        Q_shared_out = (T_shared[t-1] - T_neighbour) / R_shared

        # Heater power (converted from Watts to Joules per step)
        Q_heater = heater_signal[t-1] * heater_pow  

        # Solar heat contribution
        Q_solar = solar_rad[t-1]  

        # Update exterior wall temperature
        dT_wall = dt * (Q_wall - Q_room) / C_wall
        T_wall[t] = T_wall[t-1] + dT_wall

        # Update room temperature
        dT_room = dt * (Q_room + Q_solar + Q_heater - Q_shared_in) / C_room
        T_room[t] = T_room[t-1] + dT_room

        # Update shared wall temperature
        dT_shared = dt * (Q_shared_in - Q_shared_out) / C_shared
        T_shared[t] = T_shared[t-1] + dT_shared

    return np.column_stack((T_room, ext_temp, solar_rad, heater_signal))




# full_matrix = gen_r2c2_w_neighbour_room()
# t = np.linspace(0, 2160, 25920)
# plt.plot(t, full_matrix[:, 0], label="T_int")
# plt.plot(t, full_matrix[:, 1], label="T_ext")
# plt.plot(t, full_matrix[:, 2]/100, label="Solar irradiance")
# plt.plot(t, full_matrix[:, 3], label="Heating")
# plt.legend()
# plt.show()
