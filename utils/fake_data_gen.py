from matplotlib import pyplot as plt
import numpy as np
from scipy import signal


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
            local_square_wave = np.sin(local_t * 2 * np.pi * f) * AMPLITUDES[i]
            
            # Apply until x turns off again
            end = start
            while end < length and x[end] == 1:
                y_increase[end] = local_square_wave[end - start]
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

    
# full_matrix = gen_sum_of_exp()
# plt.plot(full_matrix[:, 0])
# plt.plot(full_matrix[:, 1])
# plt.plot(full_matrix[:, 6])
# plt.show()