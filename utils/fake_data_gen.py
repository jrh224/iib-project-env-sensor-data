from matplotlib import pyplot as plt
import numpy as np


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

def gen_sum_of_exp(hours=2160, length=25920, no_features=6, seed=42):
    np.random.seed(seed)  # For reproducibility
    full_matrix = np.zeros(shape=(25920, 7)) # to store each of the features
    for i in range(no_features):
        x_curr = np.zeros(length)
        on_times = np.random.choice(np.arange(0, length, 5), size=len(x_curr) // 200, replace=False)
        durations = np.random.randint(30, 100, size=len(on_times))
        for start, dur in zip(on_times, durations):
            x_curr[start:start + dur] = 1

        # Define the exponential kernel for response
        tau = np.random.randint(1, 12, size=1).item()  # Time constant
        t_kernel = np.arange(0, 5*tau)  # Kernel duration
        exp_kernel = np.exp(t_kernel/tau)  # Exponential decay kernel
        exp_kernel /= exp_kernel.sum()  # Normalize
        if np.random.random() > 0.5: # Random decide on heating or cooling
            exp_kernel = -exp_kernel
        scaling_factor = np.random.random() * 4

        # Step 3: Apply convolution to simulate temperature response
        temperature_increase = np.convolve(x_curr, exp_kernel, mode='same') * scaling_factor

        full_matrix[:, i] = x_curr
        full_matrix[:, 0] += temperature_increase

    full_matrix[:, 0] += 15
    return full_matrix