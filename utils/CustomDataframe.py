# 13th Oct: Combines deprecated classes SensorData and CombinedData

from datetime import timedelta
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.signal import find_peaks

from utils.openmeteocustom import *


class CustomDataframe:
    def __init__(self, filename=None, df=None):
        self.filename = filename
        self.df = df
        if self.df is None:
            self.load_and_clean_data()
        
    def load_and_clean_data(self):
        # Get filepath for CSV data required
        root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        # print(os.listdir(filepath))
        filepath = os.path.join(root_dir, "First_data_for_experimenting/", self.filename)

        self.df = pd.read_csv(filepath,
                        comment='#'
                        )
        # Drop rows that are completely empty (if any)
        self.df.dropna(how='all', inplace=True)

        if '_field' in self.df.columns:  # Type 1 long format
            # Convert the "value" column (assumed to be column 6) to a double (float)
            # If conversion fails, set those values to NaN
            self.df['_value'] = pd.to_numeric(self.df['_value'], errors='coerce')
            # Convert '_start', '_stop' and '_time' to datetime objects
            self.df['_start'] = pd.to_datetime(self.df['_start'], errors='coerce')
            self.df['_stop'] = pd.to_datetime(self.df['_stop'], errors='coerce')
            self.df['_time'] = pd.to_datetime(self.df['_time'], errors='coerce')

            # Drop rows where there are NaN values (i.e., failed conversions)
            self.df.dropna(subset=['_value', '_start', '_stop', '_time'], inplace=True)
            # Pivot the DataFrame to convert it into the desired format
            self.df = self.df.pivot(index='_time', columns='_field', values='_value')
            self.df.reset_index(inplace=True)

            self.df.rename(columns={"_time":"datetime"}, inplace=True)

        # ADD IN DEALING WITH TYPE 2 CSV
        else:
            # Drop rows which don't have an internal T reading (the first ~2200 rows)
            self.df.dropna(subset="T", inplace=True)
            # Rename the datetime column
            self.df.rename(columns={"Unnamed: 0":"datetime"}, inplace=True)
            # Convert datetime column to datetime
            self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce')



    def filter_by_date(self, start_date=None, end_date=None, days=None, in_place=True):
        """
        Provide either an end_date, or a number of days.
        """
        if (end_date is None and days is None) or (end_date is not None and days is not None):
            raise ValueError("Only one of 'end_date' and 'days' must be provided.")
        
        if start_date is None:
            start_date = self.df['datetime'].min()
        else:
            start_date = pd.to_datetime(start_date)

        # Align start_date with the timezone of the `datetime` column, if it has one
        if pd.api.types.is_datetime64tz_dtype(self.df['datetime']):
            start_date = start_date.tz_localize(self.df['datetime'].dt.tz) if start_date.tzinfo is None else start_date.tz_convert(self.df['datetime'].dt.tz)
        
        if end_date is None:
            end_date = start_date + timedelta(days=days)
        else:
            end_date = pd.to_datetime(end_date)

        # Align end_date with the timezone of the `_time` column, if it has one
        if pd.api.types.is_datetime64tz_dtype(self.df['datetime']):
            end_date = end_date.tz_localize(self.df['datetime'].dt.tz) if end_date.tzinfo is None else end_date.tz_convert(self.df['datetime'].dt.tz)

        if in_place:
            self.df = self.df[(self.df['datetime'] >= start_date) & (self.df['datetime'] <= end_date)]
        else:
            return CustomDataframe(df=self.df[(self.df['datetime'] >= start_date) & (self.df['datetime'] <= end_date)])
        
    def filter_by_date_ranges(self, dates, in_place=True):
        """
        Pass in dates in the following format: 
        [
        ("2024-11-24 00:00:15", "2024-11-30 00:00:00"),
        ("2024-12-10 00:00:15", "2025-01-15 00:00:00"),
        ("2025-01-24 00:00:15", "2025-02-14 00:00:00")
        ]
        NB: Strings may be pd.datetime objects 

        Also returns idx_blocks, which represents the indices of the ends of each continuous block (inclusive)
        e.g. [(0, 1727), (1728, 3167), (3168, 4895)]
        """
        # First convert all dates to pd.datetime objects just in case
        dates_dt = []
        for i in range(len(dates)):
            dates_dt.append((pd.to_datetime(dates[i][0]).tz_localize("UTC"), pd.to_datetime(dates[i][1]).tz_localize("UTC")))

        # # Set datetime to be the index
        # self.df.set_index("datetime", inplace=True)

        # Initialize boolean mask as a Pandas Series
        mask = pd.Series(False, index=self.df.index)

        # Apply each date range
        for start, end in dates_dt:
            mask |= (self.df.index >= start) & (self.df.index <= end)  # Accumulate using OR (|)

        # Convert to boolean if needed
        mask = mask.astype(bool)

        filtered_df = self.df.loc[mask]

        # Return the indices that correspond to continuous blocks
        idx_blocks = []
        filtered_indices = filtered_df.index.to_numpy()
        for block in dates_dt:
            # Attention: defined such that the end indices of a block are inclusive
            idx_blocks.append((np.where(filtered_indices >= block[0])[0][0], np.where(filtered_indices <= block[1])[0][-1]))

        if not in_place:
            return CustomDataframe(df=filtered_df), idx_blocks
        
        self.df = filtered_df
        return None, idx_blocks
    
        
    def plot(self, column, title=None, label=None, xlabel="Time", ylabel="Value", show=True, fontsize=12):
        # Plot the filtered data
        plt.scatter(self.df.index, self.df[column], label=label, s=10, alpha=0.7)
        plt.title(title)
        plt.gca().set_xlabel(xlabel, fontsize=fontsize)
        plt.gca().set_ylabel(ylabel, fontsize=fontsize)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.subplots_adjust(right=0.7)
        plt.grid(True)
        # Rotate and format date labels for better readability
        plt.gcf().autofmt_xdate()

        if show:
            plt.show()


    def plot_dual(self, column1, column2, title=None, xlabel="Time", ylabel1 = "Value", ylabel2="Value", show=True, fontsize=12, labelsize=12):
        fig, ax1 = plt.subplots()
        ax1.set_title(title)
        ax1.grid(True)
        
        color = 'tab:red'
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel(ylabel1, color=color, fontsize=fontsize)
        ax1.scatter(self.df['datetime'], self.df[column1], s=10, alpha=0.7, color=color)
        ax1.tick_params(axis='y', labelcolor=color, labelsize=labelsize)
        ax1.tick_params(axis='x', labelsize=labelsize)  # Set font size for x-axis ticks

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel(ylabel2, color=color, fontsize=fontsize)  # we already handled the x-label with ax1
        ax2.scatter(self.df['datetime'], self.df[column2], s=10, alpha=0.7, color=color)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=labelsize)
        
        fig.tight_layout()

        # Rotate and format date labels for better readability
        plt.gcf().autofmt_xdate()
        
        if show:
            plt.show(block=True)
            plt.clf()

    def plot_clusters(self):
        colour_map = {0: 'red', 1: 'yellow', 2: 'green', 3: 'blue', 4:'purple', 5:'grey'}
        for idx, row in self.df.iterrows():
            # Create a bar for each timestamp
            if pd.notna(row['cluster']):
                plt.bar(x=row['datetime'], height=100, width=timedelta(seconds=15), color=colour_map[row['cluster']], alpha=0.3, align='center')

        
    def smooth_data(self, column, window_size=120, in_place=True): # used to default to 50
        # Apply a rolling window to smooth the 'y' values

        if in_place:
            df = self.df
        else:
            df = self.df.copy() # Creates a deep copy of the dataframe, to avoid editing the original df

        df[column+'_smoothed'] = df[column].rolling(window=window_size).mean()

        if not in_place:
            return CustomDataframe(df=df)
        
    def get_gradient(self, column, in_place=True):
        if in_place:
            df = self.df
        else:
            df = self.df.copy() # Creates a deep copy of the dataframe, to avoid editing the original df

        df['dy'] = df[column].diff()
        df['dx'] = df['datetime'].diff().dt.total_seconds()
        df[column + '_grad'] = df['dy'] / df['dx'] # Calculate running gradient, and store as column_grad

        if not in_place:
            return CustomDataframe(df=df)
        
    def get_optimal_clusters(self, max_clusters, features):
        best_n_clusters = 2
        best_score = -1
        # Determine the optimal number of clusters using silhouette score
        for i_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i_clusters, random_state=0).fit(features)
            labels = kmeans.labels_
            score = silhouette_score(features, labels)
            if score > best_score:
                best_n_clusters = i_clusters
                best_score = score
        print(f"Optimal number of clusters: {best_n_clusters}")
        return best_n_clusters
        
    def get_sig_clusters(self, column, pos_or_neg='both', threshold_factor_no_stds=2, max_clusters=10):
        """
        Identifies clusters of significantly high values, returning the timestamps of the earliest events in each cluster.
        NB: Used by get_sig_gradients. Usually get_sig_clusters is not called directly.

        Parameters:
        - `column` (str): The target column to find the clusters of
        - `threshold_factor_no_stds` (float, default=2): Multiplier for the standard deviation to define the threshold for significant negative values.
        - `max_clusters` (int, default=10): Maximum number of clusters to evaluate for optimal clustering.
        - `pos_or_neg` (string, default='both'): Determines whether to return all sig_clusters, or only ones of a certain sign

        Returns:
        - `data_with_clusters` (pd.DataFrame): Contains all the timestamps associated with a high gradient, as well as the cluster it has been assigned to.
        """
        threshold = self.df[column].std() * threshold_factor_no_stds
        # create 2D array of all timestamps of detected event
        if pos_or_neg == 'both':
            ungrouped_signif_vals = pd.Series(self.df[abs(self.df[column]) > threshold]['datetime']) # filter df based on abs value being greater than threshold
        elif pos_or_neg == 'neg':
            ungrouped_signif_vals = pd.Series(self.df[self.df[column] < -threshold]['datetime']) # filter dataframe based on value being negative and bigger mag than threshold
        elif pos_or_neg == 'pos':
            ungrouped_signif_vals = pd.Series(self.df[self.df[column] > threshold]['datetime']) # filter df based on abs value being lower than threshold
        else:
            raise ValueError("pos_or_neg should take a valid string value")
        ungrouped_data_reshaped = ungrouped_signif_vals.values.reshape(-1, 1)

        best_n_clusters = self.get_optimal_clusters(max_clusters=max_clusters, features=ungrouped_data_reshaped)

        kmeans = KMeans(best_n_clusters).fit(ungrouped_data_reshaped)
        data_with_clusters = pd.DataFrame({
            'datetime': ungrouped_signif_vals,
            'cluster': kmeans.labels_,
        })
       
        # cluster_centers = pd.to_datetime(kmeans.cluster_centers_.flatten())

        return data_with_clusters
    
            
    def get_sig_gradients(self, column, smoothing_window=50, threshold_factor_no_stds=2, max_clusters=10, pos_or_neg='both'):
        """
        Identifies clusters of steep gradients in reading data. Returns the timestamps of the earliest events in each cluster.
        
        Parameters:
        - `pos_or_neg` (string): Takes one of 'both', 'pos', or 'neg'. Determines whether all clusters are considered, or only positive / negative ones.

        Returns:
        - `cluster_beginnings` (pd.Series): Timestamps of the earliest events in each detected cluster, sorted chronologically.

        NB: You should filter SensorData object by time and reading type first
        """
        self.smooth_data(column=column, window_size=smoothing_window)
        self.get_gradient(column=(column+'_smoothed'))
        data_with_clusters = self.get_sig_clusters(column=(column+'_smoothed_grad'), threshold_factor_no_stds=threshold_factor_no_stds, max_clusters=max_clusters, pos_or_neg=pos_or_neg)
        cluster_beginnings = data_with_clusters.groupby('cluster')['datetime'].min().sort_values()
        return cluster_beginnings
    

    def get_occupancy(self):
        """
        Identifies an arbitrary value indicating the likelihood that the room is occupied.
        """

        if 'Re_smoothed' not in self.df:
            self.smooth_data(column="Re", window_size=150)
        if 'Lux_smoothed' not in self.df:
            self.smooth_data(column="Lux", window_size=150)

        self.df["occupancy_factor"] = np.where(
            self.df["Lux"] > 1,
            self.df["Re_smoothed"],
            self.df["Re_smoothed"] * 5 # More sensitive to Re if light is low
        )
        self.df["occupancy_factor"] = np.minimum(self.df["occupancy_factor"], 100)


    def get_features(self, window_size):
        """
        Method for generating 2D numpy array of features, used for clustering timeseries data.
        """
        # NB: The mean is placed on the right hand edge of the window
        Re_means = self.df['Re'].rolling(window=window_size).mean().to_numpy().reshape(-1,1)
        Re_medians = self.df['Re'].rolling(window=window_size).median().to_numpy().reshape(-1, 1)
        
        Rd_means = self.df['Rd'].rolling(window=window_size).mean().to_numpy().reshape(-1, 1)
        # Rd_medians = self.df['Rd'].rolling(window=window_size).median().to_numpy().reshape(-1, 1)

        # Me_means = self.df['Me'].rolling(window=window_size).mean().to_numpy().reshape(-1, 1)
        # Me_medians = self.df['Me'].rolling(window=window_size).median().to_numpy().reshape(-1, 1)

        # Md_means = self.df['Md'].rolling(window=window_size).mean().to_numpy().reshape(-1, 1)
        # Md_medians = self.df['Md'].rolling(window=window_size).median().to_numpy().reshape(-1, 1)

        # features = np.hstack([Re_means, Re_medians, Rd_means, Rd_medians, Me_means, Me_medians, Md_means, Md_medians])
        features = np.hstack([Re_means, Re_medians, Rd_means])

        return features
    
    def cluster_timeseries(self, n_clusters=None, max_clusters = 5, window_size=120):
        """
        Label each timestamp entry with a cluster. Features used for clustering are defined
        in get_features(). If n_clusters is not specified, optimal number of clusters is determined automatically.
        """
        features = self.get_features(window_size=window_size)

        length_of_df = len(features)

        # Remove rows with NaN values
        features = features[~np.isnan(features).any(axis=1)]

        # Normalize the feature matrix so that clustering works correctly
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if n_clusters is None:
            best_n_clusters = self.get_optimal_clusters(max_clusters=max_clusters, features=features_scaled)
        else:
            best_n_clusters = n_clusters

        # Perform actual clustering using the optimal number of clusters
        kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)
        labels = kmeans.fit_predict(features_scaled) # Get cluster labels for each window

        # Concatenate the NaNs at the beginning of the original array to make the length of labels match the length of the df
        nan_array = np.full(length_of_df - len(labels), np.nan) 
        labels = np.concatenate((nan_array, labels))

        # Add 'cluster' labels back to self.df (order should match the correct indices)
        self.df['cluster'] = labels


    def get_fft(self, column, smoothing_window=25, sampling_interval=900, normalize=False): # Sampling interval is determined by the data. 900 seconds means one reading every 15 minutes
        """
        Set smoothing_window to None to use raw data
        Returns: fft_freqs, fft_values
        """
        if smoothing_window:
            self.smooth_data(column=column, window_size=smoothing_window)
            smoothed_data = self.df[column + "_smoothed"].to_numpy() # first lot of values are set to nan
            signal = smoothed_data[~np.isnan(smoothed_data)] # Remove NaN values
        else:
            signal = self.df[column].to_numpy()

        zero_mean_smoothed_data = signal - signal.mean() # Remove DC offset


        # N = len(self.df["datetime"].to_numpy()) # DON'T DO THIS! SINCE WE LATER CROP THE LENGTH OF THE DATA AFTER SMOOTHING

        N = len(zero_mean_smoothed_data) # Do this instead of above ^

        # Apply FFT
        fft_values = np.fft.rfft(zero_mean_smoothed_data)
        fft_freqs = np.fft.rfftfreq(N, d=sampling_interval)

        if normalize:
            fft_values = fft_values / N  # Normalize the FFT output by dividing by the number of samples

        return fft_freqs, fft_values
    

    def get_frequent_values(self, column, bins, threshold_factor=0.2, smoothing_window=None):
        """
        Used for determining common distance values that could correspond to a location in the room e.g. bed, desk

        Usually column = "Rd"

        threshold_factor: Size of significant peaks relative to the max peak height. Defaults to 20%

        Returns 
        """

        counts, bin_edges = np.histogram(self.df[column][~np.isnan(self.df[column])], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # Average of each of the bin edges

        peaks, properties = find_peaks(counts, height=0) # peaks contains indexes of heights, relative to bin_centers
        peak_heights = properties["peak_heights"]
        max_peak_height = peak_heights.max()
        threshold = max_peak_height * threshold_factor

        sig_peaks_idx = peaks[peak_heights > threshold] # get the sig peak indices, relative to bin_centers
        sig_peaks_x = bin_centers[sig_peaks_idx] # get the corresponding distances for these peaks (x axis)
        sig_peaks_heights = peak_heights[peak_heights > threshold]  # their corresponding heights

        return sig_peaks_x, sig_peaks_heights
    
    def visualise_location_of_temp_responses(self, column, smoothing_window=50, threshold_factor_no_stds=2, max_clusters=10):
        """
        Visualise the data values in 'column', plotted along with the clusters' start and end points.
        """
        self.smooth_data(column=column, window_size=smoothing_window)
        self.get_gradient(column=(column+'_smoothed'))
        grad_column = column+'_smoothed_grad'
        # Get the positive gradients that might correspond with the heater turning on
        clustered_grad_temp_data = self.get_sig_clusters(column=grad_column, threshold_factor_no_stds=threshold_factor_no_stds, max_clusters=max_clusters, pos_or_neg='pos')


        for i in range(0, clustered_grad_temp_data['cluster'].max()+1):
            start_of_cluster = clustered_grad_temp_data[(clustered_grad_temp_data['cluster'] == i)]['datetime'].min()
            end_of_cluster = clustered_grad_temp_data[(clustered_grad_temp_data['cluster'] == i)]['datetime'].max()

            if i == 0:
                plt.axvline(x=start_of_cluster, color='green', linestyle='--', linewidth=1, label="Start of steep gradient")
                plt.axvline(x=end_of_cluster, color='red', linestyle='--', linewidth=1, label="End of steep gradient")
            else:
                plt.axvline(x=start_of_cluster, color='green', linestyle='--', linewidth=1)
                plt.axvline(x=end_of_cluster, color='red', linestyle='--', linewidth=1)

        self.plot(column='T_smoothed', ylabel="Smoothed temperature", fontsize=14)



    def get_temp_response_series(self, column, smoothing_window=50, threshold_factor_no_stds=2, max_clusters=10, min_response_length=40, plot_responses=False, pos_or_neg='pos'):
        """
        Get all of the temperature responses when the heater turns on that meet the min_response_length criteria.

        Returns:
        - all_responses : An array containing the normalised time series temperature readings for each of the responses
        
        NB: the responses are not truncated, and so will likely be of varying lengths
        """
        self.smooth_data(column=column, window_size=smoothing_window)
        self.get_gradient(column=(column+'_smoothed'))
        grad_column = column+'_smoothed_grad'
        # Get the (positive) gradients that might correspond with the heater turning on/off
        data_with_clusters = self.get_sig_clusters(column=grad_column, threshold_factor_no_stds=threshold_factor_no_stds, max_clusters=max_clusters, pos_or_neg=pos_or_neg)
        # add gradient column to df to allow for future manipulation
        data_with_clusters = pd.merge(data_with_clusters, self.df, on='datetime', how='left') 

        all_responses = []
        initial_temps = []

        for i in range(0, data_with_clusters['cluster'].max()+1): # create CustomDataframe object for each response
            start_of_response = data_with_clusters[(data_with_clusters['cluster'] == i)].loc[data_with_clusters[data_with_clusters['cluster'] == i][grad_column].idxmax()]['datetime']
            end_of_response = data_with_clusters[(data_with_clusters['cluster'] == i)]['datetime'].max()
            response = self.filter_by_date(start_date=start_of_response, end_date=end_of_response, in_place=False) # Create response CustomDataframe object

            # Normalize datetime: make it relative to the start
            response.df['datetime'] = (response.df['datetime'] - start_of_response).dt.total_seconds()

            # Normalize temperature: make it relative to the initial value
            initial_temperature = response.df.iloc[0][column + '_smoothed']
            response.df[column+'_smoothed'] = response.df[column + '_smoothed'] - initial_temperature

            response_array = response.df[column+'_smoothed'].to_numpy()
            if len(response_array) > min_response_length: # 40 readings here is 10 minutes (assuming 15s period)
                all_responses.append(response.df[column+'_smoothed'].to_numpy()[0:min_response_length])
                initial_temps.append(initial_temperature) # Add initial temps as well for colour-coding

            if plot_responses:
            # Plot individual responses
                response.plot(column=(column+'_smoothed'), ylabel="Normalised temperature", xlabel="Normalised time (starts at max gradient)", show=False)
        return all_responses, initial_temps
    
    def get_avg_temp_response_distrib(self, all_responses):
        """
        Get mean and standard deviation of all_responses.
        """
        # Calculate mean and std of response (to get distribution over responses)
        mean_response = np.mean(all_responses, axis=0)
        std_response = np.std(all_responses, axis=0)
        return mean_response, std_response
        
    def get_temp_curves_for_training(self, column, pos_or_neg, max_exceptions, response_duration, smoothing_window=50, plot_sequences=False):
        """
        Returns an array of arrays, each subarray containing a pair of timestamps.
        e.g. [[19/12/24 13:00, 19/12/24 17:00], [19/12/24 21:00, 20/12/24 2:00]]
        These timestamps are the start and end points of temperature decay / increase curves.
        There are defined as places where the smoothed gradient of the temperature readings is consistently negative/positive.
        Used to get training data for RNN to characterise a room's heating/cooling curvess.

        See 15_dec_rnn.py for use case.
        """
        self.smooth_data(column=column, window_size=smoothing_window)
        self.get_gradient(column=(column+'_smoothed'))
        grad_column = column+'_smoothed_grad'

        all_sequences = []
        current_sequence = []
        exception_count = 0
        for _, row in self.df.iterrows(): # Iterate through all the rows (readings)
            if pd.notna(row[grad_column]): # Ignore rows that don't have a smoothed temperature reading in them (there will be 50 initial rows that are not included if smoothing_window = 50
                if len(current_sequence) == 0:
                    current_sequence = [row['datetime'], row['datetime']] # initialise the sequence to start with

                if exception_count > max_exceptions: # If the gradient has substantially changed, then reset the current sequence
                    # duration_of_sequence = current_sequence[1] - current_sequence[0]
                    # duration_of_sequence = duration_of_sequence.total_seconds()
                    # if duration_of_sequence == response_duration:
                    #     all_sequences.append(current_sequence)
                    current_sequence = [row['datetime'], row['datetime']] # reset the current seuquence

                if (pos_or_neg == 'pos' and row[grad_column] >= 0) or (pos_or_neg == 'neg' and row[grad_column] <= 0):
                    # exception_count = 0 # TBC: at the moment we just reset the count, but could maybe decrement instead?
                    exception_count -= 1 if exception_count > -max_exceptions else 0 # decrement the counter
                    current_sequence[1] = row['datetime'] # extend the current sequence
                    duration_of_sequence = current_sequence[1] - current_sequence[0]
                    duration_of_sequence = duration_of_sequence.total_seconds()
                    if duration_of_sequence >= response_duration:
                        all_sequences.append(current_sequence)
                        current_sequence = [row['datetime'], row['datetime']] # reset the current seuquence
                else:
                    exception_count += 1
        
        if plot_sequences:
            i = 0
            self.plot(column='T_smoothed', show=False, ylabel="Temperature (smoothed) C")
            for sequence in all_sequences:
                if i == 0:
                    plt.axvline(x=sequence[0], color='green', linestyle='--', linewidth=1, label="Start of curve")
                    plt.axvline(x=sequence[1], color='red', linestyle='--', linewidth=1, label="End of curve")
                    i += 1
                else:
                    plt.axvline(x=sequence[0], color='green', linestyle='--', linewidth=1)
                    plt.axvline(x=sequence[1], color='red', linestyle='--', linewidth=1)
            plt.legend()
            plt.show()

        return all_sequences


    def add_ext_temp_column(self, lat, long):
        """
        Add a new column to dataframe with the external temperatures between the time stamps.
        Automatically determines the start_date and end_date based on the min and max date stamps.
        Performs interpolation to fill in all rows with T_ext.

        NB: Dates should be pd.date_time objects e.g. start_date = pd.to_datetime("10/11/2024 13:00")
        NB: Assumes that readings start from exactly on the hour
        """

        start_date = self.df.index.min()
        start_date = start_date.replace(second=0, microsecond=0, minute=0) # round back to prev hour
        end_date = self.df.index.max()
        end_date = end_date.replace(second=0, microsecond=0, minute=0, hour=end_date.hour+1) # round up to next hour

        hourly_extT_dataframe = get_external_temp(start_date, end_date, lat, long)

        measurement_period = self.df.index.freq

        new_time_index = pd.date_range(start=start_date, end=end_date, freq=measurement_period)  # 15-second intervals
        new_temp_df = pd.DataFrame(index=new_time_index)

        new_temp_df = new_temp_df.merge(hourly_extT_dataframe, left_index=True, right_index=True, how="left")

        # Perform linear interpolation to fill missing temperature values
        new_temp_df['temperature_2m'] = new_temp_df['temperature_2m'].interpolate(method='linear')

        self.df = pd.merge(self.df, new_temp_df, left_index=True, right_index=True, how="left")

    def add_sunrise_sunset_column(self, lat, long):
        start_date = self.df.index.min()
        start_date = start_date.replace(second=0, microsecond=0, minute=0) # round back to prev hour
        end_date = self.df.index.max()
        end_date = end_date.replace(second=0, microsecond=0, minute=0, hour=end_date.hour+1) # round up to next hour

        daily_suntimes_df = get_sunrise_sunset(start_date=start_date, end_date=end_date, lat=lat, long=long)

        daily_suntimes_df['sunrise'] = daily_suntimes_df['sunrise'].dt.tz_localize('UTC')
        daily_suntimes_df['sunset'] = daily_suntimes_df['sunset'].dt.tz_localize('UTC')

        # Extract the date part from the timestamps
        self.df['date'] = self.df.index.floor('D')
        daily_suntimes_df['date'] = daily_suntimes_df['date'].dt.floor('D')

        # Merge the dataframes on the date
        # Reset index to move datetime into a regular column
        self.df.reset_index(inplace=True)
        self.df = self.df.merge(daily_suntimes_df, on='date', how='left')
        # Reset the index to datetime
        self.df.set_index('datetime', inplace=True)

        # Check if the timestamp is between sunrise and sunset
        self.df['daylight'] = self.df.apply(
            lambda row: 1 if row['sunrise'] <= row.name <= row['sunset'] else 0, axis=1
        )

        # Drop the extra columns
        self.df.drop(columns=['date', 'sunrise', 'sunset'], inplace=True)

    def interpolate_missing_rows(self, measurement_period='15s'):
        # Create rows where there are missing values, and interpolate their values automatically
        self.df = self.df.set_index('datetime')
        self.df = self.df.resample(measurement_period).asfreq()  # Create 15-second gaps
        self.df = self.df.interpolate(method="linear")
        self.df = self.df.reset_index()

    def add_24hr_encoding(self):
        # Convert to hours, minutes, and seconds
        times = self.df.index.to_numpy(dtype="datetime64[s]")
        hours = np.array([t.astype("datetime64[h]").astype(int) % 24 for t in times])
        minutes = np.array([t.astype("datetime64[m]").astype(int) % 60 for t in times])
        seconds = np.array([t.astype("datetime64[s]").astype(int) % 60 for t in times])
        # Compute the time fraction of a 24-hour cycle
        time_fraction = (hours + minutes / 60 + seconds / 3600) / 24
        # Compute sine and cosine encoding
        sin_24hr = np.sin(2 * np.pi * time_fraction).reshape(-1, 1)
        cos_24hr = np.cos(2 * np.pi * time_fraction).reshape(-1, 1)
        self.df['sin24hr'] = sin_24hr
        self.df['cos24hr'] = cos_24hr



    def create_pytorch_matrix(self, lat, long):
        # # Add external temperature to sensor_data object
        # self.add_ext_temp_column(lat=lat, long=long)
        # # Add sunrise and sunset column (ensure this is done AFTER interpolation, since it is binary 0-1)
        # self.add_sunrise_sunset_column(lat=lat, long=long)

        iat = self.df["T"].to_numpy().reshape(-1, 1)
        eat = self.df["temperature_2m"].to_numpy().reshape(-1, 1)
        control = self.df["C"].to_numpy().reshape(-1, 1)
        re = self.df["Re"].to_numpy().reshape(-1, 1)
        daylight = self.df["daylight"].to_numpy().reshape(-1, 1)

        # Now to add sin/cos time encoding
        # Compute time fraction of the 24-hour cycle
        # Convert to hours, minutes, and seconds
        times = self.df.index.to_numpy(dtype="datetime64[s]")
        hours = np.array([t.astype("datetime64[h]").astype(int) % 24 for t in times])
        minutes = np.array([t.astype("datetime64[m]").astype(int) % 60 for t in times])
        seconds = np.array([t.astype("datetime64[s]").astype(int) % 60 for t in times])
        # Compute the time fraction of a 24-hour cycle
        time_fraction = (hours + minutes / 60 + seconds / 3600) / 24

        # Compute sine and cosine encoding
        sin_24hr = np.sin(2 * np.pi * time_fraction).reshape(-1, 1)
        cos_24hr = np.cos(2 * np.pi * time_fraction).reshape(-1, 1)

        matrix = np.hstack((iat, eat, control, re, daylight, sin_24hr, cos_24hr))
        return matrix
    
    def resample(self, freq='5Min'):
        self.df.set_index('datetime', inplace=True)
        # self.df = self.df.resample(freq, label="right").agg({
        #     "T": "mean",
        #     "C": "mean", # Could use last? not sure
        #     "Re": "mean",
        #     "H": "mean",
        #     "Dd": "mean",
        #     "Lux": "mean"
        # })
        self.df = self.df.resample(freq, label="right").agg({
            "T": "mean",
            "C": "mean", # Could use last? not sure
            "Re": "mean",
            "Lux": "mean"
        })