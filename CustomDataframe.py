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
        
    def plot(self, column, title=None, label=None, xlabel="Time", ylabel="Value", show=True):
        # Plot the filtered data
        plt.scatter(self.df['datetime'], self.df[column], label=label, s=10, alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.subplots_adjust(right=0.7)
        plt.grid(True)
        # Rotate and format date labels for better readability
        plt.gcf().autofmt_xdate()

        if show:
            plt.show()


    def plot_dual(self, column1, column2, title=None, xlabel="Time", ylabel1 = "Value", ylabel2="Value", show=True):
        fig, ax1 = plt.subplots()
        ax1.set_title(title)
        ax1.grid(True)
        
        color = 'tab:red'
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel1, color=color)
        ax1.scatter(self.df['datetime'], self.df[column1], s=10, alpha=0.7, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel(ylabel2, color=color)  # we already handled the x-label with ax1
        ax2.scatter(self.df['datetime'], self.df[column2], s=10, alpha=0.7, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

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
        
    def get_sig_clusters(self, column, threshold_factor_no_stds=2, max_clusters=10, neg_only=False):
        """
        Identifies clusters of significantly high values, returning the timestamps of the earliest events in each cluster.
        NB: Used by get_sig_gradients. Usually get_sig_clusters is not called directly.

        Parameters:
        - `column` (str): The target column to find the clusters of
        - `threshold_factor_no_stds` (float, default=2): Multiplier for the standard deviation to define the threshold for significant negative values.
        - `max_clusters` (int, default=10): Maximum number of clusters to evaluate for optimal clustering.
        - `neg_only` (bool, default=False): If True, only consider periods of negative gradient i.e. corresponding with window open event

        Returns:
        - `cluster_beginnings` (pd.Series): Timestamps of the earliest events in each detected cluster, sorted chronologically.
        """
        threshold = self.df[column].std() * threshold_factor_no_stds
        # create 2D array of all timestamps of detected event
        if neg_only:
            ungrouped_signif_vals = pd.Series(self.df[self.df[column] < -threshold]['datetime']) # filter dataframe based on value being negative and bigger mag than threshold
        else:
            ungrouped_signif_vals = pd.Series(self.df[abs(self.df['datetime']) > threshold]['datetime']) # filter df based on abs value being greater than threshold
        ungrouped_data_reshaped = ungrouped_signif_vals.values.reshape(-1, 1)

        best_n_clusters = self.get_optimal_clusters(max_clusters=max_clusters, features=ungrouped_data_reshaped)

        kmeans = KMeans(best_n_clusters).fit(ungrouped_data_reshaped)
        data_with_clusters = pd.DataFrame({
            'timestamps': ungrouped_signif_vals,
            'cluster': kmeans.labels_,
        })
        cluster_beginnings = data_with_clusters.groupby('cluster')['timestamps'].min().sort_values()
        # cluster_centers = pd.to_datetime(kmeans.cluster_centers_.flatten())

        return cluster_beginnings
            
    def get_sig_gradients(self, column, smoothing_window=50, threshold_factor_no_stds=2, max_clusters=10):
        """
        Identifies clusters of steep negative gradients in reading data. Returns the timestamps of the earliest events in each cluster.

        Returns:
        - `cluster_beginnings` (pd.Series): Timestamps of the earliest events in each detected cluster, sorted chronologically.

        NB: You should filter SensorData object by time and reading type first
        """
        self.smooth_data(column=column, window_size=smoothing_window)
        self.get_gradient(column=(column+'_smoothed'))
        return self.get_sig_clusters(column=(column+'_smoothed_grad'), threshold_factor_no_stds=threshold_factor_no_stds, max_clusters=max_clusters, neg_only=True)
    

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
        Rd_medians = self.df['Rd'].rolling(window=window_size).median().to_numpy().reshape(-1, 1)

        Me_means = self.df['Me'].rolling(window=window_size).mean().to_numpy().reshape(-1, 1)
        Me_medians = self.df['Me'].rolling(window=window_size).median().to_numpy().reshape(-1, 1)

        Md_means = self.df['Md'].rolling(window=window_size).mean().to_numpy().reshape(-1, 1)
        Md_medians = self.df['Md'].rolling(window=window_size).median().to_numpy().reshape(-1, 1)

        features = np.hstack([Re_means, Re_medians, Rd_means, Rd_medians, Me_means, Me_medians, Md_means, Md_medians])

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

