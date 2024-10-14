# 13th Oct: Combines deprecated classes SensorData and CombinedData

from datetime import timedelta
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
        
    def plot(self, column, title=None, label=None, xlabel="Time", ylabel="Value"):
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

        
    def smooth_data(self, column, window_size=50, in_place=True):
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

        # Determine the optimal number of clusters using silhouette score
        best_n_clusters = 2
        best_score = -1
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters).fit(ungrouped_data_reshaped)
            labels = kmeans.labels_
            score = silhouette_score(ungrouped_data_reshaped, labels)
            
            if score > best_score:
                best_n_clusters = n_clusters
                best_score = score
        
        print(f"Optimal number of clusters: {best_n_clusters}")

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