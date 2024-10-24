## 9th Oct: Defines SensorData class    
## THIS CLASS IS DEPRECATED - see CustomDataframe.py instead

from datetime import timedelta
import os
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class SensorData:
    def __init__(self, filename=None, df=None):
        self.filename = filename
        self.df = df
        if self.df is None:
            self.load_and_clean_data()

    def load_and_clean_data(self):
        """
        Run by __init__ to clean data, and load it from CSV file if a filename is provided rather than a df
        """
        # Get filepath for CSV data required
        root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
        # print(os.listdir(filepath))
        filepath = os.path.join(root_dir, "First_data_for_experimenting/", self.filename)

        df = pd.read_csv(filepath,
                        comment='#'
                        )
        
        # Drop rows that are completely empty (if any)
        df.dropna(how='all', inplace=True)

        # Convert the "value" column (assumed to be column 6) to a double (float)
        # If conversion fails, set those values to NaN
        df['_value'] = pd.to_numeric(df['_value'], errors='coerce')
        # Convert '_start', '_stop' and '_time' to datetime objects
        df['_start'] = pd.to_datetime(df['_start'], errors='coerce')
        df['_stop'] = pd.to_datetime(df['_stop'], errors='coerce')
        df['_time'] = pd.to_datetime(df['_time'], errors='coerce')

        # Drop rows where there are NaN values (i.e., failed conversions)
        df.dropna(subset=['_value', '_start', '_stop', '_time'], inplace=True)

        # Force pandas to re-infer the object types
        self.df = df.infer_objects()

    def filter_by_reading_type(self, reading_type):
        # Filter the DataFrame where _field == type
        return SensorData(df=self.df[self.df['_field'] == reading_type])

    def filter_by_date(self, start_date=None, end_date=None, days=None):
        """
        Provide either an end_date, or a number of days.
        """
        if (end_date is None and days is None) or (end_date is not None and days is not None):
            raise ValueError("Only one of 'end_date' and 'days' must be provided.")
        
        if start_date is None:
            start_date = self.df['_time'].min()
        else:
            start_date = pd.to_datetime(start_date)

        # Align start_date with the timezone of the `_time` column, if it has one
        if pd.api.types.is_datetime64tz_dtype(self.df['_time']):
            start_date = start_date.tz_localize(self.df['_time'].dt.tz) if start_date.tzinfo is None else start_date.tz_convert(self.df['_time'].dt.tz)
        
        if end_date is None:
            end_date = start_date + timedelta(days=days)
        else:
            end_date = pd.to_datetime(end_date)

        # Align end_date with the timezone of the `_time` column, if it has one
        if pd.api.types.is_datetime64tz_dtype(self.df['_time']):
            end_date = end_date.tz_localize(self.df['_time'].dt.tz) if end_date.tzinfo is None else end_date.tz_convert(self.df['_time'].dt.tz)

        return SensorData(df=self.df[(self.df['_time'] >= start_date) & (self.df['_time'] <= end_date)])

    def plot(self, title, xlabel, ylabel, label=None):
        # Plot the filtered data
        plt.scatter(self.df['_time'], self.df['_value'], label=label, s=10, alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        # Rotate and format date labels for better readability
        plt.gcf().autofmt_xdate()

    def smooth_data(self, window_size=50):
        """
        NB: This only works if data has already been filtered by reading type (see filter_by_reading_type() method)
        """
        # Apply a rolling window to smooth the 'y' values

        df = self.df.copy() # Creates a deep copy of the dataframe, to avoid editing the original df
        df['_value'] = df['_value'].rolling(window=window_size).mean()
        return SensorData(df=df)
    
    def get_gradient(self):
        df = self.df.copy() # Creates a deep copy of the dataframe, to avoid editing the original df
        df['dy'] = df['_value'].diff()
        df['dx'] = df['_time'].diff().dt.total_seconds()
        df['_value'] = df['dy'] / df['dx'] # Calculate running gradient
        return SensorData(df=df)
    
    def get_sig_clusters(self, threshold_factor_no_stds=2, max_clusters=10, neg_only=False):
        """
        Identifies clusters of significantly high values, returning the timestamps of the earliest events in each cluster.

        Parameters:
        - `threshold_factor_no_stds` (float, default=2): Multiplier for the standard deviation to define the threshold for significant negative values.
        - `max_clusters` (int, default=10): Maximum number of clusters to evaluate for optimal clustering.
        - `neg_only` (bool, default=False): If True, only consider periods of negative gradient i.e. corresponding with window open event

        Returns:
        - `cluster_beginnings` (pd.Series): Timestamps of the earliest events in each detected cluster, sorted chronologically.
        """
        threshold = self.df['_value'].std() * threshold_factor_no_stds
        # create 2D array of all timestamps of detected event
        if neg_only:
            ungrouped_signif_vals = pd.Series(self.df[self.df['_value'] < -threshold]['_time']) # filter dataframe based on value being negative and bigger mag than threshold
        else:
            ungrouped_signif_vals = pd.Series(self.df[abs(self.df['_value']) > threshold]['_time']) # filter df based on abs value being greater than threshold
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

    
    def get_sig_gradients(self, smoothing_window=50, threshold_factor_no_stds=2, max_clusters=10):
        """
        Identifies clusters of steep negative gradients in reading data. Returns the timestamps of the earliest events in each cluster.

        Returns:
        - `cluster_beginnings` (pd.Series): Timestamps of the earliest events in each detected cluster, sorted chronologically.

        NB: You should filter SensorData object by time and reading type first
        """
        smoothed_temp_data = self.smooth_data(window_size=smoothing_window)
        gradient_of_smoothed_temp = smoothed_temp_data.get_gradient()
        return gradient_of_smoothed_temp.get_sig_clusters(threshold_factor_no_stds=threshold_factor_no_stds, max_clusters=max_clusters, neg_only=True)

