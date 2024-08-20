from datetime import timedelta
import os
from matplotlib import pyplot as plt
import pandas as pd


class SensorData:
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

    def plot(self, title, xlabel, ylabel):
        # Plot the filtered data
        plt.scatter(self.df['_time'], self.df['_value'], label='Temp Field Data', s=10, alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        # Rotate and format date labels for better readability
        plt.gcf().autofmt_xdate()

    def smooth_data(self, window_size=3):
        """
        NB: This only works if data has already been filtered by reading type (see filter_by_reading_type() method)
        """
        # Apply a rolling window to smooth the 'y' values

        df = self.df
        df['_value'] = df['_value'].rolling(window=window_size).mean()
        return SensorData(df=df)
    
    def get_gradient(self):
        df = self.df
        df['dy'] = df['_value'].diff()
        df['dx'] = df['_time'].diff().dt.total_seconds()
        df['_value'] = df['dy'] / df['dx'] # Calculate running gradient
        return SensorData(df=df)
    
    def get_significant_values(self, threshold_factor_no_stds=2):
        df = self.df
        threshold = df['_value'].std() * threshold_factor_no_stds
        print(threshold)
        df['_value'] = df['_value'].abs() > threshold
        return SensorData(df=df)
    
    def plot_vertical_peaks(self):
        """
        NB: Use this on data where the values are either 0 or 1 (e.g. after running 'get_significant_values())
        """
        # Iterate over the DataFrame and plot vertical lines
        for _, row in self.df.iterrows():
            if row['_value'] == 1:
                plt.axvline(x=row['_time'], color='red', linestyle='--', linewidth=1)
