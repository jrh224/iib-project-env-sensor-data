## 10/10/24: Used for dealing with the second set of data sent to me by Tash on 20th August
## Particularly with the CSV file DataSorted_2024-08-08-09-09__SENS_36DC40metdata_combined

import os

from matplotlib import pyplot as plt
import pandas as pd
from datetime import timedelta


class CombinedData:
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

        # Drop rows which don't have an internal T reading (the first ~2200 rows)
        self.df.dropna(subset="T", inplace=True)

        # Rename the datetime column
        self.df.rename(columns={"Unnamed: 0":"datetime"}, inplace=True)

        # Convert datetime column to datetime
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], errors='coerce')

    def filter_by_date(self, start_date=None, end_date=None, days=None):
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

        return CombinedData(df=self.df[(self.df['datetime'] >= start_date) & (self.df['datetime'] <= end_date)])

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

