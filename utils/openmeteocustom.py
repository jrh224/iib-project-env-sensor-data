# 18th December - This 

import json
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry



def get_external_temp(start_date, end_date, lat, long):
	"""
	dates should be datetime objects
	"""

	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	start_date=start_date.strftime('%Y-%m-%d')
	end_date=end_date.strftime('%Y-%m-%d')

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
	params = {
		"latitude": lat, # Wadham College, Oxford
		"longitude": long,
		# "start_date": "2024-12-03",
		"start_date": start_date,
		# "end_date": "2024-12-16",
		"end_date": end_date,
		"hourly": "temperature_2m",
		"timezone": "GMT" # CHECK THIS WITH DAYLIGHT SAVINGS CHANGES
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

	hourly_data = {"date": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}
	hourly_data["temperature_2m"] = hourly_temperature_2m

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])  # Convert date column to datetime
	hourly_dataframe.set_index('date', inplace=True)

	return hourly_dataframe


def interpolate_ext_temp(hourly_dataframe, start_date, end_date, sensor_data=None):
	"""
	dates should be pd.date_time objects
	e.g. start_date = pd.to_datetime("10/11/2024 13:00")
	"""
	hourly_dataframe['date'] = hourly_dataframe['date'].dt.tz_localize(None)

	new_time_index = pd.date_range(start=start_date, end=end_date, freq='15S')  # 15-second intervals
	new_temp_df = pd.DataFrame(new_time_index, columns=["datetime"])
	new_temp_df['datetime'] = new_temp_df['datetime'].dt.tz_localize(None)

	new_temp_df = pd.merge(new_temp_df, hourly_dataframe, left_on='datetime', right_on='date', how='left')
	new_temp_df = new_temp_df.drop('date', axis=1)

	# Perform linear interpolation to fill missing temperature values
	new_temp_df['temperature_2m'] = new_temp_df['temperature_2m'].interpolate(method='linear')

	print(new_temp_df)

	if sensor_data is None:
		return new_temp_df

	sensor_data.df['datetime'] = sensor_data.df['datetime'].dt.tz_localize(None)
	sensor_data.df = pd.merge(sensor_data.df, new_temp_df, left_on="datetime", right_on="datetime", how="left")


def get_sunrise_sunset(start_date, end_date, lat, long):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	start_date=start_date.strftime('%Y-%m-%d')
	end_date=end_date.strftime('%Y-%m-%d')

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": lat,
		"longitude": long,
		"start_date": start_date,
		"end_date": end_date,
		"daily": ["sunrise", "sunset"],
		"timezone": "GMT" # CHECK THIS WITH DAYLIGHT SAVINGS CHANGES
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()

	daily_sunrise = daily.Variables(0).ValuesInt64AsNumpy()
	daily_sunset = daily.Variables(1).ValuesInt64AsNumpy()

	daily_data = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}

	daily_data["sunrise"] = daily_sunrise
	daily_data["sunset"] = daily_sunset

	daily_data['sunrise'] = pd.to_datetime(daily_data['sunrise'], unit='s')
	daily_data['sunset'] = pd.to_datetime(daily_data['sunset'], unit='s')

	daily_suntimes_df= pd.DataFrame(data = daily_data)
	return daily_suntimes_df