import pandas as pd
import numpy as np
import logging

def feature_eng(df):

    # Sub-sampling the data from 10-minute intervals to one-hour intervals: Slice [start:stop:step]
    df = df[5::6]
    #df.pop('Date Time') returns Date Time values and removes the Date Time column itself
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    #creating Cartesian coordiantes for Wind
    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)')*np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv*np.cos(wd_rad)
    df['max Wy'] = max_wv*np.sin(wd_rad)

    # Create Cartesian cooridantes for Time features
    # time in seconds
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df

if __name__ == "__main__":
    df = pd.read_csv("../data/jena_climate_2009_2016.csv")
    print(df.head())
    df = feature_eng(df)
    print(df.columns)