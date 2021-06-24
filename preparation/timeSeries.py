"""
Create and manage time series data data
"""
import pandas as pd


def create_time_series(data: pd.DataFrame, freq='auto'):
    """
    :param data: (pd.DataFrame) data containing "dt" date/time column in a pandas-supported format
    :param freq: (str) time series step. "auto" for automatic estimation
    :return t_series: (pd.DataFrame) time series data.
    :return freq: (str) time series step (in seconds).
    """
    data['dt'] = pd.to_datetime(data['dt'])
    data = data.groupby('dt').mean().reset_index()
    if freq == 'auto':
        freq = int(data['dt'].diff().dt.total_seconds().mode().values[0])
        freq = str(freq) + 's'
    t_series = pd.DataFrame([])
    t_series['dt'] = pd.date_range(start=data['dt'].min(),
                                   end=data['dt'].max(),
                                   freq=freq)
    t_series = t_series.merge(data, how='left', on='dt')
    return t_series, freq
