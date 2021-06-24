"""
Reconstruction of missing periodic data.
If a missing period is shorter than the ranges[0] - linear interpolation
If a missing period is > ranges[0] and < ranges[1] - linear interpolation grouped by hours of the day
If a missing period is longer than the ranges[1] - pass
"""
import pandas as pd


def get_deltas(data: pd.DataFrame, freq: pd.Timedelta):
    """
    :param data: (pd.DataFrame) data containing "y" as a target and "dt" as datetime columns
    :param freq: (pd.Timedelta) time series step
    :return deltas: (pd.DataFrame) containing timestamps of missed periods (start point) and their duration
    """
    data = data.groupby('dt').mean().reset_index().dropna(subset=['y'])
    deltas = pd.DataFrame([])
    deltas['dt'] = data['dt']
    deltas['diff'] = data['dt'] - data['dt'].shift()
    deltas = deltas[deltas['diff'] != freq][1:]
    deltas.index = (deltas.index - deltas['diff'] / freq).astype(int)
    deltas['dt'] = deltas['dt'] - deltas['diff'] + freq
    deltas['steps'] = (deltas['diff'] / freq - 1).astype(int)
    deltas['diff'] = deltas['diff'] - freq
    return deltas


def get_interpolation(data: pd.DataFrame, start: pd.Timestamp, steps: int, freq: pd.Timedelta):
    """
    :param data: (pd.DataFrame) data containing "y" as a target and "dt" as datetime columns
    :param start: (pd.Timestamp) start of the missing period
    :param steps: (int) number of missed steps
    :param freq: (pd.Timedelta) time series step
    :return result: (pd.DataFrame) interpolated data for certain missing period
    """
    result = pd.DataFrame([])
    for i in range(0, steps):
        date = start + freq * i
        missed = pd.DataFrame([])
        dt_range = pd.date_range(date - pd.Timedelta('1d') - pd.Timedelta('1d') * (freq * i).days,
                                 date + pd.Timedelta('1d') + pd.Timedelta('1d') * (freq * (steps - i)).days,
                                 freq='1d')
        missed['dt'] = dt_range
        for point in dt_range:
            missed.loc[missed['dt'] == point, 'y'] = data.loc[data['dt'] == point, 'y'].values[0]
        missed['y'] = missed['y'].interpolate(method='linear')
        result = pd.concat([result, missed])
    result = result.sort_values('dt')
    result = result.drop_duplicates('dt')
    result.reset_index(drop=True, inplace=True)
    return result


def fill_missing(data: pd.DataFrame, freq: str, ranges=None):
    """
    :param data: (pd.DataFrame) containing "y" as a target and "dt" as datetime columns
    :param freq: (str) time series step
    :param ranges: (list) list of string representations of pf.Timedelta. 1st - short period, 2nd - long period.
    :return data: (pd.DataFrame) data with filled missing values
    :return deltas: (pd.DataFrame) timestamps of missed periods (start point) and their duration
    """
    freq = pd.Timedelta(freq)
    if ranges is None:
        ranges = ['3h', '3d']
    short_range = pd.Timedelta(ranges[0])
    long_range = pd.Timedelta(ranges[1])
    deltas = get_deltas(data, freq)
    for _, row in deltas[deltas['diff'] <= short_range].iterrows():
        start = row['dt'] - freq
        end = row['dt'] + (row['steps'] + 1) * freq
        data.loc[(data['dt'] >= start) & (data['dt'] < end), 'y'] = data.loc[
            (data['dt'] >= start) & (data['dt'] < end), 'y'].interpolate(method='linear')
    for _, row in deltas[(deltas['diff'] > short_range) & (deltas['diff'] <= long_range)].iterrows():
        start = row['dt']
        steps = row['steps']
        interp = get_interpolation(data, start, steps, freq)
        for time in interp['dt']:
            data.loc[data['dt'] == time, 'y'] = interp.loc[interp['dt'] == time, 'y'].values[0]
    return data, deltas
