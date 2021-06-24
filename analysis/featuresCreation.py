import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def add_periodic_features(spectrum: pd.DataFrame, data: pd.DataFrame, top: int):
    """
    :param spectrum: (pd.DataFrame) data containing frequencies and amplitudes of signals to be reconstructed
    :param data: (pd.DataFrame) input data
    :param top: (int) number of top intensive frequencies to be used
    :return data: (pd.DataFrame) data with extra periodic features
    """
    array = np.arange(len(data))
    spectrum = spectrum[spectrum['peak'] == 1]
    spectrum = spectrum.sort_values(by='abs', ascending=False).head(int(top))
    i = 1
    for freq in spectrum['freq'].unique():
        data['freq' + str(i) + '_sin'] = np.sin(2 * np.pi * freq * array)
        data['freq' + str(i) + '_cos'] = np.cos(2 * np.pi * freq * array)
        i += 1
    return data


def add_datetime_features(data: pd.DataFrame):
    """
    :param data: (pd.DataFrame) input data
    :return data: (pd.DataFrame) data with extra datetime features
    """
    data['dt'] = pd.to_datetime(data['dt'])
    dt = data['dt'].dt
    # hours.minutes.seconds into circle coordinates hour_x and hour_y
    data['time_sin'] = np.sin(2. * np.pi * (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.)
    data['time_cos'] = np.cos(2. * np.pi * (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.)
    # days of week into circle coordinates day_month_x and day_month_y
    data['day_of_month_sin'] = np.sin(2. * np.pi * dt.day / dt.daysinmonth)
    data['day_of_month_cos'] = np.cos(2. * np.pi * dt.day / dt.daysinmonth)
    # days of week into circle coordinates day_week_x and day_week_y
    data['day_of_week_sin'] = np.sin(2. * np.pi * dt.dayofweek / 7.)
    data['day_of_week_cos'] = np.cos(2. * np.pi * dt.dayofweek / 7.)
    # month into circle coordinates month_x and month_y
    data['month_sin'] = np.sin(2. * np.pi * dt.month / 12.)
    data['month_cos'] = np.cos(2. * np.pi * dt.month / 12.)
    return data


def add_lagged_features(data: pd.DataFrame, lags: list, freq=None):
    """
    :param data: (pd.DataFrame) input data
    :param lags: (list of int) list of time series steps-lags
    :param freq: (str) string representation (pd.Timedelta format) of a time series step
    :return data: (pd.DataFrame) data with extra lagged features
    """
    if freq is None:
        for lag in lags:
            data[f'lag_{lag}'] = data['y'].shift(lag).values
    else:
        data = data.set_index('dt')
        for lag in lags:
            shift = data['y'].shift(lag, freq).rename(f'lag_{lag}_{freq}')
            data = data.join(shift)
        data = data.reset_index()
    return data


def permutation_features_importance(data: pd.DataFrame, target: str):
    """
    :param data: (pd.DataFrame) input data
    :param target: (str) target column name
    :return FI: (pd.DataFrame) resulting table describing importance of each column
    """
    data['RAND_bin'] = np.random.randint(2, size=len(data[target]))
    data['RAND_uniform'] = np.random.uniform(0, 1, len(data[target]))
    data['RAND_int'] = np.random.randint(100, size=len(data[target]))
    columns = data.drop(target, axis=1).columns.tolist()
    estimator = RandomForestRegressor(n_estimators=50)
    estimator.fit(data[columns], data[target])
    y_pred = estimator.predict(data[columns])
    baseline = np.mean(np.fabs(y_pred - data[target]))
    imp = []
    for col in columns:
        col_imp = []
        for n in range(3):
            save = data[col].copy()
            data[col] = np.random.permutation(data[col])
            y_pred = estimator.predict(data[columns])
            m = np.mean(np.fabs(y_pred - data[target]))
            data[col] = save
            col_imp.append(baseline - m)
        imp.append(np.mean(col_imp))
    FI = pd.DataFrame([])
    FI['feature'] = columns
    FI['value'] = -np.array(imp)
    FI = FI.sort_values(by='value', ascending=False).reset_index(drop=True)
    Max = FI[FI['feature'].isin(['RAND_bin', 'RAND_int', 'RAND_uniform'])]['value'].max()
    Std = FI[FI['feature'].isin(['RAND_bin', 'RAND_int', 'RAND_uniform'])]['value'].std()
    threshold = Max + Std
    FI['important'] = np.where(FI['value'] > threshold, True, False)
    return FI
