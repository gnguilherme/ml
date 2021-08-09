"""
Naive Forecaster test - sktime

See NaiveForecaster estimator strategies:
- 'first'
- 'mean'
- 'drift'
"""

import pandas as pd
import matplotlib.pyplot as plt
from sktime.datasets import load_airline
from sktime.utils.plotting import plot_series
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.base import ForecastingHorizon

if __name__ == '__main__':
    for spine in ['right', 'top']:
        plt.rcParams[f'axes.spines.{spine}'] = False
    plt.rcParams['legend.frameon'] = False

    y = load_airline()

    data_range = pd.date_range('1961-01', periods=36, freq='M')
    fh = ForecastingHorizon(pd.PeriodIndex(data_range), is_relative=False)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    strategies = ['last', 'mean', 'drift']
    plt.suptitle("Number of airline passengers")
    for i, strategy in enumerate(strategies):
        forecaster = NaiveForecaster(strategy=strategy)
        forecaster.fit(y)

        y_pred = forecaster.predict(fh)

        plot_series(y, y_pred, labels=['y', 'y_pred'], ax=axes[i])
        axes[i].set_xlabel(strategy)
        axes[i].set_ylabel('')

    plt.show()
