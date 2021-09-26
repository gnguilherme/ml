import warnings
from typing import Union
from urllib.parse import urlparse
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.sparse import data

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn


RANDOM_STATE = 42


def parse_args():
    description = """
    Train a simple or multiple linear regression with sklearn library
    """
    parser = ArgumentParser(description=description)

    feature_help = """
    Feature to use. If None, a multiple linear regression will be trained with all
    features. Default: None
    """
    parser.add_argument('-f', '--feature', default=None, type=Union[None, str],
        help=feature_help)
    parser.add_argument('-n', '--normalize', default=False, action='store_true',
        help="Normalize? Default: False")

    return parser.parse_args()


def load_dataset():
    X, y = load_boston(return_X_y=True)

    return train_test_split(X, y, random_state=RANDOM_STATE)


def mount_dataframe(dataset : list, target: str = 'MEDV'):
    X_train, X_test, y_train, y_test = dataset
    columns = [
        'CRIM',
        'ZN',
        'INDUS',
        'CHAS',
        'NOX',
        'RM',
        'AGE',
        'DIS',
        'RAD',
        'TAX',
        'PTRATIO',
        'B',
        'LSTAT',
    ]
    X_train = pd.DataFrame(X_train, columns=columns)
    X_test = pd.DataFrame(X_test, columns=columns)

    y_train = pd.DataFrame(y_train, columns=[target])
    y_test = pd.DataFrame(y_test, columns=[target])

    return X_train, X_test, y_train, y_test


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = parse_args()

    dataset = load_dataset()
    X_train, X_test, y_train, y_test = mount_dataframe(dataset=dataset)

    input_feature = args.feature
    input_feature = args.feature
    if input_feature:
        feat = X_train[[input_feature]]
    else:
        feat = X_train

    with mlflow.start_run():
        folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        mlflow.log_param("random_state", RANDOM_STATE)
        if args.normalize:
            mlflow.log_param("normalize", True)
        
        for fold, (train_index, validation_index) in enumerate(folds.split(X_train, y_train)):
            lr = LinearRegression()
            lr.fit(feat.iloc[train_index, :], y_train.iloc[train_index, :])

            y_pred = lr.predict(feat.iloc[validation_index, :])

            (rmse, mae, r2) = eval_metrics(y_train.iloc[validation_index, :], y_pred)

            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2: {r2}")
            
            mlflow.log_metric(f'rmse-fold{fold}', rmse)
            mlflow.log_metric(f'r2-fold{fold}', r2)
            mlflow.log_metric(f'mae-fold{fold}', mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="LinearModelBoston")
        else:
            mlflow.sklearn.log_model(lr, "model")