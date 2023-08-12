from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import xgboost as xgb
from prefect import task
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer

TARGET_COL = 'duration'


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    return get_project_root() / "data"


def get_models_dir() -> Path:
    return get_project_root() / "models"


def get_categorical_features() -> [str]:
    return [
        'start_station_id',
        'end_station_id',
        'rideable_type',
        'member_casual',
    ]


def get_year_months(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> ([int], [[int]]):
    """Get list of months for each year."""
    assert (
        start_year <= end_year
    ), f"start_year must be less than or equal to end_year, {start_year} > {end_year}"
    assert (
        1 <= start_month <= 12
    ), f"start_month must be between 1 and 12, not {start_month}"
    assert (
        1 <= end_month <= 12
    ), f"end_month must be between 1 and 12, not {end_month}"
    assert (
        start_year >= 2018
    ), "monthly information is only available from 2018 onwards"
    if start_year == end_year:
        assert (
            start_month <= end_month
        ), f"start_month of the same year {start_year} must be less than or equal to end_month"
        return [start_year], [list(range(start_month, end_month + 1))]
    return list(range(start_year, end_year + 1)), [
        list(range(start_month, 13)),
        *[list(range(1, 13)) for _ in range(start_year + 1, end_year)],
        *[list(range(1, end_month + 1))],
    ]


@task
def load_pickle(file_path: Path):
    with open(file_path, "rb") as f_in:
        return joblib.load(f_in)


@task
def dump_pickle(obj, file_path: Path) -> None:
    with open(file_path, "wb") as f_out:
        return joblib.dump(obj, f_out)


def calculate_rmse(
    booster: xgb.Booster, y_true: np.ndarray, features: xgb.DMatrix
) -> float:
    y_pred = booster.predict(features)
    return mean_squared_error(y_true, y_pred, squared=False)


def convert_to_dmatrix(
    X: pd.DataFrame, y: np.ndarray, dv: DictVectorizer
) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, feature_names=dv.get_feature_names_out())
