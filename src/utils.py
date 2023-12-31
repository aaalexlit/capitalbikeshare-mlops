import os
from pathlib import Path

import numpy as np
import scipy as sp
import joblib
import xgboost as xgb
from prefect import task
from sklearn.metrics import mean_squared_error
from prefect.blocks.system import Secret  # pylint: disable=ungrouped-imports

import wandb

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


def feature_dtypes() -> dict:
    return {
        'start_station_id': 'str',
        'end_station_id': 'str',
        'rideable_type': 'str',
        'member_casual': 'str',
    }


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
def load_pickle(file_path: Path) -> object:
    with open(file_path, "rb") as f_in:
        return joblib.load(f_in)


@task
def dump_pickle(obj, file_path: Path) -> None:
    with open(file_path, "wb") as f_out:
        joblib.dump(obj, f_out)


def set_wandb_api_key():
    if not os.getenv('WANDB_API_KEY'):
        os.environ['WANDB_API_KEY'] = Secret.load('wandb-api-key').get()


def calculate_rmse(
    booster: xgb.Booster,
    y_true: np.ndarray,
    X: sp.sparse.csr_matrix,
    convert: bool = True,
) -> float:
    X = convert_to_dmatrix(X) if convert else X
    y_pred = booster.predict(
        X, validate_features=False, iteration_range=(0, booster.best_iteration)
    )
    return mean_squared_error(y_true, y_pred, squared=False)


def convert_to_dmatrix(
    X: sp.sparse.csr_matrix,
    y: np.ndarray = None,
    feature_names: np.ndarray = None,
) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, feature_names=feature_names)


def log_val_preds_table(
    table_name: str,
    booster,
    val: sp.sparse.csr_matrix | xgb.DMatrix,
    y_val: np.ndarray,
):
    preds_artifact = wandb.Artifact(table_name, type='predictions')
    val_preds = booster.predict(
        val, iteration_range=(0, booster.best_iteration + 1)
    )
    val_preds_table = wandb.Table(
        columns=["y_val_true", "y_val_preds"], data=list(zip(y_val, val_preds))
    )
    preds_artifact.add(val_preds_table, name="preds vs true for val set")
    wandb.log_artifact(preds_artifact)
