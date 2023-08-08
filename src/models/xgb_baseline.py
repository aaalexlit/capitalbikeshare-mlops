from pathlib import Path

import numpy as np
import joblib
import pandas as pd
import xgboost as xgb
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer

import wandb
from wandb.xgboost import WandbCallback
import src.wandb_params as wandb_params
from src.utils import dump_pickle, load_pickle, get_models_dir

load_dotenv(find_dotenv())


def convert_to_dmatrix(X: pd.DataFrame, y: np.ndarray, dv: DictVectorizer) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y, feature_names=dv.get_feature_names_out())


def calculate_rmse(booster: xgb.Booster, y_true: np.ndarray, features: xgb.DMatrix) -> float:
    y_pred = booster.predict(features)
    return mean_squared_error(y_true, y_pred, squared=False)


@task(log_prints=False)
def train_booster(params, train: xgb.DMatrix, val: xgb.DMatrix) -> xgb.Booster:
    return xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(val, 'validation')],
        early_stopping_rounds=50,
        callbacks=[WandbCallback(log_model=True)],
        verbose_eval=False,
    )


@flow(name="train baseline model", log_prints=True)
def train_xgboost():
    print("Training model...")
    xgb_params = {
        'objective': 'reg:squarederror',
        'seed': 42,
        'nthread': 4,
    }

    with wandb.init(project=wandb_params.WANDB_PROJECT,
                    entity=wandb_params.ENTITY,
                    job_type="train",
                    config=xgb_params,) as wandb_run:

        print("Downloading data...")
        artifact = wandb_run.use_artifact('aaalex-lit/capitalbikeshare-mlops/202304-202305-202306-processed-data:latest',
                                          type='processed_data')
        artifact_dir = Path(artifact.download())

        dv = load_pickle(artifact_dir / 'dv.pkl')
        train = convert_to_dmatrix(*load_pickle(artifact_dir / 'train.pkl'), dv)
        X_val, y_val = load_pickle(artifact_dir / 'val.pkl')
        val = convert_to_dmatrix(X_val, y_val, dv)
        X_test, y_test = load_pickle(artifact_dir / 'test.pkl')
        test = convert_to_dmatrix(X_test, y_test, dv)

        print("Training model...")
        booster = train_booster(xgb_params, train, val)

        wandb_run.log({'test RMSE': calculate_rmse(booster, y_test, test)})

        print("Saving model locally...")
        model_path = get_models_dir() / 'booster.pkl'
        dump_pickle(booster, model_path)


if __name__ == "__main__":
    train_xgboost()
