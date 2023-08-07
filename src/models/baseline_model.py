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
import src.wandb_params as wandb_params
from src.utils import dump_pickle, load_pickle, get_models_dir

load_dotenv(find_dotenv())


def convert_to_dmatrix(X: pd.DataFrame, y: np.ndarray) -> xgb.DMatrix:
    return xgb.DMatrix(X, label=y)


def calculate_rmse(booster: xgb.Booster, y_true: np.ndarray, features: xgb.DMatrix) -> float:
    y_pred = booster.predict(features)
    return mean_squared_error(y_true, y_pred, squared=False)


@flow(name="train baseline model", log_prints=True)
def train_xgboost():
    print("Training model...")
    xgb_params = {
        "objective": "reg:squarederror",
        "seed": 42
    }

    with wandb.init(project=wandb_params.WANDB_PROJECT,
                    entity=wandb_params.ENTITY,
                    job_type="train",
                    config=xgb_params,) as wandb_run:

        print("Downloading data...")
        artifact = wandb_run.use_artifact('aaalex-lit/capitalbikeshare-mlops/202304-202305-202306-processed-data:latest',
                                          type='splitted_data')
        artifact_dir = Path(artifact.download())

        train = convert_to_dmatrix(*load_pickle(artifact_dir / 'train.pkl'))
        X_val, y_val = load_pickle(artifact_dir / 'val.pkl')
        val = convert_to_dmatrix(X_val, y_val)
        X_test, y_test = load_pickle(artifact_dir / 'test.pkl')
        test = convert_to_dmatrix(X_test, y_test)

        print("Training model...")

        booster = xgb.train(
            params=xgb_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(val, 'validation')],
            early_stopping_rounds=50
        )

        wandb_run.log({'validation RMSE': calculate_rmse(booster, y_val, val)})
        wandb_run.log({'test RMSE': calculate_rmse(booster, y_test, test)})

        print("Saving model...")
        model_path = get_models_dir() / 'booster.pkl'
        dump_pickle(booster, model_path)

        model_artifact = wandb.Artifact('base_booster', type='model')
        model_artifact.add_file(model_path)
        wandb_run.log_artifact(model_artifact)


if __name__ == "__main__":
    train_xgboost()
