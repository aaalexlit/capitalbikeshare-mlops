from pathlib import Path

import joblib
import xgboost as xgb
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task

import wandb
from wandb.xgboost import WandbCallback
import src.wandb_params as wandb_params
from src.utils import calculate_rmse, load_pickle, get_models_dir, convert_to_dmatrix

load_dotenv(find_dotenv())

SWEEP_CONFIG = {
    "name": "XGBoost Sweep",
    "method": "bayes",
    "metric": {"name": "validation-rmse.min", "goal": "minimize"},
    "parameters": {
        "max_depth": {
            "distribution": "int_uniform",
            "min": 4,
            "max": 30,
        },
        "learning_rate": {
            "distribution": "log_uniform",
            "min": -3,
            "max": 0,
        },
        "reg_alpha": {
            "distribution": "log_uniform",
            "min": -5,
            "max": -1,
        },
        "reg_lambda": {
            "distribution": "log_uniform",
            "min": -6,
            "max": -1,
        },
    },
}


def train_xgb():
    xgb_params = {
        'objective': 'reg:squarederror',
        'seed': 42,
        'nthread': 8,
    }

    wandb.init(config=xgb_params)
    config = wandb.config

    artifact = wandb.use_artifact('202304-202305-202306-processed-data:latest',
                                  type='processed_data')
    artifact_dir = Path(artifact.download())

    dv = load_pickle.fn(artifact_dir / 'dv.pkl')
    train = convert_to_dmatrix(*load_pickle.fn(artifact_dir / 'train.pkl'), dv)
    X_val, y_val = load_pickle.fn(artifact_dir / 'val.pkl')
    val = convert_to_dmatrix(X_val, y_val, dv)
    X_test, y_test = load_pickle.fn(artifact_dir / 'test.pkl')
    test = convert_to_dmatrix(X_test, y_test, dv)

    print("Training model...")
    optimized_hyperparams = {"max_depth": config.max_depth,
                             "learning_rate": config.learning_rate,
                             "reg_alpha": config.reg_alpha,
                             "reg_lambda": config.reg_lambda
                             }
    xgb_params |= optimized_hyperparams
    booster = xgb.train(
        params=xgb_params,
        dtrain=train,
        num_boost_round=500,
        evals=[(val, 'validation')],
        early_stopping_rounds=50,
        callbacks=[WandbCallback(log_model=True)],
        verbose_eval=50,
    )

    wandb.log({'test-rmse': calculate_rmse(booster, y_test, test)})


@flow(name="optimize XGB hyperparameters using wandb sweeps", log_prints=True)
def train_sweep():
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=wandb_params.WANDB_PROJECT)
    wandb.agent(sweep_id, function=train_xgb, count=10)


if __name__ == "__main__":
    train_sweep()
