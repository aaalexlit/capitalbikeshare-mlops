from pathlib import Path

import xgboost as xgb
from dotenv import find_dotenv, load_dotenv
from prefect import flow
from wandb.xgboost import WandbCallback
from prefect.deployments import (  # pylint: disable=wrong-import-order
    run_deployment,
)

import wandb
from src import wandb_params
from src.utils import (
    load_pickle,
    calculate_rmse,
    set_wandb_api_key,
    convert_to_dmatrix,
)

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

    artifact = wandb.use_artifact(
        '202304-202305-202306-processed-data:latest', type='processed_data'
    )
    artifact_dir = Path(artifact.download())

    dv = load_pickle.fn(artifact_dir / 'dv.pkl')
    feature_names = dv.get_feature_names_out()

    train = convert_to_dmatrix(
        *load_pickle.fn(artifact_dir / 'train.pkl'), feature_names
    )
    X_val, y_val = load_pickle.fn(artifact_dir / 'val.pkl')
    val = convert_to_dmatrix(X_val, y_val, feature_names)
    X_test, y_test = load_pickle.fn(artifact_dir / 'test.pkl')

    print("Training model...")
    optimized_hyperparams = {
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
    }
    xgb_params |= optimized_hyperparams
    booster = xgb.train(
        params=xgb_params,
        dtrain=train,
        num_boost_round=200,
        evals=[(val, 'validation')],
        early_stopping_rounds=50,
        callbacks=[WandbCallback()],
        verbose_eval=50,
    )

    wandb.log({'test-rmse': calculate_rmse(booster, y_test, X_test)})


# pylint: disable=unused-argument,redefined-outer-name
def trigger_model_retraining(flow, flow_run, state):
    print(
        f"hello from {flow_run.name}'s completion hook |"
        f" the return value was {(r := state.result())!r}"
    )
    run_deployment(
        name="register best model/capitalbikeshare-mlops-register-best-model",
        parameters={'sweep_id': r},
    )


@flow(
    name="optimize XGB hyperparameters using wandb sweeps",
    log_prints=True,
    persist_result=True,
    on_completion=[trigger_model_retraining],
)
def train_sweep():
    set_wandb_api_key()
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=wandb_params.WANDB_PROJECT)
    wandb.agent(sweep_id, function=train_xgb, count=5)
    return sweep_id


if __name__ == "__main__":
    train_sweep()
