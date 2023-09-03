from pathlib import Path

import xgboost as xgb
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task
from wandb.xgboost import WandbCallback

import wandb
from src import wandb_params
from src.utils import (
    dump_pickle,
    load_pickle,
    calculate_rmse,
    get_models_dir,
    set_wandb_api_key,
    convert_to_dmatrix,
    log_val_preds_table,
)

load_dotenv(find_dotenv())


@task(log_prints=False)
def train_booster(params, train: xgb.DMatrix, val: xgb.DMatrix) -> xgb.Booster:
    return xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(val, 'validation')],
        early_stopping_rounds=50,
        callbacks=[WandbCallback()],
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
    set_wandb_api_key()
    with wandb.init(
        project=wandb_params.WANDB_PROJECT,
        job_type="train",
        config=xgb_params,
    ) as wandb_run:
        print("Downloading data...")
        artifact = wandb_run.use_artifact(
            '202304-202305-202306-processed-data:latest', type='processed_data'
        )
        artifact_dir = Path(artifact.download())

        dv = load_pickle(artifact_dir / 'dv.pkl')
        feature_names = dv.get_feature_names_out()
        train = convert_to_dmatrix(
            *load_pickle(artifact_dir / 'train.pkl'), feature_names
        )
        X_val, y_val = load_pickle(artifact_dir / 'val.pkl')
        val = convert_to_dmatrix(X_val, y_val, feature_names)
        X_test, y_test = load_pickle(artifact_dir / 'test.pkl')

        print("Training model...")
        booster = train_booster(xgb_params, train, val)

        log_val_preds_table('baseline_booster_val_preds', booster, val, y_val)

        wandb_run.log({'test-rmse': calculate_rmse(booster, y_test, X_test)})

        print("Saving model locally...")
        model_path = get_models_dir() / 'booster.pkl'
        dump_pickle(booster, model_path)


if __name__ == "__main__":
    train_xgboost()
