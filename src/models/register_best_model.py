from pathlib import Path

# import click
import xgboost as xgb
from dotenv import find_dotenv, load_dotenv
from prefect import flow
from wandb.xgboost import WandbCallback
from sklearn.pipeline import (  # pylint: disable=wrong-import-order
    Pipeline,
    make_pipeline,
)

import wandb
from src import wandb_params
from src.utils import (
    dump_pickle,
    load_pickle,
    calculate_rmse,
    get_models_dir,
    set_wandb_api_key,
    log_val_preds_table,
)

load_dotenv(find_dotenv())


def get_best_run_config(sweep_id: str) -> dict:
    sweep = wandb.Api().sweep(f'{wandb_params.WANDB_PROJECT}/{sweep_id}')
    return sweep.best_run().config


def save_and_log_pipeline(
    pipeline: Pipeline, wandb_run: wandb.sdk.wandb_run.Run
):
    print("Saving pipeline locally...")
    pipeline_path = get_models_dir() / "pipeline.pkl"
    dump_pickle(pipeline, pipeline_path)

    print("Uploading pipeline to W&B...")
    pipeline_artifact = wandb.Artifact('dv-model-pipeline', type="model")
    pipeline_artifact.add_file(pipeline_path)
    wandb.log_artifact(pipeline_artifact)

    # Link the model to the Model Registry
    wandb_run.link_artifact(
        pipeline_artifact,
        'model-registry/capitalbikeshare-dv-model-pipeline',
        aliases=['staging'],
    )


@flow(name="register best model", log_prints=True)
# @click.command()
# @click.argument("sweep_id", nargs=1)
# sweep_id povofsvd
def register_best_model(sweep_id: str):
    set_wandb_api_key()
    config = get_best_run_config(sweep_id)

    with wandb.init(
        project=wandb_params.WANDB_PROJECT,
        job_type="register_best_model",
        config=config,
    ) as wandb_run:
        model = xgb.XGBRegressor(
            **config,
            n_estimators=500,
            early_stopping_rounds=50,
            callbacks=[WandbCallback(log_feature_importance=False)],
        )
        data_artifact = wandb_run.use_artifact(
            '202304-202305-202306-processed-data:latest', type='processed_data'
        )
        data_artifact_dir = Path(data_artifact.download())

        print(f'Training model with best params from sweep {sweep_id}...')
        X_train, y_train = load_pickle(data_artifact_dir / 'train.pkl')
        X_val, y_val = load_pickle(data_artifact_dir / 'val.pkl')
        X_test, y_test = load_pickle(data_artifact_dir / 'test.pkl')

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
        )

        log_val_preds_table('best_model_val_preds', model, X_val, y_val)

        wandb_run.log(
            {'test-rmse': calculate_rmse(model, y_test, X_test, convert=False)}
        )

        print("Creating pipeline...")
        dv = load_pickle(data_artifact_dir / 'dv.pkl')
        pipeline = make_pipeline(dv, model)

        save_and_log_pipeline(pipeline, wandb_run)


if __name__ == "__main__":
    register_best_model()  # pylint: disable=no-value-for-parameter
