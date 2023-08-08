from pathlib import Path
from datetime import date

import joblib
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer

import wandb
import src.wandb_params as wandb_params
from src.utils import TARGET_COL, get_data_dir, \
    get_categorical_features, dump_pickle

load_dotenv(find_dotenv())


@task
def preprocess(df: pd.DataFrame,
               dv: DictVectorizer,
               fit_dv: bool = False,) -> (pd.DataFrame, DictVectorizer):
    dicts = df[get_categorical_features() + ['hour', 'year']
               ].to_dict(orient="records")
    X = dv.fit_transform(dicts) if fit_dv else dv.transform(dicts)
    return X, dv


@flow(name="prepare and split into train, val, test", log_prints=True)
def prepare_data():
    print("Preparing data...")

    wandb_run = wandb.init(project=wandb_params.WANDB_PROJECT,
                           entity=wandb_params.ENTITY,
                           job_type="prepare_and_split")

    artifact = wandb_run.use_artifact('aaalex-lit/capitalbikeshare-mlops/202004-202306-interim-data:latest',
                                      type='interim_data')

    artifact_dir = Path(artifact.download())

    print(f'Loading data from {artifact_dir}')
    df = pd.read_csv(
        artifact_dir / '202004-202306-interim.tar.gz',
        parse_dates=['started_at'],
        dtype={'start_station_id': 'str', 'end_station_id': 'str',
               'rideable_type': 'str', 'member_casual': 'str', })

    print('Splitting data...')
    train_split_year, train_split_month = 2023, 4
    val_split_year, val_split_month = 2023, 5
    test_split_year, test_split_month = 2023, 6

    train_split_date = date(train_split_year, train_split_month, 1)
    val_split_date = date(val_split_year, val_split_month, 1)
    test_split_date = date(test_split_year, test_split_month, 1)

    df_train = df[df.started_at.dt.date < train_split_date]
    df_val = df[(df.started_at.dt.date >= train_split_date)
                & (df.started_at.dt.date < val_split_date)]
    df_test = df[(df.started_at.dt.date >= val_split_date) &
                 (df.started_at.dt.date < test_split_date)]

    print(f'Extracting target column "{TARGET_COL}"')
    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values
    y_test = df_test[TARGET_COL].values

    print('Fitting the DictVectorizer and preprocessing data')
    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_val, _ = preprocess(df_val, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    print('Saving DictVectorizer and datasets')
    # Create dest_path folder unless it already exists
    dest_path = get_data_dir() / "processed"

    dump_pickle(dv, dest_path / "dv.pkl")
    dump_pickle((X_train, y_train), dest_path / "train.pkl")
    dump_pickle((X_val, y_val), dest_path / "val.pkl")
    dump_pickle((X_test, y_test), dest_path / "test.pkl")

    prefix = f'{train_split_date.strftime("%Y%m")}-{val_split_date.strftime("%Y%m")}-{test_split_date.strftime("%Y%m")}'
    artifact = wandb.Artifact(
        f'{prefix}-{wandb_params.PROCESSED_DATA}', type="processed_data")
    artifact.add_dir(dest_path)
    wandb_run.log_artifact(artifact)
    wandb_run.finish()

    print("Data prepared!")


if __name__ == "__main__":
    prepare_data()
