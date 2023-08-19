from pathlib import Path
from datetime import date

import numpy as np
import scipy as sp
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer

import wandb
from src import wandb_params
from src.utils import (
    TARGET_COL,
    dump_pickle,
    get_data_dir,
    feature_dtypes,
    set_wandb_api_key,
    get_categorical_features,
)

load_dotenv(find_dotenv())


# pylint: disable=too-many-locals
def preprocess(
    df: pd.DataFrame,
    dv: DictVectorizer,
    fit_dv: bool = False,
) -> (sp.sparse.csr_matrix, DictVectorizer):
    # Create ride start hour of day feature
    df['hour'] = df.started_at.dt.hour
    df['month'] = df.started_at.dt.month
    df['year'] = df.started_at.dt.year

    if fit_dv:
        print("Fitting DictVectorizer...")
    else:
        print("Transforming data...")
    dicts = df[get_categorical_features() + ['hour', 'year', 'month']].to_dict(
        orient="records"
    )
    X = dv.fit_transform(dicts) if fit_dv else dv.transform(dicts)
    return X, dv


@task
def dataset_split(
    df: pd.DataFrame,
    end_split_date: date,
    dv: DictVectorizer,
    fit_dv: bool = False,
    start_split_date: date = date(1970, 1, 1),
) -> (sp.sparse.csr_matrix, np.ndarray, DictVectorizer):
    print(
        f"Extract split from {start_split_date} to {end_split_date} and target {TARGET_COL}"
    )
    X = df[
        (df.started_at.dt.date >= start_split_date)
        & (df.started_at.dt.date < end_split_date)
    ]
    y = X[TARGET_COL].values
    X, dv = preprocess(X, dv, fit_dv=fit_dv)
    return X, y, dv


# to make preparation parametrized
# @click.command()
# @click.option('--start_year', help='start year for modelling data', type=int)
# @click.option('--start_month', help='start month for modelling data', type=int)
# @click.option('--end_year', default=2023, help='end year for modelling data')
# @click.option('--end_month', default=5, help='end month for modelling data')
@flow(name="prepare and split into train, val, test", log_prints=True)
# pylint: disable=too-many-arguments
def prepare_data(
    train_split_year: int = 2023,
    train_split_month: int = 4,
    val_split_year: int = 2023,
    val_split_month: int = 5,
    test_split_year: int = 2023,
    test_split_month: int = 6,
):
    print("Preparing data...")
    set_wandb_api_key()
    with wandb.init(
        project=wandb_params.WANDB_PROJECT, job_type="prepare_and_split"
    ) as wandb_run:
        artifact_dir = Path(
            wandb_run.use_artifact(
                '202004-202306-interim-data:latest', type='interim_data'
            ).download()
        )

        print(f'Loading data from {artifact_dir}')
        df = pd.read_csv(
            artifact_dir / '202004-202306-interim.tar.gz',
            parse_dates=['started_at'],
            dtype=feature_dtypes(),
        )

        train_split_date = date(train_split_year, train_split_month, 1)
        val_split_date = date(val_split_year, val_split_month, 1)
        test_split_date = date(test_split_year, test_split_month, 1)

        dv = DictVectorizer()
        X_train, y_train, dv = dataset_split(
            df, train_split_date, dv, fit_dv=True
        )
        X_val, y_val, _ = dataset_split(
            df, val_split_date, dv, start_split_date=train_split_date
        )
        X_test, y_test, _ = dataset_split(
            df, test_split_date, dv, start_split_date=val_split_date
        )

        print('Saving DictVectorizer and datasets')
        dest_path = get_data_dir() / "processed"

        dump_pickle(dv, dest_path / "dv.pkl")
        dump_pickle((X_train, y_train), dest_path / "train.pkl")
        dump_pickle((X_val, y_val), dest_path / "val.pkl")
        dump_pickle((X_test, y_test), dest_path / "test.pkl")

        # pylint: disable=line-too-long
        prefix = f'{train_split_date.strftime("%Y%m")}-{val_split_date.strftime("%Y%m")}-{test_split_date.strftime("%Y%m")}'
        artifact = wandb.Artifact(
            f'{prefix}-{wandb_params.PROCESSED_DATA}', type="processed_data"
        )
        artifact.add_dir(dest_path)
        wandb_run.log_artifact(artifact)

    print("Data prepared!")


if __name__ == "__main__":
    prepare_data()
