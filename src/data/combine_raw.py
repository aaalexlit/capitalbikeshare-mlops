import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task

import wandb
from src import wandb_params
from src.utils import (
    TARGET_COL,
    get_data_dir,
    feature_dtypes,
    get_year_months,
    set_wandb_api_key,
    get_categorical_features,
)

load_dotenv(find_dotenv())


@task
def process_data(
    file_path: Path,
    categorical: [str] = None,
    target: str = TARGET_COL,
    keep: [str] = None,
    date_columns: [str] = None,
) -> pd.DataFrame:
    """Process data for modeling."""

    if keep is None:
        keep = ['started_at']
    if date_columns is None:
        date_columns = ['started_at', 'ended_at']
    if categorical is None:
        categorical = get_categorical_features()

    print(f'processing {file_path}')
    df = pd.read_csv(
        file_path,
        parse_dates=date_columns,
        usecols=categorical + date_columns,
        dtype=feature_dtypes(),
    )

    # Drop rows with missing values - they tend to be outliers
    df = df.dropna()

    # Calculate duration in minutes
    df['duration'] = df.ended_at - df.started_at
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Drop rows with duration < 0 or > 100 minutes
    df = df[(df.duration >= 0) & (df.duration <= 100)]

    # Drop rows with start_station_id not a number
    df = df[df.start_station_id.str.contains('^[0-9]*$', regex=True, na=False)]
    df = df[df.end_station_id.str.contains('^[0-9]*$', regex=True, na=False)]

    return df[categorical + [target] + keep]


@task
def combine_save_data(dfs: [pd.DataFrame], file_path: Path) -> pd.DataFrame:
    """Combine and save data."""
    print(f'combining and saving data to {file_path}')
    df = pd.concat(dfs)
    df.to_csv(file_path, index=False)
    return df


def get_latest_data_year_month(data_dir: Path) -> (int, int):
    """Get latest data year and month."""
    latest_file_name = (sorted(data_dir.glob("*.csv"))[-1]).name
    prefix = latest_file_name.split('-')[0]
    return int(prefix[:4]), int(prefix[4:6])


@task
def extract_zip(artifact_dir: Path) -> None:
    zip_file_path = artifact_dir / 'all_raw_data.zip'
    print(f'unzipping {zip_file_path}')
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(artifact_dir)
    os.remove(zip_file_path)


def get_file_paths_to_process(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    artifact_dir: Path,
) -> [Path]:
    years, year_months = get_year_months(
        start_year, start_month, end_year, end_month
    )

    file_paths_to_process = []
    for year, months in zip(years, year_months):
        for month in months:
            file_name = f'{year}{month:02}-capitalbikeshare-tripdata.csv'
            file_paths_to_process.append(artifact_dir / file_name)
    return file_paths_to_process


@flow(name="prepare and combine raw data", log_prints=True)
def combine_raw_data():
    """Prepare data for modelling."""
    set_wandb_api_key()

    with wandb.init(
        project=wandb_params.WANDB_PROJECT, job_type="prepare_and_combine"
    ) as wandb_run:
        artifact_dir = Path(
            wandb_run.use_artifact(
                'monthly-trip-data:latest', type='raw_data'
            ).download()
        )

        extract_zip(artifact_dir)

        # hardcode start year and month for now cause before this date
        # the data is not in the same format
        start_year, start_month = 2020, 4
        end_year, end_month = get_latest_data_year_month(artifact_dir)

        file_paths_to_process = get_file_paths_to_process(
            start_year,
            start_month,
            end_year,
            end_month,
            artifact_dir,
        )

        dfs = process_data.map(file_paths_to_process)

        result_prefix = f'{start_year}{start_month:02}-{end_year}{end_month:02}'

        interim_data_path = (
            get_data_dir() / 'interim' / f'{result_prefix}-interim.tar.gz'
        )

        all_data_df = combine_save_data(dfs, interim_data_path, wait_for=[dfs])

        artifact = wandb.Artifact(
            f'{result_prefix}-{wandb_params.INTERIM_DATA}', type='interim_data'
        )
        artifact.add_file(interim_data_path)

        # Add random sample of data to wandb table cause it has 200k row limit
        interim_data_table = wandb.Table(dataframe=all_data_df.sample(200_000))
        artifact.add(interim_data_table, name='interim_data_table')

        wandb_run.log_artifact(artifact)


if __name__ == '__main__':
    combine_raw_data()
