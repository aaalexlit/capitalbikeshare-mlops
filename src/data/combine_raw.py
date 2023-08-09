import os
import shutil
import logging
from pathlib import Path
from zipfile import ZipFile

import click
import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task
from prefect.tasks import task_input_hash

import wandb
import src.wandb_params as wandb_params
from src.utils import get_data_dir, get_year_months, get_categorical_features, TARGET_COL

load_dotenv(find_dotenv())


@task
def process_data(file_path: Path, categorical: [str] = None,
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
    df = pd.read_csv(file_path, parse_dates=date_columns,
                     usecols=categorical + date_columns,
                     dtype={'start_station_id': 'str', 'end_station_id': 'str',
                            'rideable_type': 'str', 'member_casual': 'str', })

    # Drop rows with missing values - they tend to be outliers
    df = df.dropna()

    # Calculate duration in minutes
    df['duration'] = df.ended_at - df.started_at
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Drop rows with duration < 0 or > 100 minutes
    df = df[(df.duration >= 0) & (df.duration <= 100)]

    # Drop rows with start_station_id or end_station_id that are not numbers
    df = df[df.start_station_id.str.contains('^[0-9]*$', regex= True, na=False)]
    df = df[df.end_station_id.str.contains('^[0-9]*$', regex= True, na=False)]

    # Create ride start hour of day feature
    df['hour'] = df.started_at.dt.hour
    df['year'] = df.started_at.dt.year

    return df[categorical + ['hour', 'year', target] + keep]


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


@flow(name="prepare and combine raw data", log_prints=True)
def combine_raw_data():
    """Prepare data for modelling."""

    wandb_run = wandb.init(project=wandb_params.WANDB_PROJECT,
                           job_type="prepare_and_combine")

    artifact = wandb_run.use_artifact('monthly-trip-data:latest',
                                      type='raw_data')
    artifact_dir = Path(artifact.download())

    zip_file_path = artifact_dir / 'all_raw_data.zip'

    print(f'unzipping {zip_file_path}')
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(artifact_dir)
    os.remove(zip_file_path)

    # hardcode start year and month for now cause before this date
    # the data is not in the same format
    start_year, start_month = 2020, 4
    end_year, end_month = get_latest_data_year_month(artifact_dir)

    years, year_months = get_year_months(
        start_year, start_month, end_year, end_month)

    file_paths_to_process = []
    for year, months in zip(years, year_months):
        for month in months:
            file_name = f'{year}{month:02}-capitalbikeshare-tripdata.csv'
            file_paths_to_process.append(artifact_dir / file_name)

    dfs = process_data.map(file_paths_to_process)

    result_prefix = f'{start_year}{start_month:02}-{end_year}{end_month:02}'

    interim_data_file_name = f'{result_prefix}-interim.tar.gz'
    interim_data_path = get_data_dir() / 'interim' / interim_data_file_name

    all_data_df = combine_save_data(dfs,  interim_data_path, wait_for=[dfs])

    artifact = wandb.Artifact(f'{result_prefix}-{wandb_params.INTERIM_DATA}', type='interim_data')
    artifact.add_file(interim_data_path)

    # Add random sample of data to wandb table cause it has 200k row limit
    interim_data_table = wandb.Table(dataframe=all_data_df.sample(200_000))
    artifact.add(interim_data_table, name='interim_data_table')

    wandb_run.log_artifact(artifact)
    wandb_run.finish()


if __name__ == '__main__':
    combine_raw_data()
