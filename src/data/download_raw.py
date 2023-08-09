import os
import shutil
from pathlib import Path
from zipfile import ZipFile
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

import wandb
import src.wandb_params as wandb_params
from src.utils import get_data_dir, get_year_months

BASE_URL = 'https://s3.amazonaws.com/capitalbikeshare-data/'

load_dotenv(find_dotenv())


@task(retries=3, cache_key_fn=task_input_hash, cache_expiration=timedelta(weeks=27))
def download_locally(file_name: str) -> Path:
    """Download files locally to process and concatenate."""
    print(f'downloading {file_name}')
    dir_path = get_data_dir()
    filepath = Path(dir_path / 'raw' / file_name)
    if not filepath.exists():
        url = BASE_URL + file_name
        response = requests.get(url, timeout=100)
        if response.status_code != 200:
            return None
        with filepath.open('wb') as f:
            f.write(response.content)
            print(f'downloaded {file_name}')
    return filepath


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(weeks=27))
def unzip_file(zip_file_path: Path) -> None:
    """Unzip root level csv file from the zip archive."""
    if zip_file_path:
        print(f'unzipping {zip_file_path}')
        with ZipFile(zip_file_path, 'r') as zip_ref:
            csv_filename = list(
                filter(lambda x: x.startswith('2'), zip_ref.namelist()))[0]
            extracted_csv_path = zip_ref.extract(csv_filename)
        # needed because some zipped cvs files are named incorrectly
        print(f'extracted {extracted_csv_path}')
        Path(extracted_csv_path).rename(zip_file_path.with_suffix('.csv'))
        os.remove(zip_file_path)


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=30))
def zip_the_folder() -> str:
    """Zip the raw data folder."""
    print('creating zip archive with all the raw data')
    return shutil.make_archive(
        get_data_dir() / 'all_raw_data', 'zip', get_data_dir() / 'raw')



@flow(name="download and unzip all the data")
# @task(cache_key_fn=task_input_hash, cache_expiration=timedelta(days=30))
def download_and_unzip_all_the_data() -> None:
    cur_date = datetime.now()
    years, year_months = get_year_months(
        2018, 1, cur_date.year, cur_date.month
    )
    zip_file_names = []
    for year, months in zip(years, year_months):
        for month in months:
            zip_file_name = f'{year}{month:02}-capitalbikeshare-tripdata.zip'
            zip_file_names.append(zip_file_name)
    print('start downloading all the data')
    local_zips = download_locally.map(zip_file_names)
    unzip_file.map(local_zips)
    print('finished downloading all the data')


@flow(name="download raw data", log_prints=True)
def download_raw_data():
    """Download all available raw data starting from Jan 2018 up till the current date."""

    wandb_run = wandb.init(project=wandb_params.WANDB_PROJECT,
                           job_type="upload")

    all_downloaded = download_and_unzip_all_the_data()

    artifact = wandb.Artifact(wandb_params.RAW_DATA, type='raw_data')
    all_zip = zip_the_folder(wait_for=[all_downloaded])
    artifact.add_file(Path(all_zip))
    print('uploading raw data artifact to wandb')
    wandb_run.log_artifact(artifact)


if __name__ == '__main__':
    download_raw_data()
