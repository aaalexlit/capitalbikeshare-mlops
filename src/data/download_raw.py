import src.wandb_params as wandb_params
from src.utils import get_data_dir
from src.data.process import get_year_months

import os
import shutil
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from zipfile import ZipFile
import logging
import click
import wandb
from dotenv import load_dotenv, find_dotenv


BASE_URL = 'https://s3.amazonaws.com/capitalbikeshare-data/'

logger = logging.getLogger()
logger.setLevel(logging.INFO)

load_dotenv(find_dotenv())


def download_locally(file_name: str) -> Path:
    """Download files locally to process and concatenate."""
    dir_path = get_data_dir()
    filepath = Path(dir_path / 'raw' / file_name)
    if not filepath.exists():
        url = BASE_URL + file_name
        response = requests.get(url, timeout=100)
        if response.status_code != 200:
            return None
        with filepath.open('wb') as f:
            f.write(response.content)
    return filepath


def unzip_file(zip_file_path: Path) -> None:
    """Unzip files and remove the zip file and __MACOSX folder."""
    with ZipFile(zip_file_path, 'r') as zip_ref:
        csv_filename = list(
            filter(lambda x: x.startswith('2'), zip_ref.namelist()))[0]
        extracted_csv_path = zip_ref.extract(csv_filename)
    # needed because some zipped cvs files are named incorrectly
    Path(extracted_csv_path).rename(zip_file_path.with_suffix('.csv'))
    os.remove(zip_file_path)
    # shutil.rmtree(file_path.parent / "__MACOSX", ignore_errors=True)


@click.command()
def download_raw_data():
    """Download all available raw data starting from Jan 2018 up till the current date."""

    wandb_run = wandb.init(project=wandb_params.WANDB_PROJECT,
                           entity=wandb_params.ENTITY,
                           job_type="upload")

    cur_date = datetime.now()
    years, year_months = get_year_months(
        2018, 1, cur_date.year, cur_date.month
    )
    for year, months in zip(years, year_months):
        for month in months:
            zip_file_name = f'{year}{month:02}-capitalbikeshare-tripdata.zip'
            if local_zip := download_locally(zip_file_name):
                unzip_file(local_zip)

    artifact = wandb.Artifact(wandb_params.RAW_DATA, type='raw_data')
    all_zip = shutil.make_archive(
        get_data_dir() / 'all_raw_data', 'zip', get_data_dir() / 'raw')
    artifact.add_file(Path(all_zip), name='raw_data')
    wandb_run.log_artifact(artifact)


if __name__ == '__main__':
    download_raw_data()
