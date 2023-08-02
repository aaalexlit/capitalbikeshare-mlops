from src.utils import get_data_dir

import os
import shutil
from pathlib import Path
import pandas as pd
import requests
from zipfile import ZipFile
import logging


BASE_URL = 'https://s3.amazonaws.com/capitalbikeshare-data/'

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def download_locally(file_name: str) -> Path:
    """Download files locally to process and concatenate."""
    dir_path = get_data_dir()
    filepath = Path(dir_path / 'raw' / file_name)
    if not filepath.exists():
        url = BASE_URL + file_name
        response = requests.get(url, timeout=100)
        with filepath.open('wb') as f:
            f.write(response.content)
    return filepath


def unzip_file(file_path: Path) -> None:
    """Unzip files and remove the zip file and __MACOSX folder."""
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(file_path.parent)
    os.remove(file_path)
    shutil.rmtree(file_path.parent / "__MACOSX", ignore_errors=True)


def process_data(file_path: Path,
                 categorical: [str] = None,
                 numerical: [str] = None,
                 target: str = 'duration',
                 ) -> pd.DataFrame:
    """Process data for modeling."""

    if categorical is None:
        categorical = [
            'start_station_id',
            'end_station_id',
            'rideable_type',
            'member_casual',
        ]
    if numerical is None:
        numerical = ['hour']

    df = pd.read_csv(file_path, parse_dates=['started_at', 'ended_at'])

    # Drop rows with missing values - they tend to be outliers
    df = df.dropna()

    # Calculate duration in minutes
    df['duration'] = df.ended_at - df.started_at
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Drop rows with duration < 0 or > 100 minutes
    df = df[(df.duration >= 0) & (df.duration <= 100)]

    # Convert categorical columns to string
    df[categorical] = df[categorical].astype('str')

    # Create ride start hour of day feature
    df['hour'] = df.started_at.dt.hour
    df['year'] = df.started_at.dt.hour

    # Keep only columns of interest
    df = df[categorical + numerical + [target]]

    return df


def combine_save_data(dfs: [pd.DataFrame], file_path: Path) -> None:
    pd.concat(dfs).to_csv(file_path, index=False)


def prepare_data_for_modelling(start_year: int,
                               start_month: int,
                               end_year: int,
                               end_month: int,):
    years, year_months = get_year_months(
        start_year, start_month, end_year, end_month)
    for year, months in zip(years, year_months):
        for month in months:
            zip_file_name = f'{year}{month:02}-capitalbikeshare-tripdata.zip'
            file_name = f'{year}{month:02}-capitalbikeshare-tripdata.csv'
            local_zip = download_locally(zip_file_name)
            unzip_file(local_zip)
    dfs = [
        process_data(file_path)
        for file_path in Path(get_data_dir() / 'raw').glob('*.csv')
    ]
    combine_save_data(dfs, get_data_dir() / 'processed' /
                      f'{start_year}{start_month:02}-{end_year}{end_month:02}-processed.csv')


def get_year_months(start_year: int,
                    start_month: int,
                    end_year: int,
                    end_month: int,
                    ) -> ([int], [[int]]):
    """Get list of months for each year."""
    # check if start_year is less or equal to end_year
    assert start_year <= end_year, f"start_year must be less than or equal to end_year, {start_year} > {end_year}"
    assert start_month >= 1 and start_month <= 12, f"start_month must be between 1 and 12, not {start_month}"
    assert end_month >= 1 and end_month <= 12, f"end_month must be between 1 and 12, not {end_month}"
    assert start_year >= 2018, "monthly information is only available from 2018 onwards"
    assert end_year <= 2023, "monthly information is only available until 2023"
    if end_year == 2023:
        assert end_month <= 5, "monthly information for modelling is only available until May 2023"
    if start_year == end_year:
        assert start_month <= end_month, f"start_month of the same year {start_year} must be less than or equal to end_month"
        return [start_year], [list(range(start_month, end_month + 1))]
    return list(range(start_year, end_year + 1)), [
        list(range(start_month, 13)),
        *[list(range(1, 13)) for _ in range(start_year + 1, end_year)],
        *[list(range(1, end_month + 1))],
    ]


if __name__ == '__main__':
    start_year = 2023
    start_month = 1
    end_year = 2023
    end_month = 5
    prepare_data_for_modelling(start_year, start_month, end_year, end_month)
