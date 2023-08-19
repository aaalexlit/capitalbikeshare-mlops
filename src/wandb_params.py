import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

WANDB_PROJECT = os.getenv('PROJECT_NAME', 'capitalbikeshare-mlops')
RAW_DATA = 'monthly-trip-data'
INTERIM_DATA = 'interim-data'
PROCESSED_DATA = 'processed-data'
