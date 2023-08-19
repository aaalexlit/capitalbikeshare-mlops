import os

from dotenv import find_dotenv, load_dotenv
from prefect.blocks.system import Secret

load_dotenv(find_dotenv())


def create_wandb_api_key_block():
    wandb_api_key = Secret(value=os.getenv('WANDB_API_KEY'))
    wandb_api_key.save(name='wandb-api-key')


if __name__ == '__main__':
    create_wandb_api_key_block()
