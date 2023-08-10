## The purpose of the project

The aim of this project is to build a full end-to-end MLOps project.  
**Important:** The project main focus of the project is to show the MLOps flow and not to build the best model.

The underlying ML task is to predict bike ride duration given the start and end station, start time, bike type, and type of membership. 
## The data
The data is provided by [Capital Bikeshare](https://www.capitalbikeshare.com/system-data) and contains information about bike rides in Washington DC. 
Downloadable files are available on the following link https://s3.amazonaws.com/capitalbikeshare-data/index.html
The data used for the project is from April 2020 to Today (the scripts will get the new data automatically).
The reason is that in April 2020 the data format changed and the scripts are not compatible with the old format.

## The flow

1. Raw data download
1. Raw data combination
1. Data preparation
1. Modelling
    1. Baseline model
    1. Hyperparameter tuning using Weights and Biases Sweeps

## The project structure
The project structure is inspired by the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template (but not directly created from it).

# Steps to reproduce:

## General
1. Create a python-10 based environment (I use conda)
1. Install the dependencies from requirements.txt:
```shell
pip install -r requirements.txt
```
1. to be able to execute the code in packeges, the followint needs to be executed from the root of the project:
```shell
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```
1. `.env` file needs to be created in the root of the project with the following content:
```shell
WANDB_API_KEY=<your_wandb_api_key>
```
1. login to Prefect Cloud
## Data downloading and preparation:
1. Download raw data:
    ```shell
    python src/data/download_raw.py
    ```
2. Combine raw data into one file:
    ```shell
    python src/data/combine_raw.py
    ```
3. Prepare data for modelling:
    ```shell
    python src/data/prepare.py
    ```
## Modelling
1. Baseline xgboost model
    ```shell
    python src/models/xgb_baseline.py
    ```
2. Hyperparameter tuning for xgboost model using Weights and Biases Sweeps
    ```shell
    python src/models/xgb_sweep.py
    ```

---

For unit tests with WANDB do `WANDB_MODE=offline`

to make preparation parametrized
```python
@click.command()
@click.option('--start_year', help='start year for modelling data', type=int)
@click.option('--start_month', help='start month for modelling data', type=int)
@click.option('--end_year', default=2023, help='end year for modelling data')
@click.option('--end_month', default=5, help='end month for modelling data')
```