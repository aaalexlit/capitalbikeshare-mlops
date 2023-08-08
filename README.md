Steps to reproduce:


## General
1. Create a python 10-based environment (I use conda)
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