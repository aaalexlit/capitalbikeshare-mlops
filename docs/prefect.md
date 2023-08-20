# Run the project through prefect deployments

Login to prefect cloud:

```shell
prefect cloud login
```

Create prefect secret for `WANDB_API_KEY` (it either needs to be set as environment variable or in the .env file)

```shell
python prefect-infra/create_prefect_blocks.py
```

Create work pool

```shell
prefect work-pool create --type process capitalbikeshare-mlops
```

Run the following command in this new terminal to start the worker:

```shell
prefect worker start --pool capitalbikeshare-mlops
```

Create all the deployments

```shell
prefect deploy --all
```

Now all the deployments can be run from the Prefect UI
