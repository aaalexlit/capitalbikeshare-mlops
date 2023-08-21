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
The `register best model/capitalbikeshare-mlops-register-best-model` needs a parameter - sweep_id from the WANDB sweep that gets launched in `capitalbikeshare-mlops-xgb-sweep` deployment

It's also possible to run it from the command line
```shell
prefect deployment run 'register best model/capitalbikeshare-mlops-register-best-model' -p sweep_id=29jxm814
```
