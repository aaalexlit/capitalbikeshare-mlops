Steps to reproduce:

1. Create a python 10-based environment (I use conda)
2. Install the dependencies from requirements.txt:
```shell
pip install -r requirements.txt
```


For unit tests with WANDB do `WANDB_MODE=offline`


to be able to execute the code in packeges, the followint needs to be executed from the root of the project:

```shell
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```