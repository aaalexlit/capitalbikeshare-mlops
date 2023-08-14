.ONESHELL:
include .env

PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
ENV_DIR=$(shell conda info --base)
MY_ENV_DIR=$(ENV_DIR)/envs/$(PROJECT_NAME)
endif

test: export_path activate_environment
	pytest tests/

quality_checks: export_path activate_environment
	isort .
	black .
	pylint --recursive=y .

export_path:
	export PYTHONPATH="${PYTHONPATH}:$(shell pwd)"

setup: create_environment requirements export_path
	conda run -n $(PROJECT_NAME) pre-commit install

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
ifneq ("$(wildcard $(MY_ENV_DIR))","") # check if the directory is there
	@echo ">>> Found $(CONDA_ENV_NAME) environment in $(MY_ENV_DIR). Skipping installation..."
else
	@echo ">>> Detected conda, but $(CONDA_ENV_NAME) is missing in $(ENV_DIR). Installing ..."
	conda create --name $(PROJECT_NAME) python=3.10
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
endif
else
	@echo ">>> Install conda first."
	exit
#	@echo ">>> Conda not found. Checking for virtualenvwrapper..."
#	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
#	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
#	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
#	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
#	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

activate_environment:
	@echo "if you see errors make sure that you've executed"
	@echo "conda activate $(PROJECT_NAME)"

requirements: create_environment
	conda run -n $(PROJECT_NAME) $(PYTHON_INTERPRETER) -m pip install -r requirements.txt
