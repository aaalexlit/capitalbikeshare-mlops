quality_checks:
	isort .
	black .
	pylint --recursive=y .


setup:
	export PYTHONPATH="${PYTHONPATH}:$(pwd)"
	pre-commit install
