SRC?=$(shell pwd)

PYTHON?=python3.10

help:
	@cat Makefile

###########################################################################
# Dev targets
###########################################################################
setup-quality-env:
	$(PYTHON) -m pip install ruff coverage[toml] -U

setup-dev-env: setup-quality-env
	ln -nfs $(PWD) ~/.mnist_predictor
	$(PYTHON) -m pip install -r requirements.txt

run-test: setup-dev-env
	PYTHONPATH=$(SRC):$(SRC)/test:$$PYTHONPATH coverage run -m unittest discover -s test -p 'test_*.py' -v
	coverage xml

check_dirs := . mnist_predictor test

quality-check: setup-quality-env
	ruff check $(check_dirs)

format: setup-quality-env
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

.PHONY: run

run:
	PYTHONPATH=$(SRC) $(PYTHON) mnist_mlp_experiments.py