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
	ln -nfs $(PWD) ~/.gan
	$(PYTHON) -m pip install -r requirements.txt

run-test: setup-dev-env
	PYTHONPATH=$(SRC):$(SRC)/tests:$$PYTHONPATH coverage run -m unittest discover -s tests -p 'test_*.py' -v
	coverage xml

check_dirs := . gan tests

quality-check: setup-quality-env
	ruff check $(check_dirs)

format: setup-quality-env
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

.PHONY: run

run:
	PYTHONPATH=$(SRC) $(PYTHON) gan/gan.py