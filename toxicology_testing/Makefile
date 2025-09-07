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
	ln -nfs $(PWD) ~/.toxicology_testing
	$(PYTHON) -m pip install -r requirements.txt

check_dirs := . toxicology_testing

quality-check: setup-quality-env
	ruff check $(check_dirs)

format: setup-quality-env
	ruff format $(check_dirs)
	ruff check $(check_dirs) --fix

.PHONY: run

run:
	PYTHONPATH=$(SRC) $(PYTHON) toxicology.py