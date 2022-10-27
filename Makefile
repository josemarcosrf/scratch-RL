.PHONY: clean formatter help init install lint pyupgrade readme-toc tag test types

SHELL := /bin/bash
JOBS ?= 1

help:
	@echo "	install"
	@echo "		Install dependencies and download needed models."
	@echo "	clean"
	@echo "		Remove Python/build artifacts."
	@echo "	formatter"
	@echo "		Apply black formatting to code."
	@echo "	lint"
	@echo "		Lint code with flake8, and check if black formatter should be applied."
	@echo "	types"
	@echo "		Check for type errors using pytype."
	@echo "	pyupgrade"
	@echo "		Uses pyupgrade to upgrade python syntax."
	@echo "	readme-toc"
	@echo "			Generate a Table Of Content for the README.md"
	@echo "	test"
	@echo "		Run pytest on tests/."
	@echo "		Use the JOBS environment variable to configure number of workers (default: 1)."
	@echo " git-tag"
	@echo "		Create a git tag based on the current pacakge version and push"


install:
	pip install -r requirements.txt
	pip install -e .
	pip list

clean:
	find . -type d \( -path ./.venv \) -prune -o -name '*.pyc' -exec rm -f {} +
	find . -type d \( -path ./.venv \) -prune -o -name '*.pyo' -exec rm -f {} +
	find . -type d \( -path ./.venv \) -prune -o -name '*~' -exec rm -f  {} +
	find . -type d \( -path ./.venv \) -prune -o -name 'README.md.*' -exec rm -f  {} +
	rm -rf build/
	rm -rf .pytype/
	rm -rf dist/
	rm -rf docs/_build
	# rm -rf *egg-info
	# rm -rf pip-wheel-metadata

formatter:
	black my_package --exclude tests/

lint:
	flake8 my_package tests --exclude tests/
	black --check my_package tests --exclude tests/

types:
	# https://google.github.io/pytype/
	pytype --keep-going my_package --exclude my_package/tests

pyupgrade:
	find . -type d \( -path ./.venv \) -prune -o \
	    -name '*.py' | grep -v 'proto\|eggs\|docs' | xargs pyupgrade --py36-plus

readme-toc:
	# https://github.com/ekalinin/github-markdown-toc
	find . -type d \( -path ./.venv \) -prune -o \
	    -name README.md -exec gh-md-toc --insert {} \;


test: clean
	# OMP_NUM_THREADS can improve overral performance using one thread by process (on tensorflow), avoiding overload
	OMP_NUM_THREADS=1 pytest tests -n $(JOBS) --cov my_package


.ONESHELL:
tag:
	git tag $$( cat setup.cfg | grep version | awk -F' = ' '{print $$2}' )
	git push --tags


list:
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'
