SHELL := /bin/bash

install:
	poetry install

clean: clean_docs
	rm -rf dist

clean_docs:
	cd docs && make clean

test: install
	poetry run tox -p -o -r

build:
	poetry build

publish: test clean build
	poetry run python -mtwine upload dist/* --verbose

.PHONY: docs

docs:
	cd docs && make html
