SHELL := /bin/bash

install:
	poetry install

clean: clean_docs
	rm -rf dist

clean_docs:
	cd docs && make clean

test_docs_examples:
	# test that notebooks in documentation execute
	echo "Testing Jupyter notebook execution in documentation..."
	poetry run pytest --nbval-lax docs/source

test_docs: test_docs_examples docs
	echo "Testing links in documentation..."
	cd docs && poetry run make linkcheck


test: install test_docs
	poetry run tox -p -o -r

build:
	poetry build

publish: test clean build
	poetry run python -mtwine upload dist/* --verbose

.PHONY: docs

docs:
	cd docs && poetry run make html

lab: install
	poetry run jupyter lab

