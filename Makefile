.ONESHELL:
SHELL := /bin/bash

test:
	pytest tests

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist