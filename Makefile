install:
	pip3 install --upgrade pip && pip3 install -r requirements.txt

test:
	python -m pytest -vv -cov=test_py

format:
	black *.py

lint:
	pylint --disable=R,C hello.py

all: install lint format test