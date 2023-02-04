export PYTHONPATH = $(shell echo "$$PYTHONPATH"):$(shell python -c 'import os; print ":".join(os.path.abspath(line.strip()) for line in file("PYTHONPATH"))' 2>/dev/null)

install:
	pip3 install --upgrade pip && pip3 install -r requirements.txt

format:
	black src/lightgbm_gift/train.py src/tfrs_dcn_gift/train.py src/tfrs_two_tower_gift/train.py src/tfrs_listwise_ranking_gift/train.py

lint:
	pylint --disable=R,C, src/lightgbm_gift/train.py src/tfrs_dcn_gift/train.py src/tfrs_two_tower_gift/train.py src/tfrs_listwise_ranking_gift/train.py

all: install lint format