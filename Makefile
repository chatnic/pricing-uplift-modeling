.PHONY: setup data model test all

setup:
	python -m pip install -r requirements.txt

data:
	python data/generate_data.py --n 60000 --seed 42 --out data/raw

model:
	PYTHONPATH=src python -m uplift.pipeline --data data/raw --out reports

test:
	python -m pytest -q

all: data model test
