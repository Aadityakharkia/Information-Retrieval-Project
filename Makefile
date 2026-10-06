.PHONY: install build serve test eval clean

PY ?= .venv/bin/python

install:        ## create venv and install dependencies
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

build:          ## (re)build chunks, inverted index and dense embeddings from data/raw
	rm -f data/processed/*.pkl data/processed/*.npy
	$(PY) -c "from backend.pipeline import get_pipeline; get_pipeline().initialize()"

serve:          ## UI + API on http://127.0.0.1:5000
	$(PY) -m backend.app

test:
	$(PY) -m pytest

eval:           ## retrieval benchmark -> eval/results/
	$(PY) -m eval.run_retrieval_eval

clean:
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache
