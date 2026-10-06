.PHONY: install data test serve cli eval-retrieval eval-threshold eval-multilingual

PY ?= .venv/bin/python

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:            ## chunk MedQuAD + build test queries
	$(PY) -m medrag.data.processor

test:
	$(PY) -m pytest

serve:           ## API + bundled UI on :8000 (docs at /docs)
	$(PY) -m uvicorn medrag.api.main:app --reload --port 8000

cli:
	$(PY) -m scripts.cli

eval-retrieval:
	$(PY) -m scripts.eval_retrieval 100

eval-threshold:
	$(PY) -m scripts.eval_threshold

eval-multilingual:
	$(PY) -m scripts.eval_multilingual
