PYTHONPATH=src
VENV_PYTHON?=.venv/bin/python
ENV_TEMPLATE?=.env.dev
ENV_FILE?=.env
SUMMARY_FILE?=results/summary.json
SUMMARY_PREVIOUS_FILE?=results/summary.previous.json

.PHONY: install test prepare run sync-env benchmark phase8

install:
	pip install -r requirements.txt

test:
	$(PYTHONPATH) pytest -q

prepare:
	$(PYTHONPATH) python scripts/prepare_data.py --source-root ../dataset --target-root data/benchmark

run:
	$(PYTHONPATH) python scripts/run_benchmark.py --config configs/benchmark.yaml

sync-env:
	cp $(ENV_TEMPLATE) $(ENV_FILE)

benchmark: sync-env
	@if [ -f "$(SUMMARY_FILE)" ]; then cp "$(SUMMARY_FILE)" "$(SUMMARY_PREVIOUS_FILE)"; fi
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/run_benchmark.py --config configs/benchmark.yaml
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/compare_error_rates.py --current "$(SUMMARY_FILE)" --previous "$(SUMMARY_PREVIOUS_FILE)"

phase8:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/generate_phase8_final_experiment_set.py
