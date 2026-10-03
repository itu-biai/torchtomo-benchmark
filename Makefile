.DEFAULT_GOAL := help

VENV := .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff

.PHONY: help
help: ## Show this help
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: venv
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)

.PHONY: install
install: ## Install requirements, including torchtomo 0.4.0
	$(PIP) install -r requirements.txt

.PHONY: format
format: ## Format the scripts and tests
	$(RUFF) format libraries training geometry tests

.PHONY: lint
lint: ## Lint the scripts and tests
	$(RUFF) check libraries training geometry tests

.PHONY: test
test: ## Run tests; LEAP and torch-radon tests skip when those are missing
	$(PYTEST)

.PHONY: speed
speed: ## Time every torchtomo backend and every other library in one process
	$(PYTHON) libraries/speed_table.py

.PHONY: compare
compare: ## Quality, speed, and memory against the other libraries, into libraries/results/0.4.0/{parallel,fan}
	$(PYTHON) libraries/compare_libraries.py
	$(PYTHON) libraries/compare_libraries.py --geometry fan

.PHONY: calibrate
calibrate: ## Centre of rotation on HTC 2022 and the FIPS walnut, into geometry/results
	$(PYTHON) geometry/calibrate_real.py
	$(PYTHON) geometry/figures.py

.PHONY: motion
motion: ## Per-view motion and angle errors against projection matching, into geometry/results
	$(PYTHON) geometry/correct_motion.py

.PHONY: check
check: format lint test ## Run format, lint, and test
