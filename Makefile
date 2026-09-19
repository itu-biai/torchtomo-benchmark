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
install: ## Install requirements (install torchtomo from a checkout first, see requirements.txt)
	$(PIP) install -r requirements.txt

.PHONY: format
format: ## Format the scripts and tests
	$(RUFF) format libraries training tests

.PHONY: lint
lint: ## Lint the scripts and tests
	$(RUFF) check libraries training tests

.PHONY: test
test: ## Run tests; LEAP and torch-radon tests skip when those are missing
	$(PYTEST)

.PHONY: speed
speed: ## Time every torchtomo backend, LEAP, and torch-radon in one process
	$(PYTHON) libraries/speed_table.py

.PHONY: compare
compare: ## Quality, speed, and memory against LEAP and torch-radon, into libraries/results
	$(PYTHON) libraries/compare_libraries.py

.PHONY: check
check: format lint test ## Run format, lint, and test
