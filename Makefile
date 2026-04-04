.PHONY: help install install-dev setup train test lint format clean serve dvc-pipeline data-download data-validate models-evaluate docker-build docker-run check all

# Default Python interpreter
PYTHON := python
PIP := pip

# Project paths
SRC_DIR := src
TESTS_DIR := tests
DATA_DIR := data
MODELS_DIR := models
MLRUNS_DIR := mlruns
LOGS_DIR := logs

help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  Customer Sentiment Analysis - MLOps Pipeline"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Setup & Installation:"
	@echo "    make install         - Install dependencies"
	@echo "    make install-dev     - Install dependencies + dev tools"
	@echo "    make setup           - Full project setup (install + dirs)"
	@echo ""
	@echo "  Development:"
	@echo "    make lint            - Lint code (flake8, isort, black)"
	@echo "    make format          - Format code with black and isort"
	@echo "    make test            - Run all tests with coverage"
	@echo "    make check           - Run lint + type checks"
	@echo ""
	@echo "  Data & Training:"
	@echo "    make data-download   - Download raw data"
	@echo "    make data-validate   - Validate data quality"
	@echo "    make dvc-pipeline    - Run full DVC pipeline"
	@echo "    make train           - Train model"
	@echo "    make models-evaluate - Evaluate all trained models"
	@echo ""
	@echo "  Serving & Deployment:"
	@echo "    make serve           - Start API server (dev mode)"
	@echo "    make docker-build    - Build Docker image"
	@echo "    make docker-run      - Run Docker container"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make clean           - Remove artifacts, cache, logs"
	@echo "    make all             - Full build: setup -> check -> test"
	@echo "═══════════════════════════════════════════════════════════════"

# Installation targets
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: install
	$(PIP) install -e ".[dev,ml]"

setup: install
	@mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed
	@mkdir -p $(MODELS_DIR)
	@mkdir -p $(MLRUNs_DIR)
	@mkdir -p $(LOGS_DIR)
	@echo "✓ Project directories created"

# Training targets
train:
	$(PYTHON) $(SRC_DIR)/models/train.py

data-download:
	@echo "Downloading raw data..."
	$(PYTHON) $(SRC_DIR)/data/prepare.py --download

data-validate:
	@echo "Validating data quality..."
	$(PYTHON) $(SRC_DIR)/data/prepare.py --validate

dvc-pipeline:
	dvc repro

models-evaluate:
	$(PYTHON) $(SRC_DIR)/models/train.py --evaluate

# Testing targets
test:
	pytest $(TESTS_DIR)/ -v --cov=$(SRC_DIR) --cov-report=html

test-quick:
	pytest $(TESTS_DIR)/ -v --tb=short

test-coverage:
	pytest $(TESTS_DIR)/ --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

# Code quality targets
lint:
	flake8 $(SRC_DIR)/ $(TESTS_DIR)/ --max-line-length=100
	isort --check-only $(SRC_DIR)/ $(TESTS_DIR)/
	black --check $(SRC_DIR)/ $(TESTS_DIR)/

format:
	isort $(SRC_DIR)/ $(TESTS_DIR)/
	black $(SRC_DIR)/ $(TESTS_DIR)/
	@echo "✓ Code formatted"

type-check:
	mypy $(SRC_DIR)/ --ignore-missing-imports

check: lint type-check
	@echo "✓ All checks passed"

# Cleanup targets
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf $(LOGS_DIR)/*
	@echo "✓ Cleaned up"

clean-models:
	rm -rf $(MODELS_DIR)/*.pkl $(MODELS_DIR)/*.joblib
	@echo "✓ Models cleaned"

clean-all: clean clean-models
	dvc gc
	@echo "✓ Full cleanup completed"

# Serving targets
serve:
	uvicorn $(SRC_DIR).api.main:app --reload --host 0.0.0.0 --port 8000

serve-prod:
	uvicorn $(SRC_DIR).api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Docker targets
docker-build:
	docker build -t customer-sentiment-analysis:latest .

image-build: docker-build

image-tag:
	docker tag customer-sentiment-analysis:latest customer-sentiment-analysis:v0.1.0

docker-run:
	docker run -p 8000:8000 \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/data:/app/data \
		customer-sentiment-analysis:latest

# Composite targets
all: setup check test
	@echo "✓ Full build completed successfully"

.DEFAULT_GOAL := help
