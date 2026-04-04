.PHONY: help install train test lint format clean serve dvc-pipeline docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install         - Install dependencies"
	@echo "  make train           - Train the model"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Lint code"
	@echo "  make format          - Format code with black and isort"
	@echo "  make clean           - Remove build artifacts and cache"
	@echo "  make serve           - Start API server"
	@echo "  make dvc-pipeline    - Run DVC pipeline"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"

install:
	pip install -r requirements.txt
	pip install -e .

train:
	python src/models/train.py

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/

serve:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dvc-pipeline:
	dvc repro

docker-build:
	docker build -t customer-sentiment-analysis:latest .

docker-run:
	docker run -p 8000:8000 \
	-v $(PWD)/models:/app/models \
	-v $(PWD)/mlruns:/app/mlruns \
	-v $(PWD)/data:/app/data \
	customer-sentiment-analysis:latest

.DEFAULT_GOAL := help
