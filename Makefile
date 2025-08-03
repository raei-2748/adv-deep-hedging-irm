# Makefile for deep hedging project
# Convenience tasks (optional)

.PHONY: help install test run clean lint format

help:
	@echo "Deep Hedging Project - Available Commands:"
	@echo "  install    - Install dependencies"
	@echo "  test       - Run tests"
	@echo "  run        - Run the main experiment"
	@echo "  train      - Run training only"
	@echo "  clean      - Clean up generated files"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code"
	@echo "  setup      - Setup development environment"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run:
	python experiment.py

train:
	python -m src.deephedge.train

clean:
	rm -rf runs/
	rm -rf src/deephedge/data/*.png
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf src/deephedge/__pycache__/
	rm -rf src/deephedge/*/__pycache__/
	rm -rf tests/__pycache__/

lint:
	flake8 src/ tests/ experiment.py
	pylint src/ tests/ experiment.py

format:
	black src/ tests/ experiment.py
	isort src/ tests/ experiment.py

setup: install
	@echo "Setting up development environment..."
	@echo "Creating necessary directories..."
	mkdir -p runs
	mkdir -p notebooks
	@echo "Setup complete!"

docker-build:
	docker build -t deep-hedging .

docker-run:
	docker run -it deep-hedging

docker-test:
	docker run -it deep-hedging python -m pytest tests/ -v 