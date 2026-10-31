.PHONY: setup data train app test all clean

PYTHON := python3
SRC := src

setup:
	@echo "==> Creating virtual environment and installing dependencies"
	$(PYTHON) -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "Setup complete. Activate with: source venv/bin/activate"

data:
	@echo "==> Generating synthetic data and loading to database"
	$(PYTHON) -m src.pipeline --stage data

train:
	@echo "==> Training models and evaluating"
	$(PYTHON) -m src.pipeline --stage train

exports:
	@echo "==> Exporting Power BI CSVs"
	$(PYTHON) -m src.pipeline --stage exports

all:
	@echo "==> Running full pipeline: data -> train -> exports"
	$(PYTHON) -m src.pipeline --stage all

app:
	@echo "==> Launching Streamlit dashboard"
	streamlit run app/main.py

test:
	@echo "==> Running pytest"
	pytest tests/ -v --tb=short

clean:
	@echo "==> Cleaning generated files"
	rm -rf data/processed/* db/*.db reports/figures/* models/* exports/powerbi/*.csv __pycache__ src/__pycache__ .pytest_cache
	@echo "Clean complete."

lint:
	@echo "==> Linting"
	python -m py_compile src/*.py app/main.py app/pages/*.py

help:
	@echo "FraudLens Makefile targets:"
	@echo "  make setup    - Create venv and install dependencies"
	@echo "  make all      - Run full pipeline (data + train + exports)"
	@echo "  make data     - Generate/load data only"
	@echo "  make train    - Train models only"
	@echo "  make exports  - Export Power BI CSVs"
	@echo "  make app      - Launch Streamlit dashboard"
	@echo "  make test     - Run pytest"
	@echo "  make clean    - Remove generated files"
