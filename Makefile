.PHONY: install test lint format type-check coverage run train docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short 2>&1 | tail -60

coverage:
	pytest tests/ --cov=app --cov-report=term-missing --cov-report=html:htmlcov -q 2>&1 | tail -40

lint:
	ruff check . --fix
	ruff check .

format:
	ruff format .

type-check:
	mypy app/ --ignore-missing-imports --no-error-summary 2>&1 | tail -20

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

train:
	python -c "from app.model import generate_synthetic_data, train_model; df, y = generate_synthetic_data(5000); pipe, m = train_model(df, y); print('RMSE:', m['rmse_mean'])"

diagram:
	python scripts/generate_diagram.py

docker-build:
	docker build -t energy-oracle:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f model.joblib metrics.json test_energy_oracle.db energy_oracle.db
	rm -rf htmlcov .coverage
