.PHONY: run test lint frontend evaluate docker-up docker-down clean

run:            ## start everything with one command (creates .venv on first use)
	python run.py

test:           ## offline test-suite
	.venv/bin/python -m pytest

lint:
	.venv/bin/ruff check .

frontend:       ## rebuild the website into backend/app/static (needs Node 20+)
	python scripts/build_frontend.py

evaluate:       ## full walk-forward evaluation (about 10 minutes), updates ml/registry/metrics.json
	.venv/bin/python -m ml.evaluate --write-seed

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache frontend/.next frontend/out
