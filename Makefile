.PHONY: lint typecheck test security ci

# Ensure src is in python path for all targets
export PYTHONPATH := src

lint:
	ruff check .
	ruff format --check .
	bandit -r src/ungar

typecheck:
	mypy .

test:
	pytest
	coverage report --fail-under=85

security:
	pip-audit

ci: lint typecheck test security
