.PHONY: lint typecheck test ci

# Ensure src is in python path for all targets
export PYTHONPATH := src

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy .

test:
	pytest
	coverage report --fail-under=85

ci: lint typecheck test
