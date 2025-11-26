.PHONY: lint typecheck test ci

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy .

test:
	pytest
	coverage report --fail-under=85

ci: lint typecheck test

