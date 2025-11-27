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

.PHONY: test-core
test-core:
	pytest src tests --cov=src/ungar

.PHONY: test-bridge
test-bridge:
	pytest bridge/tests --cov=bridge/src/ungar_bridge

.PHONY: test-all
test-all:
	pytest src tests bridge/tests \
	  --cov=src/ungar \
	  --cov=bridge/src/ungar_bridge

.PHONY: demo-bridge
demo-bridge:
	python -m bridge.examples.demo_rediai

.PHONY: train-high-card
train-high-card:
	python bridge/examples/train_high_card_duel.py --episodes 500 --seed 123

.PHONY: bench-rl-adapter
bench-rl-adapter:
	python bridge/benchmarks/benchmark_rl_adapter_loop.py