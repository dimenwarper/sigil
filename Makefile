SHELL := /bin/bash

.PHONY: bootstrap test clean

bootstrap:
	bash scripts/uv-bootstrap.sh

test:
	uv run pytest -q

clean:
	rm -rf .venv .pytest_cache **/__pycache__

