SHELL := /bin/bash

.PHONY: bootstrap test clean

bootstrap:
	bash scripts/uv-bootstrap.sh

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf .venv .pytest_cache **/__pycache__

