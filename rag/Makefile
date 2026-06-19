.PHONY: install test lint format check

install:
	uv sync --dev && uv run pre-commit install

test:
	uv run pytest -v

lint:
	uv run mypy src/ && uv run ruff check src/ tests/ && uv run black --check src/ tests/

format:
	uv run black src/ tests/ && uv run ruff check --fix src/ tests/

check: lint test
