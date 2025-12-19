.PHONY: help lint format typecheck typecheck-watch check fix test test-unit test-integration pre-commit-install pre-commit-run pre-commit-update all clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make help                 - Show this help message"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint                 - Run ruff linting checks (read-only)"
	@echo "  make format               - Run ruff formatting (modifies files)"
	@echo "  make check                - Run linting, formatting, and type checks (read-only)"
	@echo "  make fix                  - Auto-fix linting issues and format code"
	@echo ""
	@echo "Type Checking:"
	@echo "  make typecheck            - Run pyright type checking"
	@echo "  make typecheck-watch      - Run pyright in watch mode"
	@echo ""
	@echo "Testing:"
	@echo "  make test                 - Run pytest on all tests"
	@echo "  make test-unit            - Run unit tests only (exclude integration tests)"
	@echo "  make test-integration     - Run integration tests only"
	@echo ""
	@echo "Pre-commit:"
	@echo "  make pre-commit-install   - Install pre-commit hooks"
	@echo "  make pre-commit-run       - Run pre-commit on all files"
	@echo "  make pre-commit-update    - Update pre-commit hook versions"
	@echo ""
	@echo "Combined:"
	@echo "  make all                  - Run linting, formatting, type checking, and tests"
	@echo "  make clean                - Remove cache files"

# Code Quality
lint:
	@echo "Running ruff linting checks..."
	uv run ruff check src/ tests/

format:
	@echo "Running ruff formatting..."
	uv run ruff format src/ tests/

typecheck:
	@echo "Running pyright type checking..."
	uv run pyright

typecheck-watch:
	@echo "Running pyright in watch mode..."
	uv run pyright --watch

check:
	@echo "Running linting and formatting checks..."
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	@echo "Running type checking..."
	uv run pyright

fix:
	@echo "Auto-fixing linting issues and formatting code..."
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/

# Testing
test:
	@echo "Running all tests..."
	uv run pytest tests/

test-unit:
	@echo "Running unit tests only..."
	uv run pytest tests/ -m "not integration"

test-integration:
	@echo "Running integration tests only..."
	uv run pytest tests/ -m "integration"

# Pre-commit
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install

pre-commit-run:
	@echo "Running pre-commit on all files..."
	uv run pre-commit run --all-files

pre-commit-update:
	@echo "Updating pre-commit hook versions..."
	uv run pre-commit autoupdate

# Combined
all: check test
	@echo "All checks and tests completed successfully!"

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup complete!"
