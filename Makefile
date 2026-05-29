.DEFAULT_GOAL := help
.PHONY: help install install-dl env frontend dev notebook lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: env frontend ## Install everything (Python env + frontend)

env: ## Create the Python environment with uv (base deps)
	uv sync

install-dl: ## Also install the Deep Learning dependencies (Phase 4)
	uv sync --extra dl

frontend: ## Install frontend dependencies with pnpm
	cd frontend && pnpm install

dev: ## Start the project (frontend dev server)
	cd frontend && pnpm dev

notebook: ## Open Jupyter for the notebooks (EDA, etc.)
	uv run jupyter lab

lint: ## Python linter
	uv run ruff check src

clean: ## Clean caches and artifacts
	rm -rf .ruff_cache **/__pycache__ frontend/dist
