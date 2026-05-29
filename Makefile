.DEFAULT_GOAL := help
.PHONY: help install notebook preprocess baseline lint clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create the Python environment with uv
	uv sync

notebook: ## Open Jupyter for the notebooks (EDA, etc.)
	uv run jupyter lab

preprocess: ## Run the preprocessing pipeline (raw -> tensors)
	uv run sadar-preprocess

baseline: ## Fit and evaluate the Isolation Forest baseline
	uv run sadar-baseline

lint: ## Python linter
	uv run ruff check src

clean: ## Clean caches and artifacts
	rm -rf .ruff_cache **/__pycache__
