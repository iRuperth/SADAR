.DEFAULT_GOAL := help
.PHONY: help install notebook preprocess baseline train-lstm train-transformer train-vae tune-vae evaluate compare serve web dev artifacts mlflow-ui lint clean docker-build docker-up docker-down docker-logs

PROCESSED_DIR ?= data/processed
MODELS_DIR ?= models
VAE_CHECKPOINT ?= $(MODELS_DIR)/vae_lstm.pt
SCALER ?= $(PROCESSED_DIR)/scaler.npz

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

train-lstm: ## Train the LSTM autoencoder on the normal flights
	uv run sadar-train-lstm

train-transformer: ## Train the Transformer autoencoder on the normal flights
	uv run sadar-train-transformer

train-vae: ## Train the VAE-LSTM on the normal flights
	uv run sadar-train-vae

tune-vae: ## Search VAE-LSTM hyperparameters with Optuna
	uv run sadar-tune-vae

evaluate: ## Evaluate a trained autoencoder against synthetic anomalies
	uv run sadar-evaluate

compare: ## Compare every detector and select the final model
	uv run sadar-compare

artifacts: ## Ensure the preprocessed tensors and the VAE checkpoint exist
	@if [ ! -f "$(SCALER)" ]; then \
		echo "[artifacts] $(SCALER) missing, running preprocess..."; \
		$(MAKE) preprocess; \
	fi
	@if [ ! -f "$(VAE_CHECKPOINT)" ]; then \
		echo "[artifacts] $(VAE_CHECKPOINT) missing, trying to restore from MLflow..."; \
		uv run python -m sadar.models.restore || true; \
	fi
	@if [ ! -f "$(VAE_CHECKPOINT)" ]; then \
		echo "[artifacts] still missing, training VAE-LSTM..."; \
		$(MAKE) train-vae; \
	fi

serve: artifacts ## Run the inference API for the dashboard
	uv run uvicorn sadar.serve.app:app --port 8000

web: ## Run the frontend dev server (Vite)
	pnpm -C frontend dev

dev: artifacts ## Run the backend API and the frontend dev server together
	@trap 'kill 0' EXIT; uv run uvicorn sadar.serve.app:app --port 8000 & pnpm -C frontend dev

mlflow-ui: ## Open the MLflow experiment tracking UI
	uv run mlflow ui --backend-store-uri file:mlruns

lint: ## Python linter
	uv run ruff check src

clean: ## Clean caches and artifacts
	rm -rf .ruff_cache **/__pycache__

docker-build: ## Build the backend and frontend container images
	docker compose build

docker-up: ## Start the stack in the background (backend on :8000, frontend on :5180)
	docker compose up -d

docker-down: ## Stop the stack and remove the containers
	docker compose down

docker-logs: ## Tail the logs of the running stack
	docker compose logs -f
