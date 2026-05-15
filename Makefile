.PHONY: install dev test lint format train-tokenizer pretrain pretrain-ddp finetune serve docker-build docker-run deploy clean

# ============================================================
# Development
# ============================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev,notebooks]"

test:
	pytest tests/ -v --tb=short

lint:
	ruff check codeforge/ tests/
	mypy codeforge/ --ignore-missing-imports

format:
	ruff format codeforge/ tests/
	ruff check --fix codeforge/ tests/

# ============================================================
# Data & Tokenizer
# ============================================================

download-data:
	python -m codeforge.data.download --subset small

download-data-full:
	python -m codeforge.data.download --subset full

train-tokenizer:
	python -m codeforge.data.tokenizer --train --vocab-size 32000 --data-dir data/raw

# ============================================================
# Training
# ============================================================

pretrain:
	python -m codeforge.training.pretrain --config configs/pretrain.yaml

pretrain-ddp:
	torchrun --nproc_per_node=4 -m codeforge.training.pretrain --config configs/pretrain.yaml

finetune:
	python -m codeforge.training.finetune --config configs/finetune.yaml

# ============================================================
# Serving
# ============================================================

serve:
	python -m codeforge.serving.app --host 0.0.0.0 --port 8000

# ============================================================
# Docker & Deploy
# ============================================================

docker-build:
	docker build -t codeforge:latest .

docker-run:
	docker run --gpus all -p 8000:8000 codeforge:latest

deploy:
	cd infra/terraform && terraform apply -auto-approve

# ============================================================
# Cleanup
# ============================================================

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
