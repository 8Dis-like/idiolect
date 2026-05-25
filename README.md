# 🔥 Idiolect — Personal Code Style LLM

> *Your code has a fingerprint. Idiolect learns it.*
>
> Pre-train a GPT-2-style Transformer from scratch on Python code, then fine-tune it to write code in **your** personal style.

[![CI](https://github.com/8Dis-like/idiolect/actions/workflows/ci.yml/badge.svg)](https://github.com/8Dis-like/idiolect/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is Idiolect?

**Idiolect** *(noun): an individual's distinctive language patterns.* This project is an **end-to-end ML pipeline** that:

1. **Pre-trains** an 85M-parameter GPT-2 Transformer from scratch on Python code from [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
2. **Fine-tunes** with LoRA on any developer's personal GitHub repositories to capture their unique coding style
3. **Serves** predictions via a production REST API on AWS
4. **Analyzes** coding style fingerprints — discover which famous developer you code like!

### 🌟 Key Features

- **From-scratch Transformer**: Custom implementation of GPT-2 architecture with multi-head attention, not a HuggingFace wrapper
- **Custom BPE Tokenizer**: Trained on code corpus with code-specific vocabulary (operators, indentation tokens)
- **Distributed Training**: PyTorch DDP across multiple A10G GPUs with mixed-precision (AMP)
- **Parameter-Efficient Fine-tuning**: LoRA (rank=8) reduces trainable parameters by 98%
- **Style Fingerprinting**: Novel coding style analysis using embedding-space clustering
- **Production Deployment**: FastAPI on AWS ECS with <100ms P95 latency

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Idiolect Pipeline                         │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ The Stack │──▶│ BPE Tokenizer│──▶│ Pre-training       │  │
│  │ v2 Python │   │ (custom)     │   │ GPT-2 85M params   │  │
│  │ 50GB      │   │ 32K vocab    │   │ Causal LM objective│  │
│  └──────────┘   └──────────────┘   └─────────┬─────────┘  │
│                                               │             │
│  ┌──────────┐   ┌──────────────┐   ┌─────────▼─────────┐  │
│  │ Personal │──▶│ Style        │──▶│ Fine-tuning        │  │
│  │ GitHub   │   │ Extraction   │   │ LoRA r=8, α=16     │  │
│  │ Repos    │   │ Pipeline     │   │ ~2M trainable      │  │
│  └──────────┘   └──────────────┘   └─────────┬─────────┘  │
│                                               │             │
│  ┌──────────────────────────────────┐  ┌─────▼─────────┐  │
│  │ Style Fingerprint Engine         │  │ FastAPI Server │  │
│  │ • Embedding clustering           │  │ • REST API     │  │
│  │ • Style similarity matching      │  │ • Web Demo     │  │
│  │ • Coding pattern analysis        │  │ • AWS ECS      │  │
│  └──────────────────────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Pre-training loss (final) | 3.52164386212826 |
| Tokens per second | 129854.046183306 |
| Fine-tune style accuracy | 5.22863829135896 |
| Inference latency (P95) | <100ms |
| Training throughput (DDP 4x) | TBD tokens/sec |
| LoRA trainable params | ~2M / 85M (~2%) |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/8Dis-like/idiolect.git
cd idiolect

# Install dependencies
pip install -e ".[dev]"

# Download dataset (subset for testing)
python -m codeforge.data.download --subset small

# Train tokenizer
python -m codeforge.data.tokenizer --train --vocab-size 32000

# Pre-train (single GPU)
python -m codeforge.training.pretrain --config configs/pretrain.yaml

# Pre-train (multi-GPU DDP)
torchrun --nproc_per_node=4 -m codeforge.training.pretrain --config configs/pretrain.yaml

# Fine-tune on personal code
python -m codeforge.training.finetune --config configs/finetune.yaml --user-repo https://github.com/YOUR_USERNAME

# Launch API server
python -m codeforge.serving.app

# Run tests
pytest tests/ -v
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | PyTorch 2.x (custom Transformer) |
| Training | DDP, AMP, gradient accumulation |
| Fine-tuning | LoRA (peft) |
| Tokenizer | Custom BPE (tokenizers library) |
| Experiment Tracking | Weights & Biases |
| API | FastAPI + Uvicorn |
| Container | Docker (multi-stage) |
| Cloud | AWS (EC2 g5, S3, ECR, ECS, API Gateway) |
| CI/CD | GitHub Actions |
| IaC | Terraform |
| Frontend | React + Vite |

## 📁 Project Structure

```
codeforge/
├── configs/              # YAML training configs
├── codeforge/
│   ├── data/             # Data download, tokenizer, dataloader
│   ├── model/            # Transformer architecture, LoRA
│   ├── training/         # Pre-train, fine-tune, DDP
│   ├── evaluation/       # Metrics, style analysis
│   └── serving/          # FastAPI app, inference
├── infra/terraform/      # AWS infrastructure as code
├── frontend/             # React web demo
├── notebooks/            # Exploration & visualization
├── tests/                # Unit & integration tests
└── docs/                 # Architecture & guides
```

