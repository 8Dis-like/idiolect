# 🔥 CodeForge — Personal Code Style LLM

> Pre-train a GPT-2-style Transformer from scratch on Python code, then fine-tune it to write code in **your** personal style.

[![CI](https://github.com/YOUR_USERNAME/codeforge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/codeforge/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![CodeForge Demo](docs/assets/demo_preview.png)

## 🎯 What is CodeForge?

CodeForge is an **end-to-end ML pipeline** that:

1. **Pre-trains** a 200M-parameter GPT-2 Transformer on 50GB of Python code from [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2)
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
- **Interactive Web Demo**: Try it live at [demo-url]

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CodeForge Pipeline                        │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │ The Stack │──▶│ BPE Tokenizer│──▶│ Pre-training       │  │
│  │ v2 Python │   │ (custom)     │   │ GPT-2 200M params  │  │
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
| Pre-training loss (final) | TBD |
| Perplexity (The Stack val) | TBD |
| Fine-tune style accuracy | TBD |
| Inference latency (P95) | <100ms |
| Training throughput (DDP 4x) | TBD tokens/sec |
| LoRA trainable params | ~2M / 200M (1%) |

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/codeforge.git
cd codeforge

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

## 📝 Blog Posts

- [How I Pre-trained a 200M-param Code LLM on a Single A10G](blog-link)
- [LoRA Fine-tuning for Personal Code Style Transfer](blog-link)
- [Building a Coding Style Fingerprint Engine](blog-link)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
