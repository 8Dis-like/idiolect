"""FastAPI application for CodeForge inference and style analysis.

Endpoints:
- POST /generate — Generate code from a prompt
- POST /analyze-style — Analyze coding style fingerprint
- POST /fine-tune — Trigger fine-tuning on user's code (async)
- GET /health — Health check
- GET /model-info — Model metadata
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from codeforge.model import CodeForgeConfig, CodeForgeModel, load_lora_weights
from codeforge.data.tokenizer import load_tokenizer
from huggingface_hub import hf_hub_download


# ============================================================
# Request/Response Schemas
# ============================================================

class GenerateRequest(BaseModel):
    """Code generation request."""

    prompt: str = Field(..., description="Code prompt to complete", min_length=1, max_length=4096)
    max_new_tokens: int = Field(256, ge=1, le=1024, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(50, ge=0, le=500, description="Top-k sampling")


class GenerateResponse(BaseModel):
    """Code generation response."""

    generated_code: str
    prompt: str
    tokens_generated: int
    latency_ms: float


class StyleAnalysisRequest(BaseModel):
    """Code style analysis request."""

    code: str = Field(..., description="Code snippet to analyze", min_length=10)


class StyleMatch(BaseModel):
    """A single style match result."""

    style_name: str
    similarity: float
    description: str


class StyleAnalysisResponse(BaseModel):
    """Style analysis response with top matches."""

    matches: list[StyleMatch]
    summary: str
    total_tokens: int


class ModelInfoResponse(BaseModel):
    """Model metadata."""

    model_name: str
    parameters: int
    trainable_parameters: int
    vocab_size: int
    max_seq_len: int
    device: str
    lora_enabled: bool


# ============================================================
# Global state
# ============================================================

model: Optional[CodeForgeModel] = None
tokenizer = None
config = None
device = None


# ============================================================
# App lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, tokenizer, config, device

    # Load serving config
    config_path = os.environ.get("CODEFORGE_CONFIG", "configs/serve.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    device = torch.device(model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Load tokenizer
    tokenizer = load_tokenizer(model_config.get("tokenizer_path", "artifacts/tokenizer"))
    print(f"[API] Tokenizer loaded (vocab_size={tokenizer.get_vocab_size()})")

    # Load model
    checkpoint_path = model_config.get("checkpoint_path", "checkpoints/pretrain/step_50000.pt")
    if not os.path.exists(checkpoint_path):
        print(f"[API] Local checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
        # Make sure to install huggingface_hub: pip install huggingface_hub
        hf_repo = os.environ.get("HF_REPO_ID", "Zagho/idiolect")
        hf_filename = os.environ.get("HF_MODEL_FILENAME", "model.pt")
        try:
            checkpoint_path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
            print(f"[API] Successfully downloaded model from Hugging Face to {checkpoint_path}")
        except Exception as e:
            print(f"[API] Failed to download from Hugging Face: {e}")
            raise
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_cfg = checkpoint.get("config", CodeForgeConfig())
    if isinstance(model_cfg, dict):
        model_cfg.pop("head_dim", None)
        model_cfg = CodeForgeConfig(**model_cfg)

    model = CodeForgeModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Load LoRA adapter if available
    lora_path = model_config.get("lora_path")
    if lora_path and os.path.exists(lora_path):
        model = load_lora_weights(model, lora_path)
        print(f"[API] LoRA adapter loaded from {lora_path}")

    model.eval()
    print(f"[API] Model loaded on {device} ({model.count_parameters():,} params)")

    yield  # App is running

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("[API] Shutdown complete")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="CodeForge API",
    description="Generate code in your personal style with a from-scratch trained LLM",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name="CodeForge",
        parameters=model.count_parameters(),
        trainable_parameters=model.count_parameters(trainable_only=True),
        vocab_size=model.config.vocab_size,
        max_seq_len=model.config.max_seq_len,
        device=str(device),
        lora_enabled=model.config.lora_rank > 0,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate code from a prompt."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.time()

    # Tokenize prompt
    encoded = tokenizer.encode(request.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )

    # Decode
    generated_ids = output_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    prompt_text = tokenizer.decode(encoded.ids, skip_special_tokens=True)

    # Only return the new generated part
    new_text = generated_text[len(prompt_text):]
    tokens_generated = len(generated_ids) - len(encoded.ids)
    latency_ms = (time.time() - t_start) * 1000

    return GenerateResponse(
        generated_code=new_text,
        prompt=request.prompt,
        tokens_generated=tokens_generated,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/analyze-style", response_model=StyleAnalysisResponse)
async def analyze_style(request: StyleAnalysisRequest):
    """Analyze the coding style of a code snippet.

    Compares the code's embedding fingerprint against known developer styles.
    This is the unique product feature of CodeForge.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Tokenize code
    encoded = tokenizer.encode(request.code)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    # Get model embedding (average of last hidden states)
    with torch.no_grad():
        # Forward pass without loss computation
        logits, _ = model(input_ids)
        # Use the mean of hidden states as the style embedding
        # (In production, we'd extract from an intermediate layer)

    # TODO: Implement actual style matching against pre-computed embeddings
    # For now, return placeholder results
    matches = [
        StyleMatch(
            style_name="Pythonic Clean",
            similarity=0.89,
            description="Clean, PEP-8 compliant code with descriptive variable names",
        ),
        StyleMatch(
            style_name="Functional Concise",
            similarity=0.76,
            description="Heavy use of list comprehensions and functional patterns",
        ),
        StyleMatch(
            style_name="Object-Oriented Enterprise",
            similarity=0.65,
            description="Class-heavy architecture with design patterns",
        ),
    ]

    return StyleAnalysisResponse(
        matches=matches,
        summary="Your code most closely matches the 'Pythonic Clean' style, "
                "characterized by readable, well-structured code following PEP-8 conventions.",
        total_tokens=len(encoded.ids),
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("codeforge.serving.app:app", host=host, port=port, reload=False)
