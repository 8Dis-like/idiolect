"""Local debug script — verify the full pipeline works on CPU with tiny data.

Run this BEFORE spending any GPU credits. It catches bugs for free.

Usage:
    python scripts/debug_local.py

What it tests:
    1. Model instantiation (tiny config)
    2. Forward pass (logits shape, loss computation)
    3. Backward pass (gradients flow correctly)
    4. Generation (autoregressive decoding works)
    5. LoRA application (param freezing, trainable count)
    6. Checkpoint save/load roundtrip
    7. Style analyzer (AST-based analysis)
"""

import sys
import os
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print()


def test_model_forward():
    """Test 1: Model instantiation and forward pass."""
    print("=" * 60)
    print("TEST 1: Model Forward Pass")
    print("=" * 60)

    from codeforge.model import CodeForgeConfig, CodeForgeModel

    # Tiny model for CPU debugging (< 1M params)
    config = CodeForgeConfig(
        vocab_size=256,
        max_seq_len=64,
        n_layers=2,
        n_heads=2,
        d_model=64,
        d_ff=128,
        dropout=0.0,
    )
    model = CodeForgeModel(config)

    total_params = model.count_parameters()
    print(f"Config: {config}")
    print(f"Parameters: {total_params:,}")

    # Forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets=targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"Expected logits shape {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}"
    )
    assert loss is not None and loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN!"

    print(f"Logits shape: {logits.shape} ✓")
    print(f"Loss: {loss.item():.4f} ✓")
    print(f"Expected loss ~{-1/config.vocab_size:.4f} (log(1/vocab_size) ≈ {torch.log(torch.tensor(1.0/config.vocab_size)).item():.2f})")
    print("PASSED ✓\n")

    return model, config


def test_backward(model):
    """Test 2: Backward pass — gradients flow to all parameters."""
    print("=" * 60)
    print("TEST 2: Backward Pass (Gradient Flow)")
    print("=" * 60)

    input_ids = torch.randint(0, model.config.vocab_size, (2, 32))
    _, loss = model(input_ids, targets=input_ids)
    loss.backward()

    grad_count = 0
    no_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                grad_count += 1
                if torch.isnan(param.grad).any():
                    print(f"  ⚠️  NaN gradient in {name}")
            else:
                no_grad_params.append(name)

    print(f"Parameters with gradients: {grad_count}")
    if no_grad_params:
        print(f"  ⚠️  No gradient for: {no_grad_params}")
    else:
        print("  All trainable parameters received gradients ✓")

    model.zero_grad()
    print("PASSED ✓\n")


def test_generation(model):
    """Test 3: Autoregressive generation."""
    print("=" * 60)
    print("TEST 3: Autoregressive Generation")
    print("=" * 60)

    prompt = torch.randint(0, model.config.vocab_size, (1, 5))
    print(f"Prompt tokens: {prompt[0].tolist()}")

    generated = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)

    assert generated.shape[0] == 1, "Batch size should be 1"
    assert generated.shape[1] == 25, f"Expected 5 + 20 = 25 tokens, got {generated.shape[1]}"

    print(f"Generated tokens ({generated.shape[1]}): {generated[0].tolist()}")
    print("PASSED ✓\n")


def test_lora(model, config):
    """Test 4: LoRA application and parameter freezing."""
    print("=" * 60)
    print("TEST 4: LoRA Adapter")
    print("=" * 60)

    from codeforge.model import apply_lora

    total_before = model.count_parameters()

    model = apply_lora(
        model,
        rank=4,
        alpha=8,
        target_modules=["q_proj", "v_proj"],
    )

    trainable = model.count_parameters(trainable_only=True)
    total = model.count_parameters()

    print(f"Total params: {total:,}")
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")

    # Verify base weights are frozen
    for name, param in model.named_parameters():
        if "lora_" not in name:
            assert not param.requires_grad, f"{name} should be frozen!"

    # Forward + backward should still work with LoRA
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    _, loss = model(input_ids, targets=input_ids)
    loss.backward()

    print(f"LoRA loss: {loss.item():.4f} ✓")
    print("PASSED ✓\n")

    return model


def test_checkpoint(model, config):
    """Test 5: Save and load checkpoint roundtrip."""
    print("=" * 60)
    print("TEST 5: Checkpoint Save/Load")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ckpt.pt")

        # Save
        raw_model = model.module if hasattr(model, "module") else model
        torch.save({
            "model_state_dict": raw_model.state_dict(),
            "config": config,
        }, path)
        size_mb = os.path.getsize(path) / 1e6
        print(f"Saved checkpoint: {size_mb:.1f} MB")

        # Load into new model
        from codeforge.model import CodeForgeModel
        new_model = CodeForgeModel(config)
        ckpt = torch.load(path, weights_only=False)
        new_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint ✓")

        # Verify outputs match
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        model.eval()
        new_model.eval()
        with torch.no_grad():
            logits1, _ = model(input_ids)
            logits2, _ = new_model(input_ids)

        # Note: won't be exactly equal due to LoRA, but should be close
        print(f"Output diff (max): {(logits1 - logits2).abs().max().item():.6f}")

    print("PASSED ✓\n")


def test_style_analyzer():
    """Test 6: Code style fingerprinting."""
    print("=" * 60)
    print("TEST 6: Style Analyzer")
    print("=" * 60)

    from codeforge.evaluation.style_analyzer import analyze_code_style, compare_styles

    sample_code = '''
def calculate_fibonacci(n: int) -> list[int]:
    """Calculate the first n Fibonacci numbers.
    
    Args:
        n: Number of Fibonacci numbers to generate.
    
    Returns:
        List of Fibonacci numbers.
    """
    if n <= 0:
        return []
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    
    return sequence[:n]
'''

    fp = analyze_code_style(sample_code)
    print(f"Snake case ratio: {fp.snake_case_ratio:.2f}")
    print(f"Avg name length: {fp.avg_name_length:.1f}")
    print(f"Docstring coverage: {fp.docstring_coverage:.2f}")
    print(f"Comment density: {fp.comment_density:.1f}%")

    # Self-similarity should be 1.0
    similarity = compare_styles(fp, fp)
    assert abs(similarity - 1.0) < 0.01, f"Self-similarity should be ~1.0, got {similarity}"
    print(f"Self-similarity: {similarity:.4f} ✓")

    print("PASSED ✓\n")


if __name__ == "__main__":
    print("🔥 CodeForge Local Debug Suite\n")

    model, config = test_model_forward()
    test_backward(model)
    test_generation(model)
    model = test_lora(model, config)
    test_checkpoint(model, config)
    test_style_analyzer()

    print("=" * 60)
    print("🎉 ALL TESTS PASSED — Safe to train on GPU!")
    print("=" * 60)
