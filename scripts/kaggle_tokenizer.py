# ============================================================
# CodeForge: Tokenizer Training on Kaggle
# ============================================================
# 
# HOW TO USE:
# 1. Go to kaggle.com/code → New Notebook
# 2. Settings → Accelerator: None (CPU is fine for tokenizer)
# 3. Settings → Internet: ON
# 4. Paste this ENTIRE file into one cell and run
# 5. After completion, download artifacts/tokenizer/ from the Output tab
#
# Runtime: ~10-15 minutes on CPU
# ============================================================

# --- Cell 1: Install dependencies ---
import subprocess
subprocess.run(["pip", "install", "-q", "tokenizers>=0.19.0", "datasets>=2.18.0", "huggingface_hub"], check=True)

# --- Cell 1b: Hugging Face Login ---
import os
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

print("🔐 Logging into Hugging Face...")
try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    login(hf_token)
    print("✅ Successfully logged in!")
except Exception as e:
    print("⚠️  Error accessing HF_TOKEN secret. Did you add it to Kaggle Secrets?")
    print("   1. Go to Add-ons -> Secrets")
    print("   2. Add a new secret named HF_TOKEN")
    print("   3. Paste your Hugging Face token (get it from huggingface.co/settings/tokens)")
    print("   4. Attach it to this notebook and run again.")
    raise e

# --- Cell 2: Download Python code subset ---
import os
from pathlib import Path
# pyrefly: ignore [missing-import]
from datasets import load_dataset

print("📥 Downloading Python code from StarCoder data...")
print("   (Using starcoderdata — lighter than full Stack v2)")

# Download ~5GB subset — enough for a solid 32K vocab tokenizer
# On Kaggle this takes ~5-8 minutes
ds = load_dataset(
    "bigcode/starcoderdata",
    data_dir="python",
    split="train",
    streaming=True,
)

# Save first 200K files as .txt for tokenizer training
data_dir = Path("tokenizer_data")
data_dir.mkdir(exist_ok=True)

num_files = 0
target_files = 200_000  # 200K Python files is plenty for tokenizer

print(f"📝 Extracting {target_files:,} Python files...")
batch_file = data_dir / "batch_0.txt"
current_batch = 0
lines_in_batch = 0
f = open(batch_file, "w", encoding="utf-8")

for sample in ds:
    code = sample.get("content", "")
    if not code or len(code) < 50:
        continue
    
    f.write(code)
    f.write("\n\n# === FILE_SEPARATOR === #\n\n")
    lines_in_batch += 1
    num_files += 1
    
    # Split into 10MB-ish files for efficient training
    if lines_in_batch >= 10000:
        f.close()
        current_batch += 1
        batch_file = data_dir / f"batch_{current_batch}.txt"
        f = open(batch_file, "w", encoding="utf-8")
        lines_in_batch = 0
    
    if num_files >= target_files:
        break
    
    if num_files % 10000 == 0:
        print(f"   Extracted {num_files:,}/{target_files:,} files...")

f.close()
print(f"✅ Extracted {num_files:,} files into {current_batch + 1} batches")

# --- Cell 3: Train BPE Tokenizer ---
# pyrefly: ignore [missing-import]
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders

print("\n🔧 Training BPE tokenizer (vocab_size=32,000)...")

# Special tokens for code LLM
SPECIAL_TOKENS = [
    "<|pad|>",         # 0: Padding
    "<|bos|>",         # 1: Beginning of sequence
    "<|eos|>",         # 2: End of sequence
    "<|unk|>",         # 3: Unknown
    "<|sep|>",         # 4: File separator
    "<|indent|>",      # 5: Indentation marker
    "<|dedent|>",      # 6: De-indentation
    "<|newline|>",     # 7: Explicit newline
    "<|fim_prefix|>",  # 8: Fill-in-middle prefix
    "<|fim_suffix|>",  # 9: Fill-in-middle suffix
    "<|fim_middle|>",  # 10: Fill-in-middle target
    "<|user_style|>",  # 11: Personal style marker
]

# Initialize BPE
tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.TemplateProcessing(
    single="<|bos|> $A <|eos|>",
    special_tokens=[("<|bos|>", 1), ("<|eos|>", 2)],
)

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

# Collect all .txt files
train_files = sorted(str(f) for f in data_dir.glob("*.txt"))
print(f"   Training on {len(train_files)} files...")

# Train! (~5-10 min on Kaggle CPU)
tokenizer.train(train_files, trainer=trainer)

print(f"✅ Tokenizer trained: vocab_size = {tokenizer.get_vocab_size()}")

# --- Cell 4: Save & Test ---
import json

output_dir = Path("artifacts/tokenizer")
output_dir.mkdir(parents=True, exist_ok=True)

# Save tokenizer
tokenizer.save(str(output_dir / "tokenizer.json"))

# Save config
config = {
    "vocab_size": tokenizer.get_vocab_size(),
    "special_tokens": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
    "num_training_files": num_files,
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"\n📁 Saved to {output_dir}/")
print(f"   tokenizer.json: {os.path.getsize(output_dir / 'tokenizer.json') / 1e6:.1f} MB")

# Test it
print("\n🧪 Testing tokenizer:")
test_cases = [
    'def hello_world():\n    print("Hello, World!")\n    return True',
    'import torch\nimport torch.nn as nn\nfrom typing import Optional',
    'class CodeForgeModel(nn.Module):\n    """Custom Transformer for code generation."""',
    'x = [i**2 for i in range(10) if i % 2 == 0]',
]

for code in test_cases:
    encoded = tokenizer.encode(code)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\n  Input:   {code[:60]}{'...' if len(code) > 60 else ''}")
    print(f"  Tokens:  {len(encoded.ids)} → {encoded.tokens[:8]}...")
    print(f"  Decoded: {decoded[:60]}{'...' if len(decoded) > 60 else ''}")

# Compression ratio
long_code = "\n".join(test_cases)
encoded = tokenizer.encode(long_code)
ratio = len(long_code) / len(encoded.ids)
print(f"\n📊 Compression ratio: {ratio:.1f} chars/token (good: 3-5)")

print("\n" + "=" * 60)
print("✅ DONE! Download 'artifacts/tokenizer/' from the Output tab.")
print("   Upload it to your GitHub repo under artifacts/tokenizer/")
print("=" * 60)
