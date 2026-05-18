#!/usr/bin/env python3
"""LoRA Fine-tuning script for CodeForge.

Fine-tunes a pre-trained CodeForge model on a personal GitHub repository 
to adapt it to your specific coding style.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from codeforge.model import CodeForgeConfig, CodeForgeModel, apply_lora
from codeforge.data.tokenizer import load_tokenizer
from codeforge.data.dataloader import PersonalCodeDataset
from codeforge.training.trainer import Trainer, TrainerConfig

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning for CodeForge")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pre-trained checkpoint")
    parser.add_argument("--github-repos", type=str, required=True, help="Comma-separated list of GitHub repo URLs")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save LoRA adapters")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting LoRA fine-tuning on {device}")
    
    # 1. Load Tokenizer
    tokenizer = load_tokenizer("artifacts/tokenizer")
    print(f"✅ Loaded tokenizer (vocab size: {tokenizer.get_vocab_size()})")
    
    # 2. Load Base Model Config First
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_cfg = checkpoint.get("config", CodeForgeConfig())
    if isinstance(model_cfg, dict):
        model_cfg.pop("head_dim", None)
        model_cfg = CodeForgeConfig(**model_cfg)
    
    # 3. Setup Dataset
    repos = [repo.strip() for repo in args.github_repos.split(",")]
    dataset = PersonalCodeDataset(
        tokenizer=tokenizer,
        max_seq_len=model_cfg.max_seq_len,
        github_repos=repos
    )
    
    if len(dataset) == 0:
        print("❌ Error: No valid python files found in the provided repositories.")
        return
        
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
    model = CodeForgeModel(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    # 4. Apply LoRA
    print(f"🔧 Applying LoRA (Rank {args.lora_rank})...")
    model = apply_lora(model, rank=args.lora_rank)
    model.to(device)
    
    total_params = model.count_parameters()
    trainable_params = model.count_parameters(trainable_only=True)
    print(f"📊 Trainable params: {trainable_params:,} / {total_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    
    # 5. Initialize Trainer
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    config = TrainerConfig(
        max_steps=total_steps,
        batch_size=args.batch_size,
        gradient_accumulation=1,
        learning_rate=args.lr,
        warmup_steps=min(100, total_steps // 10),
        checkpoint_dir=args.output_dir,
        save_every_steps=steps_per_epoch,
        eval_every_steps=steps_per_epoch,
        wandb_project="idiolect-finetune"
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,
        config=config,
        device=device
    )
    
    # 6. Train!
    print("🔥 Beginning training loop...")
    trainer.train()
    
    # 7. Save Final LoRA Weights
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "lora_weights.pt")
    
    # Extract only LoRA weights
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, final_path)
    print(f"🎉 Fine-tuning complete! LoRA weights saved to {final_path}")


if __name__ == "__main__":
    main()
