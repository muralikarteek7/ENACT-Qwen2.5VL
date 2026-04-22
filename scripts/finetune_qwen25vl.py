"""
Finetune Qwen2.5-VL-7B-Instruct on ENACT ordering task using unsloth + LoRA.
Optimized for A100-SXM4-40GB VRAM. Flash Attention 2 is used automatically by unsloth.

Setup (run once):
    pip install unsloth unsloth_zoo
    pip install trl datasets pillow qwen-vl-utils

Quick test (100 samples, 1 epoch — verify everything works):
    python finetune_qwen25vl.py --limit 100 --epochs 1 --output ./test_adapter

Full training run:
    python finetune_qwen25vl.py

After training, download the adapter folder to your Mac:
    scp -P <port> -r user@<ip>:~/ENACT/lora_enact_ordering/ ./

Then on your Mac convert to MLX:
    python -m mlx_lm.fuse \
        --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit \
        --adapter-path lora_enact_ordering/ \
        --save-path fused_model/

Then run inference as normal:
    python scripts/inference_mlx.py --model fused_model/ --output enact_finetuned.jsonl
"""

import argparse
import json
import random
import shutil
import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(__file__).parent.parent / "data"
QA_FILE   = DATA_ROOT / "QA" / "enact_ordering.jsonl"


def load_samples(qa_file: Path, data_root: Path, limit: int | None = None) -> list[dict]:
    """Load ENACT QA pairs and filter out any with missing images."""
    samples = []
    with open(qa_file) as f:
        for line in f:
            samples.append(json.loads(line))
    if limit:
        samples = samples[:limit]

    valid = []
    missing_count = 0
    for s in samples:
        paths = [data_root / p for p in s["images"]]
        if all(p.exists() for p in paths):
            valid.append(s)
        else:
            missing_count += 1

    print(f"Valid samples: {len(valid)} / {len(samples)}  (skipped {missing_count} with missing images)")
    return valid


class ENACTDataset(torch.utils.data.Dataset):
    """Lazy-loading dataset — images loaded per batch by dataloader workers, not upfront."""

    def __init__(self, samples: list[dict], data_root: Path):
        self.samples = samples
        self.data_root = data_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        sample = self.samples[idx]
        image_paths = [self.data_root / p for p in sample["images"]]
        images = [PILImage.open(p).convert("RGB") for p in image_paths]
        num_future = len(sample["images"]) - 1

        fixed_question = (
            sample["question"]
            + f"\nOutput EXACTLY {num_future} integers as a Python list."
            + f"\nExample format: {list(range(1, num_future + 1))}"
        )

        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": fixed_question})

        return {
            "images": images,
            "conversations": [
                {"role": "user",      "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": str(sample["gt_answer"])}]},
            ],
        }


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-VL on ENACT")

    # Model
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model ID from HuggingFace")

    # Output
    parser.add_argument("--output", default="./lora_enact_ordering",
                        help="Where to save the LoRA adapter")

    # Training hyperparameters — tuned for A100-SXM4-40GB
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=4,    # 40GB BF16 + VL images; increase to 6-8 if VRAM allows
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum",  type=int,   default=4,    # effective batch = 16
                        help="Gradient accumulation steps")
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int,   default=4096)
    parser.add_argument("--lora-rank",   type=int,   default=64,   # 40GB comfortably handles rank=64; try 128 for more capacity
                        help="LoRA rank — higher = more expressive but more VRAM")

    # Data
    parser.add_argument("--limit",     type=int,   default=None,
                        help="Limit number of samples (use 100 for quick test)")
    parser.add_argument("--val-split",  type=float, default=0.10,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test-split", type=float, default=0.05,
                        help="Fraction of data to hold out as test (never seen during training)")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--image-size", type=int,   default=336,
                        help="Max image edge in pixels (336=2x faster than 512, fine for activity recognition)")
    parser.add_argument("--hardest-first", action="store_true",
                        help="Sort by most images first (stress test VRAM)")

    parser.add_argument("--clean", action="store_true",
                        help="Delete output dir before training (removes old weights)")
    args = parser.parse_args()

    # ── Imports (delayed so argparse --help works without GPU) ────────────────
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    import torch
    from trl import SFTTrainer, SFTConfig

    # ── Validate GPU ──────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. This script requires a CUDA GPU.")

    # TF32 — faster matmuls on Ampere/Blackwell, no precision loss for training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")

    if args.clean and Path(args.output).exists():
        print(f"Removing old weights at {args.output}")
        shutil.rmtree(args.output)

    # Warn if less than 32GB — may OOM with these defaults
    if vram_gb < 32:
        print(f"WARNING: Only {vram_gb:.1f}GB VRAM detected. "
              f"Consider reducing --batch-size to 2 and --lora-rank to 32.")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=args.max_seq_len,          # must match trainer max_seq_length
        device_map={"": 0},                       # force all layers onto GPU 0, no CPU offload
    )
    # Set image resolution on the processor (controls visual token count)
    tokenizer.image_processor.size = {
        "shortest_edge": args.image_size * args.image_size // 4,
        "longest_edge":  args.image_size * args.image_size,
    }
    print(f"Image size: {args.image_size}px  (~{(args.image_size//14)**2 // 4} tokens/image)")

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    # Training ALL layer types since 32GB VRAM gives us room
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,      # train vision encoder (image understanding)
        finetune_language_layers=True,    # train LLM layers (reasoning)
        finetune_attention_modules=True,  # train attention (sequence ordering)
        finetune_mlp_modules=True,        # train MLP (feature transformation)
        r=args.lora_rank,                 # LoRA rank (64 for 32GB)
        lora_alpha=args.lora_rank,        # alpha = rank is standard practice
        lora_dropout=0,                   # unsloth recommends 0 for speed
        bias="none",
        random_state=args.seed,
    )

    # Print how many parameters are actually being trained
    model.print_trainable_parameters()

    # ── Load and split data ───────────────────────────────────────────────────
    print(f"\nLoading data from {QA_FILE}")
    samples = load_samples(QA_FILE, DATA_ROOT, limit=None)  # load all, limit after sort

    random.seed(args.seed)
    random.shuffle(samples)
    if args.limit:
        samples = samples[: args.limit]

    # Carve test first (completely held out — never touches training or val selection)
    test_n        = max(1, int(len(samples) * args.test_split)) if args.test_split > 0 else 0
    test_samples  = samples[:test_n]
    remaining     = samples[test_n:]

    # Then split remaining into val and train
    val_n         = max(1, int(len(remaining) * args.val_split))
    val_samples   = remaining[:val_n]
    train_samples = remaining[val_n:]

    # Sort each split independently by image count so batches have similar VRAM cost
    train_samples.sort(key=lambda s: len(s["images"]), reverse=args.hardest_first)
    val_samples.sort(key=lambda s: len(s["images"]))
    test_samples.sort(key=lambda s: len(s["images"]))
    print(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}  |  Test: {len(test_samples)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split IDs so inference can evaluate each split separately
    split_path = output_dir / "split_ids.json"
    with open(split_path, "w") as f:
        json.dump({
            "train": [s["id"] for s in train_samples],
            "val":   [s["id"] for s in val_samples],
            "test":  [s["id"] for s in test_samples],
            "seed":  args.seed,
            "val_split":  args.val_split,
            "test_split": args.test_split,
        }, f, indent=2)
    print(f"Split IDs saved to {split_path}")

    # Save training config so inference can auto-match resolution and seq length
    train_config_path = output_dir / "train_config.json"
    with open(train_config_path, "w") as f:
        json.dump({
            "model":       args.model,
            "image_size":  args.image_size,
            "max_seq_len": args.max_seq_len,
            "lora_rank":   args.lora_rank,
        }, f, indent=2)
    print(f"Train config saved to {train_config_path}")

    # ── Build datasets (lazy — images loaded per batch by workers) ────────────
    print("Building datasets (lazy loading — no images preloaded)...")
    train_dataset = ENACTDataset(train_samples, DATA_ROOT)
    val_dataset   = ENACTDataset(val_samples,   DATA_ROOT)

    # ── Set model to training mode ────────────────────────────────────────────
    FastVisionModel.for_training(model)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            output_dir=args.output,

            # Training schedule
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,   # effective batch = batch_size × grad_accum
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",                    # cosine decay

            # Precision — bf16 on 5090 (Blackwell supports bf16 natively)
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",                                # 8-bit optimizer saves ~4GB VRAM

            # Logging and saving
            logging_steps=10,
            save_steps=200,
            eval_steps=200,
            eval_strategy="steps",
            save_total_limit=2,                            # keep only 2 checkpoints
            load_best_model_at_end=True,                   # restore best val checkpoint

            # Misc
            report_to="none",                              # no wandb/tensorboard
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},

            # Memory optimization for long sequences
            max_seq_length=args.max_seq_len,
            dataloader_num_workers=32,                     # parallel data loading (90 cores available)
            dataloader_pin_memory=True,                    # faster H2D transfers on A100
            dataloader_prefetch_factor=2,                  # prefetch batches ahead of GPU
        ),

        data_collator=UnslothVisionDataCollator(model, tokenizer, max_seq_length=args.max_seq_len),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"  LoRA rank:       {args.lora_rank}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Train samples:   {len(train_samples)}")
    print(f"  Val samples:     {len(val_samples)}")
    print()

    trainer_stats = trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Print training summary
    runtime_mins = trainer_stats.metrics.get("train_runtime", 0) / 60
    samples_per_sec = trainer_stats.metrics.get("train_samples_per_second", 0)
    peak_gb   = torch.cuda.max_memory_allocated() / 1e9
    total_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    total_images = sum(len(s["images"]) for s in train_samples)
    print(f"\nTraining complete!")
    print(f"  Total time:      {runtime_mins:.1f} minutes")
    print(f"  Samples/second:  {samples_per_sec:.1f}")
    print(f"  Peak VRAM:       {peak_gb:.1f} GB / {total_gb:.1f} GB ({100*peak_gb/total_gb:.0f}% used)")
    print(f"  Train samples:   {len(train_samples)} samples, {total_images} images total")
    print(f"  Avg images/sample: {total_images/len(train_samples):.1f}")
    print(f"  Adapter saved:   {args.output}/")
    print()
    print("Next steps:")
    print("  1. Download adapter to Mac:  scp -r user@ip:~/ENACT/lora_enact_ordering/ ./")
    print("  2. Fuse on Mac:  python -m mlx_lm.fuse --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit --adapter-path lora_enact_ordering/ --save-path fused_model/")
    print("  3. Run inference: python scripts/inference_mlx.py --model fused_model/ --output enact_finetuned.jsonl")


if __name__ == "__main__":
    main()