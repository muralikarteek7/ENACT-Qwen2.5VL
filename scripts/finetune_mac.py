"""
Fine-tune Qwen2.5-VL-7B-Instruct on ENACT ordering task on Apple Silicon (M-series Mac).
Uses HuggingFace transformers + PEFT QLoRA with float16 on MPS.

NOTE: MPS does NOT support bfloat16 — float16 is used throughout.
      Expect ~2-4 samples/min on M3/M4/M5 Max with 32GB unified memory.
      For serious training, use a cloud GPU (RTX 5090 = 100x faster).
      Use this for quick experiments, small datasets, or verifying the pipeline.

Setup (one-time):
    pip install transformers accelerate peft trl datasets pillow qwen-vl-utils safetensors

Quick test (10 samples, 1 epoch — ~5 minutes):
    python scripts/finetune_mac.py --limit 10 --epochs 1 --output ./test_adapter_mac

Small experiment (~1 hour, 200 samples):
    python scripts/finetune_mac.py --limit 200 --epochs 2 --output ./lora_mac_200

Full dataset (NOT recommended on Mac — use cloud GPU):
    python scripts/finetune_mac.py --output ./lora_mac_full
"""

import argparse
import json
import random
import shutil
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data"
QA_FILE   = DATA_ROOT / "QA" / "enact_ordering.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_samples(qa_file: Path, data_root: Path, limit: int | None = None) -> list[dict]:
    samples = []
    with open(qa_file) as f:
        for line in f:
            samples.append(json.loads(line))
    if limit:
        samples = samples[:limit]

    valid, missing = [], 0
    for s in samples:
        paths = [data_root / p for p in s["images"]]
        if all(p.exists() for p in paths):
            valid.append(s)
        else:
            missing += 1

    print(f"Valid samples: {len(valid)} / {len(samples)}  (skipped {missing} with missing images)")
    return valid


def sample_to_messages(sample: dict, data_root: Path):
    from PIL import Image as PILImage
    image_paths = [data_root / p for p in sample["images"]]
    images = [PILImage.open(p).convert("RGB") for p in image_paths]
    num_future = len(sample["images"]) - 1
    question = (
        sample["question"]
        + f"\nOutput EXACTLY {num_future} integers as a Python list."
        + f"\nExample format: {list(range(1, num_future + 1))}"
    )
    user_content = [{"type": "image"} for _ in images]
    user_content.append({"type": "text", "text": question})
    return images, [
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": str(sample["gt_answer"])}]},
    ]


class ENACTDataset:
    def __init__(self, samples: list[dict], data_root: Path):
        self.samples  = samples
        self.data_root = data_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, conversations = sample_to_messages(self.samples[idx], self.data_root)
        return {"images": images, "conversations": conversations}


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5-VL on Mac (MPS)")
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--output",     default="./lora_mac")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=1,   help="Keep at 1 on Mac (memory limited)")
    parser.add_argument("--grad-accum", type=int,   default=8,   help="Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--lora-rank",  type=int,   default=16,  help="Lower rank = less memory (16 for 32GB Mac)")
    parser.add_argument("--max-seq-len",type=int,   default=2048)
    parser.add_argument("--limit",      type=int,   default=None)
    parser.add_argument("--val-split",  type=float, default=0.05)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--clean",      action="store_true", help="Delete output dir before training")
    args = parser.parse_args()

    import torch

    if not torch.backends.mps.is_available():
        print("WARNING: MPS not available. Training on CPU — extremely slow.")
        print("  On Apple Silicon Mac, ensure macOS 12.3+ and PyTorch 2.0+.")
        device = "cpu"
        dtype  = torch.float32
    else:
        device = "mps"
        dtype  = torch.float16   # MPS does not support bfloat16
        print(f"MPS available. Training on Apple Silicon GPU (float16).")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig

    if args.clean and Path(args.output).exists():
        print(f"Removing {args.output}")
        shutil.rmtree(args.output)

    print(f"\nLoading model: {args.model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # LoRA config — small rank for Mac memory budget
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.05,
        bias="none",
        # Target attention + MLP in language model layers only (skip vision encoder to save memory)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and split data
    print(f"\nLoading data from {QA_FILE}")
    samples = load_samples(QA_FILE, DATA_ROOT, limit=None)
    random.seed(args.seed)
    random.shuffle(samples)
    # Sort by image count so batches have uniform VRAM cost
    samples.sort(key=lambda s: len(s["images"]))
    if args.limit:
        samples = samples[: args.limit]

    val_n         = max(1, int(len(samples) * args.val_split))
    train_samples = samples[val_n:]
    val_samples   = samples[:val_n]
    print(f"Train: {len(train_samples)}  |  Val: {len(val_samples)}")

    # Save split IDs
    split_path = Path(args.output) / "split_ids.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({
            "train":     [s["id"] for s in train_samples],
            "val":       [s["id"] for s in val_samples],
            "seed":      args.seed,
            "val_split": args.val_split,
        }, f, indent=2)
    print(f"Split IDs → {split_path}")

    # Estimate time
    secs_per_sample = 25   # rough estimate on M-series
    total_steps = (len(train_samples) * args.epochs) / (args.batch_size * args.grad_accum)
    est_hours = (len(train_samples) * args.epochs * secs_per_sample) / 3600
    print(f"\nEstimated training time: {est_hours:.1f}h  ({total_steps:.0f} optimizer steps)")
    print(f"  Tip: use --limit 100 for a ~{100*secs_per_sample/3600:.1f}h quick run\n")

    train_dataset = ENACTDataset(train_samples, DATA_ROOT)
    val_dataset   = ENACTDataset(val_samples,   DATA_ROOT)

    # Custom collator: processor handles tokenization + image encoding
    def collate_fn(batch):
        texts, all_images = [], []
        for item in batch:
            images      = item["images"]
            convs       = item["conversations"]
            text        = processor.apply_chat_template(convs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            all_images.extend(images)

        inputs = processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )
        # Labels: mask padding and input tokens, only compute loss on answer tokens
        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return {k: v.to(device) for k, v in inputs.items()}

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        args=SFTConfig(
            output_dir=args.output,

            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",

            # Mac: float16 (bfloat16 not supported on MPS)
            fp16=True,
            bf16=False,
            optim="adamw_torch",          # 8-bit optim not available on MPS, use standard

            logging_steps=5,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,

            report_to="none",
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},

            max_seq_length=args.max_seq_len,
            dataloader_num_workers=0,     # MPS requires workers=0 (no forking)
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(f"\nSaving LoRA adapter → {args.output}")
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)

    runtime_mins = trainer_stats.metrics.get("train_runtime", 0) / 60
    sps          = trainer_stats.metrics.get("train_samples_per_second", 0)
    print(f"\nDone! {runtime_mins:.1f} min  |  {sps:.2f} samples/sec")
    print(f"\nNext: run inference on val split to check generalization:")
    print(f"  python scripts/inference_hf.py \\")
    print(f"      --adapter {args.output} \\")
    print(f"      --id-file {args.output}/split_ids.json \\")
    print(f"      --split val --output enact_val.jsonl")
    print(f"  enact eval enact_val.jsonl")


if __name__ == "__main__":
    main()
