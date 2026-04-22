"""
Run ENACT benchmark inference on Linux/GPU using HuggingFace transformers + PEFT.
Works with both base model and fine-tuned LoRA adapter.

Usage:
    # Single GPU
    python scripts/inference_hf.py --adapter ./lora_enact_ordering --output enact_finetuned.jsonl

    # Multi-GPU sharding (2 GPUs → ~2x faster)
    python scripts/inference_hf.py --adapter ./lora_enact_ordering --run-shards 2 --output enact_finetuned.jsonl

    # Quick test
    python scripts/inference_hf.py --adapter ./lora_enact_ordering --limit 50 --output test.jsonl
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data"
QA_FILE   = DATA_ROOT / "QA" / "enact_ordering.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def parse_answer(raw: str) -> str:
    match = re.search(r"\[[\d,\s]+\]", raw)
    return match.group(0) if match else raw


def run_shards(num_shards: int, args: argparse.Namespace):
    """Launch one subprocess per GPU shard (set CUDA_VISIBLE_DEVICES per process)."""
    base_cmd = [sys.executable, __file__,
                "--model", args.model,
                "--input", args.input,
                "--data-root", args.data_root,
                "--num-shards", str(num_shards),
                "--prefetch", str(args.prefetch)]
    if args.adapter:
        base_cmd += ["--adapter", args.adapter]
    if args.output:
        base_cmd += ["--output", args.output]
    if args.limit:
        base_cmd += ["--limit", str(args.limit)]

    procs = []
    for shard in range(num_shards):
        log_path = Path(f"shard_hf_{shard}.log")
        log_f = open(log_path, "w")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard)}
        cmd = base_cmd + ["--shard", str(shard)]
        print(f"Starting shard {shard} (GPU {shard}) → {log_path}")
        procs.append((shard, subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env), log_f))

    print(f"\n{num_shards} shards running. Tail logs with:")
    for shard, _, _ in procs:
        print(f"  tail -f shard_hf_{shard}.log")

    for shard, proc, log_f in procs:
        proc.wait()
        log_f.close()
        print(f"Shard {shard} finished (exit {proc.returncode})")

    print("\nAll shards done. Merging...")
    model_slug = (args.adapter or args.model).replace("/", "-").split("/")[-1]
    output_path = Path(args.output) if args.output else Path(f"enact_ordering_{model_slug}.jsonl")
    shard_files = [Path(f"enact_ordering_{model_slug}_shard{i}.jsonl") for i in range(num_shards)]

    with open(output_path, "w") as out_f:
        for sf in shard_files:
            if sf.exists():
                with open(sf) as f:
                    out_f.write(f.read())
                sf.unlink()

    print(f"Merged → {output_path}")
    print(f"Evaluate with:  enact eval {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (optional)")
    parser.add_argument("--input",   default=str(QA_FILE))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output",  default=None)
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--id-file",    default=None,
                        help="JSON file with split_ids (from finetune). Use with --split train|val to filter samples.")
    parser.add_argument("--split",      default=None, choices=["train", "val", "test"],
                        help="Which split to run (requires --id-file)")
    parser.add_argument("--image-size", type=int, default=336,
                        help="Max image edge in pixels — must match what was used during training (default 336)")
    parser.add_argument("--prefetch",   type=int, default=4,
                        help="CPU prefetch workers for image loading (default 4)")
    parser.add_argument("--shard",      type=int, default=0,  help="This shard index (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1,  help="Total shards")
    parser.add_argument("--run-shards", type=int, default=None,
                        help="Launch N shards as subprocesses (one per GPU), then merge")
    args = parser.parse_args()

    # Auto-load image_size from train_config.json if adapter is given and flag wasn't explicitly set
    if args.adapter:
        train_config_path = Path(args.adapter) / "train_config.json"
        if train_config_path.exists():
            with open(train_config_path) as f:
                train_cfg = json.load(f)
            loaded_size = train_cfg.get("image_size")
            if loaded_size and loaded_size != args.image_size:
                print(f"[train_config] Overriding --image-size {args.image_size} → {loaded_size} (matched to training)")
                args.image_size = loaded_size
            elif loaded_size:
                print(f"[train_config] image_size={loaded_size}px (matches training)")
        else:
            print(f"WARNING: No train_config.json found in {args.adapter} — using --image-size {args.image_size}. "
                  f"Make sure this matches the resolution used during training.")

    if args.run_shards:
        run_shards(args.run_shards, args)
        return

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model_slug = (args.adapter or args.model).replace("/", "-").split("/")[-1]
    if args.num_shards > 1:
        base = Path(args.output).with_suffix("") if args.output else Path(f"enact_ordering_{model_slug}")
        output_path = Path(str(base) + f"_shard{args.shard}.jsonl")
    else:
        output_path = Path(args.output) if args.output else Path(f"enact_ordering_{model_slug}.jsonl")
    data_root   = Path(args.data_root)

    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        if done_ids:
            print(f"[shard {args.shard}] Resuming — {len(done_ids)} done.")

    # MPS (Apple Silicon) doesn't support bfloat16 — use float16 and explicit device
    if torch.backends.mps.is_available():
        dtype = torch.float16
        device_map = {"": "mps"}
    else:
        dtype = torch.bfloat16
        device_map = "auto"

    print(f"Loading model: {args.model}  (device: {'mps' if torch.backends.mps.is_available() else 'cuda'})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device_map,
    )

    if args.adapter:
        from peft import PeftModel
        print(f"Loading adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.size = {
        "shortest_edge": args.image_size * args.image_size // 4,
        "longest_edge":  args.image_size * args.image_size,
    }
    model.eval()
    print(f"Model ready. Image size: {args.image_size}px")

    with open(args.input) as f:
        all_samples = [json.loads(line) for line in f]
    if args.limit:
        all_samples = all_samples[: args.limit]

    # Filter by train/val split if requested
    if args.id_file and args.split:
        with open(args.id_file) as f:
            split_ids = set(json.load(f)[args.split])
        all_samples = [s for s in all_samples if s["id"] in split_ids]
        print(f"Filtered to {args.split} split: {len(all_samples)} samples")

    # Assign this shard's slice
    samples = [s for i, s in enumerate(all_samples) if i % args.num_shards == args.shard]
    pending = [s for s in samples if s["id"] not in done_ids]
    total   = len(samples)
    skipped = total - len(pending)
    print(f"[shard {args.shard}/{args.num_shards}] {len(pending)} pending, {skipped} skipped")
    def prepare_inputs(sample):
        """CPU-side work: load images, tokenize. Returns (sample, inputs_cpu) or (sample, None) if missing."""
        image_paths = [data_root / p for p in sample["images"]]
        if any(not p.exists() for p in image_paths):
            return sample, None
        num_future = len(sample["images"]) - 1
        question = (
            sample["question"]
            + f"\nOutput EXACTLY {num_future} integers as a Python list."
            + f"\nExample format: {list(range(1, num_future + 1))}"
        )
        content = [{"type": "image", "image": str(p)} for p in image_paths]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs_cpu = processor(text=[text], images=image_inputs, return_tensors="pt", padding=True)
        return sample, inputs_cpu

    speeds: list[float] = []
    wall_start = time.perf_counter()

    with open(output_path, "a" if args.resume else "w") as out_f:
        with ThreadPoolExecutor(max_workers=args.prefetch) as pool:
            futures = {}
            # Pre-submit first batch
            submit_idx = 0
            for _ in range(min(args.prefetch, len(pending))):
                fut = pool.submit(prepare_inputs, pending[submit_idx])
                futures[fut] = submit_idx
                submit_idx += 1

            for i in range(len(pending)):
                # Wait for next ready future (FIFO via order list)
                fut = next(iter(futures))
                del futures[fut]
                sample, inputs_cpu = fut.result()

                # Submit next item to keep prefetch queue full
                if submit_idx < len(pending):
                    nf = pool.submit(prepare_inputs, pending[submit_idx])
                    futures[nf] = submit_idx
                    submit_idx += 1

                if inputs_cpu is None:
                    print(f"[{i+1}/{len(pending)}] SKIP — missing images")
                    continue

                inputs = inputs_cpu.to(model.device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
                elapsed = time.perf_counter() - t0

                out_trimmed = out[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(out_trimmed, skip_special_tokens=True)[0].strip()
                tok_per_sec = out_trimmed.shape[1] / elapsed
                speeds.append(tok_per_sec)

                answer = parse_answer(raw)
                has_labels = "gt_answer" in sample
                record = {**sample, "answer": answer} if has_labels else {"id": sample["id"], "answer": answer}
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

                done_count    = i + 1
                wall_elapsed  = time.perf_counter() - wall_start
                rate          = done_count / wall_elapsed
                remaining     = (len(pending) - done_count) / rate if rate > 0 else 0
                eta           = f"{remaining/3600:.1f}h" if remaining > 3600 else f"{remaining/60:.0f}m"

                print(
                    f"[{skipped+done_count}/{total}] {sample['id'][:50]}  →  {answer}"
                    f"  ({elapsed:.1f}s, {tok_per_sec:.1f} tok/s, ETA {eta})"
                )

    if speeds:
        avg = sum(speeds) / len(speeds)
        print(f"\nSpeed  avg {avg:.1f} tok/s  |  min {min(speeds):.1f}  max {max(speeds):.1f}")
    total_wall = time.perf_counter() - wall_start
    print(f"Wall time: {total_wall/60:.1f} min  |  Output: {output_path}")
    print(f"Evaluate with:  enact eval {output_path}")


if __name__ == "__main__":
    main()
