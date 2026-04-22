"""
Run ENACT benchmark inference using mlx-vlm (fastest on Apple Silicon).
Runs natively on the M-series Neural Engine — expect 3-5s per sample vs ~18s with Ollama.

Setup (one time):
    pip install mlx-vlm

Single process:
    python scripts/inference_mlx.py --resume

4 parallel shards (recommended for 32GB RAM):
    python scripts/inference_mlx.py --shard 0 --num-shards 4 &
    python scripts/inference_mlx.py --shard 1 --num-shards 4 &
    python scripts/inference_mlx.py --shard 2 --num-shards 4 &
    python scripts/inference_mlx.py --shard 3 --num-shards 4 &

Or use the helper script:
    python scripts/inference_mlx.py --run-shards 4
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data"
QA_FILE = DATA_ROOT / "QA" / "enact_ordering.jsonl"
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"


def parse_answer(raw: str) -> str:
    match = re.search(r"\[[\d,\s]+\]", raw)
    return match.group(0) if match else raw


def run_shards(num_shards: int, args: argparse.Namespace):
    """Launch num_shards subprocesses, one per shard, then wait for all."""
    base_cmd = [sys.executable, __file__,
                "--model", args.model,
                "--input", args.input,
                "--data-root", args.data_root,
                "--num-shards", str(num_shards),
                "--max-tokens", str(args.max_tokens)]
    if args.output:
        base_cmd += ["--output", args.output]

    procs = []
    for shard in range(num_shards):
        log_path = Path(f"shard_{shard}.log")
        log_f = open(log_path, "w")
        cmd = base_cmd + ["--shard", str(shard)]
        print(f"Starting shard {shard} → {log_path}")
        procs.append((shard, subprocess.Popen(cmd, stdout=log_f, stderr=log_f), log_f))
        # Stagger launches so each process fully allocates GPU memory before the next starts
        time.sleep(30)

    print(f"\n{num_shards} shards running. Tail logs with:")
    for shard, _, _ in procs:
        print(f"  tail -f shard_{shard}.log")
    print()

    for shard, proc, log_f in procs:
        proc.wait()
        log_f.close()
        print(f"Shard {shard} finished (exit {proc.returncode})")

    print("\nAll shards done. Merging outputs...")
    model_slug = args.model.split("/")[-1].replace(":", "-")
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace mlx-community model ID")
    parser.add_argument("--input", default=str(QA_FILE), help="Path to enact_ordering.jsonl")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Root dir for image paths")
    parser.add_argument("--output", default=None, help="Output JSONL path (auto-named if omitted)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run (for testing)")
    parser.add_argument("--max-tokens", type=int, default=30, help="Max output tokens (default 30)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed IDs in output file")
    parser.add_argument("--shard", type=int, default=0, help="Which shard this process handles (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--run-shards", type=int, default=None,
                        help="Launch this many shards as subprocesses and wait (e.g. --run-shards 4)")
    args = parser.parse_args()

    if args.run_shards:
        run_shards(args.run_shards, args)
        return

    try:
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError:
        print("mlx-vlm not installed. Run:  pip install mlx-vlm")
        raise SystemExit(1)

    model_slug = args.model.split("/")[-1].replace(":", "-")

    # Each shard writes to its own file; --run-shards merges them at the end
    if args.num_shards > 1:
        shard_suffix = f"_shard{args.shard}"
        output_path = Path(args.output).with_suffix("") if args.output else Path(f"enact_ordering_{model_slug}")
        output_path = Path(str(output_path) + shard_suffix + ".jsonl")
    else:
        output_path = Path(args.output) if args.output else Path(f"enact_ordering_{model_slug}.jsonl")

    data_root = Path(args.data_root)

    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        if done_ids:
            print(f"[shard {args.shard}] Resuming — {len(done_ids)} samples already done.")

    with open(args.input) as f:
        all_samples = [json.loads(line) for line in f]
    if args.limit:
        all_samples = all_samples[: args.limit]

    # Assign samples to this shard
    samples = [s for i, s in enumerate(all_samples) if i % args.num_shards == args.shard]
    pending = [s for s in samples if s["id"] not in done_ids]
    total_shard = len(samples)
    skipped = total_shard - len(pending)

    print(f"[shard {args.shard}/{args.num_shards}] Loading model {args.model} ...")
    model, processor = load(args.model)
    config = load_config(args.model)
    print(f"[shard {args.shard}/{args.num_shards}] Model loaded. Samples: {len(pending)} pending, {skipped} skipped")

    speeds: list[float] = []
    wall_start = time.perf_counter()

    with open(output_path, "a") as out_f:
        for i, sample in enumerate(pending):
            image_paths = [str(data_root / p) for p in sample["images"]]
            missing = [p for p in image_paths if not Path(p).exists()]
            if missing:
                print(f"[shard {args.shard}] SKIP {sample['id'][:50]} — missing images")
                continue

            t0 = time.perf_counter()
            try:
                num_future = len(sample["images"]) - 1
                question = (
                    sample["question"]
                    + f"\nOutput EXACTLY {num_future} integers as a Python list."
                    + f"\nExample format: {list(range(1, num_future + 1))}"
                )
                prompt = apply_chat_template(
                    processor, config, question, num_images=len(image_paths)
                )
                result = generate(
                    model, processor, prompt, image_paths,
                    max_tokens=args.max_tokens, verbose=False,
                )
            except Exception as e:
                print(f"[shard {args.shard}] ERROR {sample['id'][:50]}: {e}")
                continue
            elapsed = time.perf_counter() - t0

            tok_per_sec = result.generation_tps
            speeds.append(tok_per_sec)

            answer = parse_answer(result.text)
            out_f.write(json.dumps({**sample, "answer": answer}) + "\n")
            out_f.flush()

            done_count = i + 1
            wall_elapsed = time.perf_counter() - wall_start
            rate = done_count / wall_elapsed
            remaining = (len(pending) - done_count) / rate if rate > 0 else 0
            eta = f"{remaining/3600:.1f}h" if remaining > 3600 else f"{remaining/60:.0f}m"

            print(
                f"[s{args.shard} {skipped+done_count}/{total_shard}] {sample['id'][:45]}  →  {answer}"
                f"  ({elapsed:.1f}s, {tok_per_sec:.1f} tok/s, ETA {eta})"
            )

    if speeds:
        avg = sum(speeds) / len(speeds)
        print(f"\n[shard {args.shard}] Speed  avg {avg:.1f} tok/s  |  min {min(speeds):.1f}  max {max(speeds):.1f}")
    total_wall = time.perf_counter() - wall_start
    print(f"[shard {args.shard}] Wall time: {total_wall/60:.1f} min  |  Output: {output_path}")


if __name__ == "__main__":
    main()
