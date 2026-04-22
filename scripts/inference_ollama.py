"""
Run ENACT benchmark inference using a local Ollama model (e.g. qwen2.5vl:7b).

Usage:
    python scripts/inference_ollama.py --model qwen2.5vl:7b
    python scripts/inference_ollama.py --model qwen2.5vl:7b --workers 3
    python scripts/inference_ollama.py --model qwen2.5vl:7b --limit 50 --output my_output.jsonl
"""

import argparse
import base64
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DATA_ROOT = Path(__file__).parent.parent / "data"
QA_FILE = DATA_ROOT / "QA" / "enact_ordering.jsonl"
OLLAMA_URL = "http://localhost:11434/api/chat"


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_ollama(model: str, question: str, image_paths: list[Path]) -> tuple[str, float]:
    images_b64 = [encode_image(p) for p in image_paths]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question, "images": images_b64}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 30},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    content = data["message"]["content"].strip()
    eval_tokens = data.get("eval_count", 0)
    eval_ns = data.get("eval_duration", 0)
    tok_per_sec = (eval_tokens / eval_ns * 1e9) if eval_ns > 0 else 0.0
    return content, tok_per_sec


def parse_answer(raw: str) -> str:
    match = re.search(r"\[[\d,\s]+\]", raw)
    return match.group(0) if match else raw


def process_sample(sample: dict, model: str, data_root: Path) -> dict | None:
    image_paths = [data_root / p for p in sample["images"]]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        return {"_skip": True, "id": sample["id"], "reason": f"missing images: {missing}"}

    t0 = time.perf_counter()
    try:
        raw, tok_per_sec = query_ollama(model, sample["question"], image_paths)
    except Exception as e:
        return {"_error": True, "id": sample["id"], "reason": str(e)}
    elapsed = time.perf_counter() - t0

    return {
        **sample,
        "answer": parse_answer(raw),
        "_elapsed": elapsed,
        "_tok_per_sec": tok_per_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Ollama model name")
    parser.add_argument("--input", default=str(QA_FILE), help="Path to enact_ordering.jsonl")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Root dir for image paths")
    parser.add_argument("--output", default=None, help="Output JSONL path (auto-named if omitted)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run (for testing)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel request workers (default 1)")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed IDs in output file")
    args = parser.parse_args()

    model_slug = args.model.replace(":", "-").replace("/", "-")
    output_path = Path(args.output) if args.output else Path(f"enact_ordering_{model_slug}.jsonl")
    data_root = Path(args.data_root)

    done_ids: set[str] = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(obj["id"])
        print(f"Resuming — {len(done_ids)} samples already done.")

    with open(args.input) as f:
        samples = [json.loads(line) for line in f]

    if args.limit:
        samples = samples[: args.limit]

    pending = [s for s in samples if s["id"] not in done_ids]
    total = len(samples)
    skipped = total - len(pending)
    print(f"Samples to run: {len(pending)}  |  workers: {args.workers}")

    write_lock = threading.Lock()
    speeds: list[float] = []
    completed = skipped

    out_f = open(output_path, "a" if args.resume else "w")

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_sample, s, args.model, data_root): s for s in pending}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result.get("_skip") or result.get("_error"):
                tag = "SKIP" if result.get("_skip") else "ERROR"
                print(f"[{completed}/{total}] {tag} {result['id'][:55]} — {result['reason']}")
                continue

            elapsed = result.pop("_elapsed")
            tok_per_sec = result.pop("_tok_per_sec")
            speeds.append(tok_per_sec)

            with write_lock:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

            wall_elapsed = time.perf_counter() - wall_start
            rate = (completed - skipped) / wall_elapsed
            remaining = (len(pending) - (completed - skipped)) / rate if rate > 0 else 0
            eta = f"{remaining/3600:.1f}h" if remaining > 3600 else f"{remaining/60:.0f}m"

            print(
                f"[{completed}/{total}] {result['id'][:50]}  →  {result['answer']}"
                f"  ({elapsed:.1f}s, {tok_per_sec:.1f} tok/s, ETA {eta})"
            )

    out_f.close()

    if speeds:
        avg = sum(speeds) / len(speeds)
        print(f"\nSpeed  avg {avg:.1f} tok/s  |  min {min(speeds):.1f}  max {max(speeds):.1f}")
    total_wall = time.perf_counter() - wall_start
    print(f"Wall time: {total_wall/60:.1f} min  |  Output: {output_path}")
    print(f"Evaluate with:  enact eval {output_path}")


if __name__ == "__main__":
    main()
