"""
Convert HuggingFace PEFT adapter → merged HF model → MLX 4-bit model.

Usage:
    python scripts/convert_adapter_to_mlx.py \
        --adapter ./lora_enact_ordering \
        --mlx-path ./fused_mlx_model

Steps:
    1. Load base HF model (CPU, float16)
    2. Merge PEFT LoRA adapter
    3. Save merged HF model to disk
    4. Convert to MLX 4-bit via mlx_vlm.convert
    5. Delete intermediate merged HF model

Disk required: ~14GB (merged HF) + ~4GB (MLX 4-bit) = ~18GB free
RAM required:  ~14GB peak (model load + merge)
"""

import argparse
import gc
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter",    default="./lora_enact_ordering")
    parser.add_argument("--mlx-path",   default="./fused_mlx_model")
    parser.add_argument("--merged-path", default="./merged_hf_model",
                        help="Temp dir for merged HF model (deleted after conversion)")
    parser.add_argument("--q-bits",     default=4, type=int, choices=[4, 8],
                        help="Quantization bits for MLX model (default 4)")
    parser.add_argument("--keep-merged", action="store_true",
                        help="Keep the intermediate merged HF model (default: delete)")
    args = parser.parse_args()

    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    merged_path = Path(args.merged_path)
    mlx_path    = Path(args.mlx_path)

    # ── Step 1: Load base model on CPU ────────────────────────────────────────
    print(f"Loading base model: {args.base_model}  (CPU, float16)")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    # ── Step 2: Merge adapter ─────────────────────────────────────────────────
    print(f"Loading and merging adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    print("Adapter merged.")

    # ── Step 3: Save merged HF model ──────────────────────────────────────────
    merged_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {merged_path}  (~14GB)...")
    model.save_pretrained(merged_path)

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(merged_path)

    # Free RAM before conversion subprocess
    del model, processor
    gc.collect()
    print("RAM freed.")

    # ── Step 4: Convert to MLX ────────────────────────────────────────────────
    print(f"\nConverting to MLX {args.q_bits}-bit → {mlx_path}")
    cmd = [
        sys.executable, "-m", "mlx_vlm", "convert",
        "--hf-path",  str(merged_path),
        "--mlx-path", str(mlx_path),
        "--quantize",
        "--q-bits",   str(args.q_bits),
    ]
    result = subprocess.run(cmd, check=True)

    # ── Step 5: Cleanup ───────────────────────────────────────────────────────
    if not args.keep_merged:
        print(f"Deleting intermediate merged model at {merged_path}...")
        shutil.rmtree(merged_path)

    print(f"\nDone! MLX model at {mlx_path}")
    print(f"\nRun inference:")
    print(f"  python scripts/inference_mlx.py --model {mlx_path} \\")
    print(f"      --input data/QA_dev/enact_ordering_dev.jsonl \\")
    print(f"      --data-root data \\")
    print(f"      --output enact_dev_finetuned.jsonl")


if __name__ == "__main__":
    main()
