#!/bin/bash
set -e

ADAPTER="./lora_enact_ordering"
SPLIT_IDS="./lora_enact_ordering/split_ids.json"

# ── 1. Train ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "STEP 1: Training"
echo "============================================================"
python scripts/finetune_qwen25vl.py \
    --output "$ADAPTER" \
    --batch-size 12 \
    --grad-accum 2 \
    --image-size 336 \
    --epochs 1

# ── 2. Inference — internal test split (labelled) ─────────────────────────────
echo ""
echo "============================================================"
echo "STEP 2: Inference on internal test split (labelled)"
echo "============================================================"
python scripts/inference_hf.py \
    --adapter "$ADAPTER" \
    --id-file "$SPLIT_IDS" \
    --split test \
    --output test_results.jsonl

# ── 3. Inference — internal val split (labelled) ──────────────────────────────
echo ""
echo "============================================================"
echo "STEP 3: Inference on internal val split (labelled)"
echo "============================================================"
python scripts/inference_hf.py \
    --adapter "$ADAPTER" \
    --id-file "$SPLIT_IDS" \
    --split val \
    --output val_results.jsonl

# ── 4. Inference — challenge test set (unlabelled) ────────────────────────────
echo ""
echo "============================================================"
echo "STEP 4: Inference on challenge test set (unlabelled)"
echo "============================================================"
python scripts/inference_hf.py \
    --adapter "$ADAPTER" \
    --input  ./data/QA_test/enact_ordering_test.jsonl \
    --data-root ./data \
    --output test_challenge_results.jsonl

# ── 5. Evaluate labelled outputs ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo "STEP 5: Evaluation"
echo "============================================================"
echo "--- Internal test split ---"
enact eval test_results.jsonl

echo ""
echo "--- Internal val split ---"
enact eval val_results.jsonl

echo ""
echo "--- Challenge test (no labels — skipping eval) ---"
echo "Submit test_challenge_results.jsonl for official scoring."

echo ""
echo "============================================================"
echo "Pipeline complete."
echo "============================================================"
