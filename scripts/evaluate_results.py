"""
Evaluate fine-tuned model predictions and generate accuracy graphs.

Usage:
    python scripts/evaluate_results.py
    python scripts/evaluate_results.py --val val_results.jsonl --test test_results.jsonl
    python scripts/evaluate_results.py --output-dir ./plots
"""

import argparse
import ast
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── Metrics ───────────────────────────────────────────────────────────────────

def parse_answer(raw):
    if isinstance(raw, list):
        return raw
    try:
        return ast.literal_eval(str(raw))
    except Exception:
        return None


def task_accuracy(pred, gt):
    """1 if predicted sequence exactly matches ground truth, else 0."""
    p = parse_answer(pred)
    g = parse_answer(gt)
    if p is None or g is None or len(p) != len(g):
        return None
    return int(p == g)


def pairwise_accuracy(pred, gt):
    """
    Fraction of (i,j) pairs whose relative order matches between pred and gt.
    For a sequence of length n there are n*(n-1)/2 pairs.
    """
    p = parse_answer(pred)
    g = parse_answer(gt)
    if p is None or g is None or len(p) != len(g) or len(p) < 2:
        return None

    # Build rank lookup: element → position
    rank_pred = {v: i for i, v in enumerate(p)}
    rank_gt   = {v: i for i, v in enumerate(g)}

    correct = total = 0
    for a, b in combinations(g, 2):
        if a not in rank_pred or b not in rank_pred:
            continue
        gt_order   = rank_gt[a]   < rank_gt[b]
        pred_order = rank_pred[a] < rank_pred[b]
        correct += int(gt_order == pred_order)
        total   += 1

    return correct / total if total > 0 else None


# ── Loading ───────────────────────────────────────────────────────────────────

def load_results(path):
    records = []
    for line in open(path):
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def compute_stats(records):
    """Return dicts keyed by image_count with lists of (task_acc, pair_acc)."""
    by_len   = defaultdict(lambda: {"task": [], "pair": []})
    by_type  = defaultdict(lambda: {"task": [], "pair": []})
    overall  = {"task": [], "pair": []}

    for r in records:
        pred = r.get("answer")
        gt   = r.get("gt_answer")
        if pred is None or gt is None:
            continue

        ta = task_accuracy(pred, gt)
        pa = pairwise_accuracy(pred, gt)
        if ta is None:
            continue

        n_images = len(r.get("images", []))
        task_type = r.get("type", r.get("task_name", "unknown"))
        # Shorten type label
        if "forward" in task_type:
            short_type = "forward"
        elif "inverse" in task_type:
            short_type = "inverse"
        else:
            short_type = task_type

        by_len[n_images]["task"].append(ta)
        by_type[short_type]["task"].append(ta)
        overall["task"].append(ta)

        if pa is not None:
            by_len[n_images]["pair"].append(pa)
            by_type[short_type]["pair"].append(pa)
            overall["pair"].append(pa)

    return by_len, by_type, overall


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "val":  {"task": "#2196F3", "pair": "#64B5F6"},
    "test": {"task": "#F44336", "pair": "#EF9A9A"},
    "dev":  {"task": "#4CAF50", "pair": "#A5D6A7"},
}

def plot_by_length(splits_data, output_dir):
    """Line chart: accuracy vs image count for each split."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle("Accuracy vs Step Length", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes, ["task", "pair"], ["Task Accuracy", "Pairwise Accuracy"]):
        for split_name, (by_len, _, _) in splits_data.items():
            lengths = sorted(by_len.keys())
            accs    = [np.mean(by_len[l][metric]) * 100 for l in lengths]
            counts  = [len(by_len[l][metric]) for l in lengths]

            ax.plot(lengths, accs, marker="o", linewidth=2,
                    color=COLORS[split_name]["task"],
                    label=f"{split_name} (n={sum(counts)})")

            # Annotate each point with sample count
            for x, y, n in zip(lengths, accs, counts):
                ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7,
                            color=COLORS[split_name]["task"])

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Number of Images (Step Length)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(range(3, 11))
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out = output_dir / "accuracy_by_length.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_by_type(splits_data, output_dir):
    """Grouped bar chart: task vs pairwise accuracy by task type."""
    fig, axes = plt.subplots(1, len(splits_data), figsize=(7 * len(splits_data), 5), sharey=True)
    if len(splits_data) == 1:
        axes = [axes]
    fig.suptitle("Accuracy by Task Type", fontsize=14, fontweight="bold")

    for ax, (split_name, (_, by_type, _)) in zip(axes, splits_data.items()):
        types  = sorted(by_type.keys())
        task_accs = [np.mean(by_type[t]["task"]) * 100 for t in types]
        pair_accs = [np.mean(by_type[t]["pair"]) * 100 if by_type[t]["pair"] else 0 for t in types]
        counts    = [len(by_type[t]["task"]) for t in types]

        x = np.arange(len(types))
        w = 0.35
        bars1 = ax.bar(x - w/2, task_accs, w, label="Task Accuracy",
                       color=COLORS[split_name]["task"], alpha=0.85)
        bars2 = ax.bar(x + w/2, pair_accs, w, label="Pairwise Accuracy",
                       color=COLORS[split_name]["pair"], alpha=0.85)

        for bar, val in zip(list(bars1) + list(bars2), task_accs + pair_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{split_name.capitalize()} Split", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(types, counts)], fontsize=10)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 112)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

    plt.tight_layout()
    out = output_dir / "accuracy_by_type.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_overall_summary(splits_data, output_dir):
    """Bar chart: overall task + pairwise accuracy per split."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("Overall Accuracy Summary", fontsize=14, fontweight="bold")

    split_names = list(splits_data.keys())
    task_accs   = [np.mean(splits_data[s][2]["task"]) * 100 for s in split_names]
    pair_accs   = [np.mean(splits_data[s][2]["pair"]) * 100 for s in split_names]
    counts      = [len(splits_data[s][2]["task"]) for s in split_names]

    x = np.arange(len(split_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, task_accs, w, label="Task Accuracy",    color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + w/2, pair_accs, w, label="Pairwise Accuracy", color="#4CAF50", alpha=0.85)

    for bar, val in zip(list(bars1) + list(bars2), task_accs + pair_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s.capitalize()}\n(n={c})" for s, c in zip(split_names, counts)], fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11)

    plt.tight_layout()
    out = output_dir / "overall_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_table(splits_data):
    print("\n" + "="*65)
    print(f"{'Split':<8} {'N':>5}  {'Task Acc':>10}  {'Pairwise Acc':>13}")
    print("="*65)
    for split_name, (by_len, by_type, overall) in splits_data.items():
        ta = np.mean(overall["task"]) * 100
        pa = np.mean(overall["pair"]) * 100
        n  = len(overall["task"])
        print(f"{split_name:<8} {n:>5}  {ta:>9.2f}%  {pa:>12.2f}%")
        print(f"  By step length:")
        for length in sorted(by_len.keys()):
            lta = np.mean(by_len[length]["task"]) * 100
            lpa = np.mean(by_len[length]["pair"]) * 100
            ln  = len(by_len[length]["task"])
            print(f"    {length:2d} images: task={lta:6.2f}%  pair={lpa:6.2f}%  (n={ln})")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val",        default="val_results.jsonl")
    parser.add_argument("--test",       default="test_results.jsonl")
    parser.add_argument("--dev",        default=None, help="Challenge dev set results (optional)")
    parser.add_argument("--output-dir", default="./plots")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_data = {}
    candidates = [("val", args.val), ("test", args.test)]
    if args.dev:
        candidates.append(("dev", args.dev))
    for split_name, path in candidates:
        p = Path(path)
        if not p.exists():
            print(f"Skipping {split_name} — {path} not found")
            continue
        records = load_results(p)
        print(f"Loaded {len(records)} records from {path}")
        splits_data[split_name] = compute_stats(records)

    if not splits_data:
        print("No result files found.")
        return

    print_table(splits_data)
    plot_overall_summary(splits_data, output_dir)
    plot_by_length(splits_data, output_dir)
    plot_by_type(splits_data, output_dir)
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
