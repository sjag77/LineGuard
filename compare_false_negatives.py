#!/usr/bin/env python3
"""
compare_false_negatives.py

Compute label-level False Negatives (FNs) from predictions in results_test/*_TEST
by comparing to ground-truth metadata in corresponding test/* folders.

Outputs:
  - results_test/false_negative_summary.csv
  - results_test/false_negative_comparison.png

Usage (auto-detects test roots Overflow-Underflow and tx.origin if present):
  python compare_false_negatives.py --results_root results_test

Custom test roots (semicolon-separated):
  python compare_false_negatives.py --results_root results_test \
      --test_roots "test/Overflow-Underflow;test/tx.origin"

Notes:
- Expects prediction files:
    results_test/<Label>_TEST/buggy_<i>_pred.csv
  and ground truth files:
    test/<Label>/BugLog_<i>.csv
- FN per contract = |GroundTruthLines - PredictedLines|
- Label totals aggregate across that label’s contracts.
"""

import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Utilities ----------


def read_truth_lines(meta_csv: Path) -> set[int]:
    """Read ground-truth vulnerable line numbers from a BugLog CSV."""
    if not meta_csv.exists():
        return set()
    df = pd.read_csv(meta_csv)
    if df.empty:
        return set()
    # Try to locate columns containing line info
    cols = [c for c in df.columns if re.search(r'line|loc|linen', c, re.I)]
    if not cols:
        cols = [df.columns[0]]  # fallback to first column
    vals = set()
    for c in cols:
        for v in df[c].dropna().astype(str):
            m = re.search(r'(\d+)', v)
            if m:
                vals.add(int(m.group(1)))
    return vals


def read_pred_lines(pred_csv: Path) -> set[int]:
    """Read predicted line numbers from a *_pred.csv (first column)."""
    if not pred_csv.exists():
        return set()
    df = pd.read_csv(pred_csv)
    if df.empty or df.shape[1] == 0:
        return set()
    col = df.columns[0]
    vals = set()
    for v in df[col].dropna().astype(str):
        m = re.search(r'(\d+)', v)
        if m:
            vals.add(int(m.group(1)))
    return vals


def infer_label_from_results_folder(folder_name: str) -> str:
    """Strip the '_TEST' suffix to get base label name (e.g., 'Overflow-Underflow')."""
    return folder_name[:-5] if folder_name.endswith("_TEST") else folder_name


def discover_test_roots(test_roots_arg: str) -> dict[str, Path]:
    """Build a mapping from label folder name to its test root path."""
    mapping: dict[str, Path] = {}
    if test_roots_arg.strip():
        for part in test_roots_arg.split(";"):
            p = Path(part.strip())
            if not p:
                continue
            mapping[p.name] = p
    else:
        # Auto-detect common test roots
        for name in ["Overflow-Underflow", "tx.origin"]:
            p = Path("test") / name
            if p.exists():
                mapping[name] = p
    return mapping

# ---------- Main logic ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", required=True,
                    help="Path to results_test directory.")
    ap.add_argument("--test_roots", default="",
                    help="Semicolon-separated paths to test label roots (default: auto-detect).")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    if not results_root.exists():
        raise SystemExit(f"[ERROR] results_root not found: {results_root}")

    test_root_map = discover_test_roots(args.test_roots)
    if not test_root_map:
        print("[WARN] No test roots detected/provided. Will attempt naive inference from results folders.")

    # Scan label folders in results_root
    label_dirs = [p for p in results_root.iterdir() if p.is_dir()
                  and p.name.endswith("_TEST")]
    if not label_dirs:
        raise SystemExit(
            f"[ERROR] No '*_TEST' label folders found in {results_root}")

    rows = []
    for label_dir in label_dirs:
        display_label = label_dir.name                # e.g. Overflow-Underflow_TEST
        base_label = infer_label_from_results_folder(
            display_label)  # Overflow-Underflow
        # Find corresponding test root
        test_root = test_root_map.get(base_label, Path("test") / base_label)
        if not test_root.exists():
            print(
                f"[WARN] Skipping {display_label}: test root not found at {test_root}")
            continue

        total_fn = 0
        total_truth = 0
        total_pred = 0
        contracts_counted = 0

        # Iterate over predictions in this label folder
        for pred_csv in sorted(label_dir.glob("*_pred.csv")):
            # Expect filenames like buggy_2_pred.csv
            m = re.search(r'buggy_(\d+)_pred\.csv$', pred_csv.name)
            if not m:
                # allow also 'BugLog_<i>_pred.csv'
                m = re.search(r'BugLog_(\d+)_pred\.csv$', pred_csv.name)
            if not m:
                print(
                    f"[WARN] Unrecognized filename, skipping: {pred_csv.name}")
                continue
            idx = int(m.group(1))

            meta_csv = test_root / f"BugLog_{idx}.csv"
            truth = read_truth_lines(meta_csv)
            pred = read_pred_lines(pred_csv)

            fn = len(truth - pred)
            total_fn += fn
            total_truth += len(truth)
            total_pred += len(pred)
            contracts_counted += 1

        rows.append({
            "label": base_label,
            "contracts": contracts_counted,
            "false_negatives": total_fn,
            "total_truth_lines": total_truth,
            "total_pred_lines": total_pred,
            "fn_rate_vs_truth": (total_fn / total_truth) if total_truth else 0.0
        })

    if not rows:
        raise SystemExit(
            "[ERROR] No data aggregated. Check your folder structure.")

    df = pd.DataFrame(rows).sort_values(
        ["false_negatives", "label"], ascending=[False, True])

    # Save CSV summary
    summary_csv = results_root / "false_negative_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[OK] Wrote summary → {summary_csv}")

    # Plot bar chart of FNs per label
    try:
        plt.figure(figsize=(10, 5))
        x = df["label"]
        y = df["false_negatives"]
        plt.bar(x, y)
        plt.ylabel("False Negatives (count)")
        plt.title("False Negatives per Label (TEST)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        out_png = results_root / "false_negative_comparison.png"
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Wrote chart → {out_png}")
    except Exception as e:
        print(f"[WARN] Could not render chart: {e}")

    # Pretty print to console
    print("\n=== False Negative Summary (by label) ===")
    with pd.option_context('display.max_colwidth', None, 'display.width', 120):
        print(df.to_string(index=False, formatters={
              "fn_rate_vs_truth": "{:.2%}".format}))


if __name__ == "__main__":
    main()
