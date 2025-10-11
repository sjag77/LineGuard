#!/usr/bin/env python3
"""
smartguard_adaptive_trainer_demo.py
===================================

Two-score scheme focused on Precision & Recall:

- Per-contract metrics:
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    combined_per_contract_score = F1 * 100
- Threshold applies to **precision** and **recall** independently:
    if precision < TH or recall < TH -> retry (REAL mode), with **dynamic feedback**.

Dynamic feedback rules sent in the next prompt:
  1) P < TH and R < TH -> "Be more precise and find all vulnerabilities."
  2) P < TH and R >= TH -> "Be more careful choosing the exact line numbers."
  3) P >= TH and R < TH -> "You chose good lines; now find all remaining vulnerabilities."
  4) P >= TH and R >= TH -> "Great work." (move on)

Overall score: simple average of **final** per-contract scores (F1*100), priming not counted.

Modes:
- REAL: iterate full dataset; retry below-threshold up to --max_retries.
- TEST: test/Overflow-Underflow with 5 contracts, sample predictions for 2..5; no retries.

Usage (TEST):
  python smartguard_adaptive_trainer_demo.py \
    --mode test \
    --test_root "test/Overflow-Underflow" \
    --test_pred_root "test/Overflow-Underflow/sample_preds" \
    --results_root "./results_test" \
    --memory_root "./memory_test" \
    --api_key "DUMMY"

Usage (REAL):
  python smartguard_adaptive_trainer_demo.py \
    --mode real \
    --contracts_root "/path/to/buggy_contracts" \
    --results_root "/path/to/results" \
    --memory_root "/path/to/memory" \
    --api_key "$OPENAI_API_KEY" \
    --threshold 0.7 \
    --max_retries 1
"""

import re
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Optional for REAL mode
try:
    import openai  # type: ignore
except Exception:
    openai = None

# ---------------- Defaults ----------------
BASE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3

LABEL_FOLDERS = {
    "Re-entrancy": "Re-entrancy",
    "Timestamp dep": "Timestamp-Dependency",
    "Unchecked-send": "Unchecked-Send",
    "Unhandled exp": "Unhandled-Exceptions",
    "TOD": "TOD",
    "Integer flow": "Overflow-Underflow",
    "tx.origin": "tx.origin"
}

# ---------------- Utilities ----------------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def read_vuln_lines_from_csv(meta_path: Path) -> List[int]:
    df = pd.read_csv(meta_path)
    cols = [c for c in df.columns if re.search(r'line|loc|linen', c, re.I)]
    if not cols and df.shape[1] == 1:
        cols = [df.columns[0]]
    lines = set()
    for c in cols:
        for v in df[c].dropna().astype(str):
            for part in re.split(r'[;, \|]+', v.strip()):
                if not part:
                    continue
                if part.isdigit():
                    lines.add(int(part))
                else:
                    m = re.search(r'(\d+)', part)
                    if m:
                        lines.add(int(m.group(1)))
    return sorted(lines)


def parse_pred_csv_lines(pred_csv: Path) -> List[int]:
    """Read the first column of CSV and extract ints robustly."""
    if not pred_csv.exists():
        return []
    df = pd.read_csv(pred_csv)
    if df.empty:
        return []
    col = df.columns[0]
    vals: List[int] = []
    for v in df[col].dropna().astype(str):
        if v.isdigit():
            vals.append(int(v))
        else:
            m = re.search(r'(\d+)', v)
            if m:
                vals.append(int(m.group(1)))
    return sorted(set(vals))


def precision_recall(pred: List[int], truth: List[int]) -> Tuple[float, float, int, int, int]:
    pred_set, truth_set = set(pred), set(truth)
    TP = len(pred_set & truth_set)
    FP = len(pred_set - truth_set)
    FN = len(truth_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    return prec, rec, TP, FP, FN


def f1_from_pr(prec: float, rec: float) -> float:
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def call_llm(prompt: str, api_key: str) -> str:
    if openai is None:
        raise RuntimeError(
            "openai package not installed; cannot call LLM in REAL mode.")
    openai.api_key = api_key
    for _ in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=BASE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            print(f"[WARN] LLM API error: {e}, retrying...")
            time.sleep(2)
    raise RuntimeError("LLM API failed after 3 retries")


def memory_to_text(memory: List[Dict], last_n: int = 5) -> str:
    text = ""
    for turn in memory[-last_n:]:
        p = f", P={turn['precision']:.2f}" if 'precision' in turn else ""
        r = f", R={turn['recall']:.2f}" if 'recall' in turn else ""
        s = f", F1*100={turn['per_contract_score']:.2f}" if 'per_contract_score' in turn else ""
        o = f", overall={turn['overall_score']:.2f}" if 'overall_score' in turn else ""
        fb = f"\n[Feedback]: {turn['feedback']}" if 'feedback' in turn and turn['feedback'] else ""
        text += f"\n[User]: {turn['user']}\n[LLM{p}{r}{s}{o}]: {turn['llm']}{fb}\n"
    return text


def feedback_from_pr(prec: float, rec: float, threshold: float) -> str:
    bad_p = prec < threshold
    bad_r = rec < threshold
    if bad_p and bad_r:
        return ("Your precision and recall are both below the target. "
                "Be more precise in choosing exact line numbers AND ensure you find all vulnerabilities.")
    if bad_p and not bad_r:
        return ("Your recall is good but precision is below target. "
                "Be more careful selecting the exact line numbers to better match ground truth.")
    if not bad_p and bad_r:
        return ("Your precision is good but recall is below target. "
                "You selected good lines but missed some vulnerabilities—find all remaining ones.")
    return "Great work—both precision and recall met the target. Proceed to the next contract."


def build_instruction(label_name: str,
                      overall_score_so_far: Optional[float] = None,
                      last_contract_score: Optional[float] = None,
                      last_precision: Optional[float] = None,
                      last_recall: Optional[float] = None,
                      last_feedback: Optional[str] = None) -> str:
    parts = [
        "You are an expert in smart contract vulnerability detection.",
        "A Solidity smart contract will be provided, and your task is to identify all vulnerabilities present in the code.",
        f"Specifically, focus on detecting instances of {label_name}.",
        "Your output must follow the structure and text format of the metadata (a list of vulnerable line numbers).",
        "The most critical factor is the accuracy of the **line number** where the vulnerability occurs.",
        "If you detect a vulnerability on line 30 but ground-truth indicates 29, bias toward 29.",
    ]
    if overall_score_so_far is not None:
        parts.append(
            f"Overall accuracy so far (F1*100): {overall_score_so_far:.2f}.")
    if last_contract_score is not None:
        parts.append(
            f"Last contract score (F1*100): {last_contract_score:.2f}.")
    if last_precision is not None and last_recall is not None:
        parts.append(
            f"Last precision: {last_precision:.2f}, last recall: {last_recall:.2f}.")
    if last_feedback:
        parts.append(f"Guidance from last iteration: {last_feedback}")
    parts.append(
        "Aim for the best possible alignment with metadata. Return only the line numbers.")
    return "\n".join(parts)


def build_prompt(label_name: str,
                 contract_text: str,
                 memory_text: str,
                 overall_score_so_far: Optional[float],
                 last_contract_score: Optional[float],
                 last_precision: Optional[float],
                 last_recall: Optional[float],
                 last_feedback: Optional[str]) -> str:
    return (
        memory_text
        + "\n"
        + build_instruction(label_name, overall_score_so_far,
                            last_contract_score, last_precision, last_recall, last_feedback)
        + "\n\n--- CONTRACT ---\n"
        + contract_text
    )


def find_sample_pred_csv(test_pred_root: Path, i: int) -> Optional[Path]:
    candidates = [
        test_pred_root / f"buggy_{i}_pred.csv",
        test_pred_root / f"BugLog_{i}_pred.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

# ---------------- REAL mode ----------------


def process_label_real(label_key: str,
                       folder_name: str,
                       contracts_root: Path,
                       results_root: Path,
                       memory_root: Path,
                       api_key: str,
                       threshold: float,
                       max_retries: int):
    label_dir = contracts_root / folder_name
    if not label_dir.exists():
        print(f"[SKIP] Missing directory: {label_dir}")
        return

    results_dir = results_root / folder_name
    mem_dir = memory_root / folder_name
    ensure_dir(results_dir)
    ensure_dir(mem_dir)
    memory_path = mem_dir / "memory_full.json"

    # Load existing memory
    memory: List[Dict] = []
    if memory_path.exists():
        try:
            memory = json.load(open(memory_path))
            print(
                f"[INFO] Loaded memory for {label_key} ({len(memory)} turns).")
        except Exception:
            memory = []

    # Overall reconstruction from memory (counted only)
    overall_sum, overall_count = 0.0, 0
    for turn in memory:
        if turn.get("counted", False):
            overall_sum += float(turn.get("per_contract_score", 0))
            overall_count += 1
    overall_score = round(overall_sum / overall_count,
                          2) if overall_count else None

    last_contract_score = None
    last_precision, last_recall = None, None
    last_feedback = memory[-1].get("feedback") if memory else None
    if memory:
        if 'per_contract_score' in memory[-1]:
            last_contract_score = float(memory[-1]['per_contract_score'])
        if 'precision' in memory[-1] and 'recall' in memory[-1]:
            last_precision = float(memory[-1]['precision'])
            last_recall = float(memory[-1]['recall'])

    sol_files = sorted(label_dir.glob("buggy_*.sol"))
    if not sol_files:
        print(f"[WARN] No contracts found in {label_dir}")
        return

    for sol_file in sol_files:
        m = re.search(r'buggy_(\d+)\.sol', sol_file.name)
        if not m:
            print(f"[SKIP] Non-matching filename: {sol_file.name}")
            continue
        idx = int(m.group(1))
        meta_file = label_dir / f"BugLog_{idx}.csv"
        if not meta_file.exists():
            print(f"[WARN] Missing metadata for {sol_file.name}")
            continue

        contract_text = read_text(sol_file)
        truth_lines = read_vuln_lines_from_csv(meta_file)

        # attempt loop
        attempt = 0
        final_pred_lines: List[int] = []
        final_llm_answer = ""
        final_prec = 0.0
        final_rec = 0.0
        final_score = 0.0
        feedback = last_feedback or ""

        while True:
            attempt += 1
            mem_text = memory_to_text(memory)
            prompt = build_prompt(
                folder_name, contract_text, mem_text,
                overall_score, last_contract_score, last_precision, last_recall, feedback
            )
            print(
                f"\n[REAL::{folder_name}] {sol_file.name} | Attempt {attempt}")
            llm_answer = call_llm(prompt, api_key)

            # Parse prediction from free text → store as CSV lines
            pred_lines = sorted({int(x)
                                for x in re.findall(r'\b\d+\b', llm_answer)})
            prec, rec, TP, FP, FN = precision_recall(pred_lines, truth_lines)
            f1 = f1_from_pr(prec, rec)
            score = round(100.0 * f1, 2)

            # Craft feedback for next iteration decision
            feedback = feedback_from_pr(prec, rec, threshold)

            # Log this attempt (not counted yet)
            memory.append({
                "contract_id": idx,
                "attempt": attempt,
                "user": prompt,
                "llm": llm_answer,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "per_contract_score": score,
                "overall_score": overall_score if overall_score is not None else 0.0,
                "feedback": feedback,
                "counted": False
            })
            json.dump(memory, open(memory_path, "w"), indent=2)

            print(
                f"   P={prec:.3f}, R={rec:.3f}, F1*100={score:.2f} | feedback: {feedback}")

            # Threshold check on P & R independently
            if (prec >= threshold and rec >= threshold) or attempt >= (max_retries + 1):
                final_pred_lines = pred_lines
                final_llm_answer = llm_answer
                final_prec, final_rec, final_score = prec, rec, score
                break
            # else: loop again with tailored feedback embedded via memory context

        # mark final attempt as counted, update overall
        overall_sum += final_score
        overall_count += 1
        overall_score = round(overall_sum / overall_count, 2)

        memory[-1]["counted"] = True
        memory[-1]["overall_score"] = overall_score
        json.dump(memory, open(memory_path, "w"), indent=2)

        # Save final prediction CSV (after retries if any)
        out_csv = (results_dir / f"{sol_file.stem}_pred.csv")
        pd.DataFrame({"predicted_lines": final_pred_lines}
                     ).to_csv(out_csv, index=False)
        print(
            f"   FINAL → P={final_prec:.3f}, R={final_rec:.3f}, F1*100={final_score:.2f} | overall={overall_score:.2f} (saved {out_csv.name})")

        # prepare context for next contract
        last_contract_score = final_score
        last_precision, last_recall = final_prec, final_rec
        last_feedback = feedback

# ---------------- TEST mode ----------------


def process_test_mode(test_root: Path,
                      test_pred_root: Path,
                      results_root: Path,
                      memory_root: Path):
    """
    TEST layout:
      test_root/buggy_1.sol ... buggy_5.sol
      test_root/BugLog_1.csv ... BugLog_5.csv
      test_pred_root/buggy_2_pred.csv ... buggy_5_pred.csv (or BugLog_*.csv)
    Priming:
      - memory['llm'] = FULL TEXT of buggy_2_pred.csv for the first entry (contract 1).
      - No retries in TEST mode.
    """
    label_display = "Overflow-Underflow_TEST"
    results_dir = results_root / label_display
    mem_dir = memory_root / label_display
    ensure_dir(results_dir)
    ensure_dir(mem_dir)
    memory_path = mem_dir / "memory_full.json"

    memory: List[Dict] = []
    if memory_path.exists():
        try:
            memory = json.load(open(memory_path))
            print(
                f"[INFO][TEST] Loaded existing memory ({len(memory)} turns).")
        except Exception as e:
            print(f"[WARN][TEST] Could not load memory: {e}. Starting fresh.")

    sol_files = [test_root / f"buggy_{i}.sol" for i in range(1, 6)]
    meta_files = [test_root / f"BugLog_{i}.csv" for i in range(1, 6)]
    for p in sol_files + meta_files:
        if not p.exists():
            print(f"[ERROR][TEST] Missing file: {p}")

    # Overall reconstruction
    overall_sum, overall_count = 0.0, 0
    for turn in memory:
        if turn.get("counted", False):
            overall_sum += float(turn.get("per_contract_score", 0))
            overall_count += 1
    overall_score = round(overall_sum / overall_count,
                          2) if overall_count else None
    last_contract_score = memory[-1].get(
        "per_contract_score") if memory else None
    last_precision = memory[-1].get("precision") if memory else None
    last_recall = memory[-1].get("recall") if memory else None
    last_feedback = memory[-1].get("feedback") if memory else None

    # 1) Priming (contract 1)
    if sol_files[0].exists():
        sol1_text = read_text(sol_files[0])
        mem_text = memory_to_text(memory)
        prompt1 = build_prompt("Overflow-Underflow", sol1_text, mem_text,
                               overall_score, last_contract_score, last_precision, last_recall, last_feedback)
        pred_csv_for_2 = find_sample_pred_csv(test_pred_root, 2)
        if pred_csv_for_2 and pred_csv_for_2.exists():
            priming_llm = read_text(pred_csv_for_2)
        else:
            priming_llm = "ERROR: Missing sample prediction CSV for contract 2."
            print(f"[WARN][TEST] {priming_llm}")
        memory.append({
            "contract_id": 1,
            "attempt": 1,
            "user": prompt1,
            "llm": priming_llm,
            "precision": 1.0,
            "recall": 1.0,
            "per_contract_score": 100.0,  # priming default
            "overall_score": overall_score if overall_score is not None else 0.0,
            "feedback": "Priming completed.",
            "counted": False  # not counted
        })
        json.dump(memory, open(memory_path, "w"), indent=2)
        last_contract_score = 100.0
        last_precision, last_recall = 1.0, 1.0
        last_feedback = "Priming completed."
        print("[TEST] Priming complete.")
    else:
        print("[ERROR][TEST] Missing buggy_1.sol; aborting priming.")
        return

    # 2) Contracts 2..5 (no retries)
    for i in range(2, 6):
        sol_path = sol_files[i - 1]
        meta_path = meta_files[i - 1]
        if not sol_path.exists() or not meta_path.exists():
            warn = f"[WARN][TEST] Skipping buggy_{i}: missing files."
            print(warn)
            memory.append({
                "contract_id": i,
                "attempt": 1,
                "user": warn,
                "llm": "",
                "precision": 0.0,
                "recall": 0.0,
                "per_contract_score": 0.0,
                "overall_score": overall_score if overall_score is not None else 0.0,
                "feedback": "Missing files.",
                "counted": False
            })
            json.dump(memory, open(memory_path, "w"), indent=2)
            continue

        truth_lines = read_vuln_lines_from_csv(meta_path)
        pred_csv = find_sample_pred_csv(test_pred_root, i)
        if not (pred_csv and pred_csv.exists()):
            warn = f"[WARN][TEST] Missing sample prediction CSV for contract {i}."
            print(warn)
            memory.append({
                "contract_id": i,
                "attempt": 1,
                "user": warn,
                "llm": "",
                "precision": 0.0,
                "recall": 0.0,
                "per_contract_score": 0.0,
                "overall_score": overall_score if overall_score is not None else 0.0,
                "feedback": "Missing sample prediction.",
                "counted": False
            })
            json.dump(memory, open(memory_path, "w"), indent=2)
            continue

        mem_text = memory_to_text(memory)
        prompt = build_prompt("Overflow-Underflow",
                              read_text(sol_path),
                              mem_text,
                              overall_score,
                              last_contract_score,
                              last_precision,
                              last_recall,
                              last_feedback)

        # store raw CSV text in memory
        llm_answer_text = read_text(pred_csv)
        pred_lines = parse_pred_csv_lines(pred_csv)   # parse for metrics
        prec, rec, TP, FP, FN = precision_recall(pred_lines, truth_lines)
        f1 = f1_from_pr(prec, rec)
        per_score = round(100.0 * f1, 2)
        # test uses a nominal 0.7 for messaging
        feedback = feedback_from_pr(prec, rec, threshold=0.7)

        # Save standardized output CSV
        out_pred = results_dir / f"buggy_{i}_pred.csv"
        pd.DataFrame({"predicted_lines": pred_lines}
                     ).to_csv(out_pred, index=False)

        # Update overall
        if overall_score is None:
            overall_sum, overall_count = per_score, 1
        else:
            # reconstruct on the fly
            overall_sum, overall_count = overall_score * len([t for t in memory if t.get("counted", False)]), \
                len([t for t in memory if t.get("counted", False)])
            overall_sum += per_score
            overall_count += 1
        overall_score = round(overall_sum / overall_count, 2)

        # Log memory
        memory.append({
            "contract_id": i,
            "attempt": 1,
            "user": prompt,
            "llm": llm_answer_text,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "per_contract_score": per_score,
            "overall_score": overall_score,
            "feedback": feedback,
            "counted": True
        })
        json.dump(memory, open(memory_path, "w"), indent=2)

        print(f"[TEST] {sol_path.name} → P={prec:.3f}, R={rec:.3f}, F1*100={per_score:.2f} | overall={overall_score:.2f} (saved {out_pred.name})")

        # next context
        last_contract_score = per_score
        last_precision, last_recall = prec, rec
        last_feedback = feedback

# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "test"], default="real",
                    help="Run mode: real (7 labels) or test (5-contract demo).")
    ap.add_argument("--contracts_root",
                    help="Root of labeled contract folders (required in real mode).")
    ap.add_argument("--results_root", required=True,
                    help="Where to save final prediction CSVs.")
    ap.add_argument("--memory_root", required=True,
                    help="Where to save memory JSONs.")
    ap.add_argument("--api_key", required=True, help="LLM API key.")
    ap.add_argument("--threshold", type=float, default=0.7,  # precision/recall threshold in [0,1]
                    help="Per-metric threshold applied to both precision and recall.")
    ap.add_argument("--max_retries", type=int, default=1,
                    help="Max retries per contract when below threshold (REAL mode).")
    # TEST mode paths
    ap.add_argument("--test_root", default="test/Overflow-Underflow",
                    help="Test folder with 5 contracts + metadata.")
    ap.add_argument("--test_pred_root", default="test/Overflow-Underflow/sample_preds",
                    help="Folder with sample predictions (buggy_2..5_pred.csv or BugLog_2..5_pred.csv).")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    memory_root = Path(args.memory_root)
    ensure_dir(results_root)
    ensure_dir(memory_root)

    if args.mode == "test":
        process_test_mode(
            test_root=Path(args.test_root),
            test_pred_root=Path(args.test_pred_root),
            results_root=results_root,
            memory_root=memory_root
        )
        print("\n✅ TEST mode completed.")
        return

    # REAL mode
    if not args.contracts_root:
        raise SystemExit("--contracts_root is required for real mode.")
    contracts_root = Path(args.contracts_root)

    for label_key, folder_name in LABEL_FOLDERS.items():
        print(f"\n=== REAL MODE: Processing Label '{label_key}' ===")
        process_label_real(
            label_key=label_key,
            folder_name=folder_name,
            contracts_root=contracts_root,
            results_root=results_root,
            memory_root=memory_root,
            api_key=args.api_key,
            threshold=args.threshold,
            max_retries=args.max_retries
        )

    print("\n✅ REAL mode completed.")


if __name__ == "__main__":
    main()
