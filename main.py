#!/usr/bin/env python3
"""
smartguard_user_feedback_system_predictions_only_vfinal.py

Key behavior (final):
- SYSTEM replies are ONLY predictions (e.g. "[prediction] 10,20,36,...") or a warm-up ack.
- Metrics + feedback are logged in USER as:
    [analysis] P=... R=... F1*100=... ; [feedback_for_next] ...
- First turn: USER sends base instruction + contract_1 (warm-up). SYSTEM may ack.
- From contract_2 onward: USER sends contract (+ injected feedback), waits for SYSTEM [prediction], evaluates; retry same contract if needed.
- TEST mode uses a single results folder: "Overflow-Underflow_TEST".
- REAL mode PRE-CREATES the 7 label folders under both results_root and memory_root and stores each label's outputs in its same-named folder.

Usage (TEST):
  python smartguard_user_feedback_system_predictions_only_vfinal.py \
    --mode test \
    --test_root "test/Overflow-Underflow" \
    --test_pred_root "test/Overflow-Underflow/sample_preds" \
    --results_root "./results_test" \
    --memory_root "./memory_test" \
    --api_key "DUMMY" \
    --threshold 0.7

Usage (REAL):
  python smartguard_user_feedback_system_predictions_only_vfinal.py \
    --mode real \
    --contracts_root "/path/to/buggy_contracts" \
    --results_root "/path/to/results" \
    --memory_root "/path/to/memory" \
    --api_key "$OPENAI_API_KEY" \
    --threshold 0.7 \
    --max_attempts 3 \
    --history_turns 4
"""

import re
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

# Optional: only needed for REAL mode calls
try:
    import openai  # type: ignore
except Exception:
    openai = None

# ---------------- Defaults ----------------
BASE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
HISTORY_TURNS_DEFAULT = 4

# Map (display → folder_name_on_disk)
LABEL_FOLDERS = {
    "Re-entrancy": "Re-entrancy",
    "Timestamp dep": "Timestamp-Dependency",
    "Unchecked-send": "Unchecked-Send",
    "Unhandled exp": "Unhandled-Exceptions",
    "TOD": "TOD",
    "Integer flow": "Overflow-Underflow",
    "tx.origin": "tx.origin"
}

# ---------------- Utility funcs ----------------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def ensure_label_dirs(base: Path) -> None:
    """Create all 7 label directories under the given base (results or memory)."""
    for folder_name in LABEL_FOLDERS.values():
        ensure_dir(base / folder_name)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def read_vuln_lines_from_csv(meta_path: Path) -> List[int]:
    df = pd.read_csv(meta_path)
    cols = [c for c in df.columns if re.search(r'line|loc|linen', c, re.I)]
    if not cols and df.shape[1] >= 1:
        cols = [df.columns[0]]
    lines = set()
    for c in cols:
        for v in df[c].dropna().astype(str):
            m = re.search(r'(\d+)', v)
            if m:
                lines.add(int(m.group(1)))
    return sorted(lines)


def parse_pred_csv_lines(pred_csv: Path) -> List[int]:
    if not pred_csv or not pred_csv.exists():
        return []
    df = pd.read_csv(pred_csv)
    if df.empty:
        return []
    col = df.columns[0]
    vals: List[int] = []
    for v in df[col].dropna().astype(str):
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


def feedback_from_pr(prec: float, rec: float, threshold: float) -> str:
    bad_p = prec < threshold
    bad_r = rec < threshold
    if bad_p and bad_r:
        return ("Your precision and recall are both below target. "
                "Be more precise in selecting exact line numbers AND ensure you find all vulnerabilities.")
    if bad_p and not bad_r:
        return ("Recall is good but precision is below target. "
                "Be more careful choosing the exact line numbers to better match ground truth.")
    if not bad_p and bad_r:
        return ("Precision is good but recall is below target. "
                "You selected good lines but missed some vulnerabilities—find all remaining ones.")
    return "Great work—both precision and recall met the target. Proceed to the next contract."

# ---------------- 2-role chat helpers ----------------


def instruction_block(label_name: str) -> str:
    return (
        "You are an expert in smart contract vulnerability detection.\n"
        "A Solidity smart contract will be provided; identify all vulnerabilities present in the code.\n"
        f"Focus specifically on detecting instances of {label_name}.\n"
        "Return ONLY the line numbers (comma-separated or newline-separated).\n"
        "When ready, you may first reply 'I understand the patterns and I'm ready for the next contract' or directly give predictions."
    )


def ensure_user_instruction(memory_chat: List[Dict], label_name: str) -> None:
    if not memory_chat or memory_chat[0].get("role") != "user":
        memory_chat.insert(
            0, {"role": "user", "content": instruction_block(label_name)})


def compress_recent_systems(memory_chat: List[Dict], last_n: int) -> List[str]:
    systems = [m.get("content", "")
               for m in memory_chat if m.get("role") == "system"]
    return systems[-last_n:] if last_n > 0 else []


def get_last_user_feedback(memory_chat: List[Dict]) -> Optional[str]:
    for m in reversed(memory_chat):
        if m.get("role") == "user":
            txt = str(m.get("content", "")).strip()
            if txt.startswith("[feedback_for_next]"):
                return txt.replace("[feedback_for_next]", "", 1).strip()
    return None


def build_user_contract_prompt(contract_text: str, injected_feedback: Optional[str]) -> str:
    parts = []
    if injected_feedback:
        parts.append(
            "=== PREVIOUS FEEDBACK (for this attempt) ===\n" + injected_feedback.strip())
    parts.append(
        "Analyze the following contract. Return only the line numbers.")
    parts.append("--- CONTRACT ---\n" + contract_text)
    return "\n\n".join(parts)


def build_messages_for_attempt(memory_chat: List[Dict],
                               contract_text: str,
                               history_turns: int,
                               carry_feedback: Optional[str]) -> List[Dict]:
    msgs: List[Dict] = []
    for s in compress_recent_systems(memory_chat, history_turns):
        msgs.append({"role": "system", "content": s})
    user_prompt = build_user_contract_prompt(contract_text, carry_feedback)
    msgs.append({"role": "user", "content": user_prompt})
    return msgs


def append_attempt_to_chat(memory_chat: List[Dict],
                           user_prompt: str,
                           system_content: str,
                           analysis_and_feedback_user: str) -> None:
    memory_chat.append({"role": "user", "content": user_prompt})
    memory_chat.append({"role": "system", "content": system_content})
    memory_chat.append({"role": "user", "content": analysis_and_feedback_user})

# ---------------- LLM call ----------------


def call_llm_messages(messages: List[Dict], api_key: str) -> str:
    if openai is None:
        raise RuntimeError(
            "openai package not installed; cannot call LLM in REAL mode.")
    openai.api_key = api_key
    for _ in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=TEMPERATURE
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            print(f"[WARN] LLM API error: {e}, retrying...")
            time.sleep(2)
    raise RuntimeError("LLM API failed after 3 retries")

# ---------------- Migration loader ----------------


def migrate_legacy_metrics_array_to_chat(raw: List[Dict], label_name: str) -> List[Dict]:
    chat: List[Dict] = [
        {"role": "user", "content": instruction_block(label_name)}]
    for it in raw:
        cid = int(it.get("contract_id", -1))
        att = int(it.get("attempt", 1))
        prec = float(it.get("precision", 0.0))
        rec = float(it.get("recall", 0.0))
        f1x = float(it.get("per_contract_score", 0.0))
        fb = str(it.get("feedback", "")) or "No feedback available."
        lines = sorted({int(x) for x in re.findall(
            r'\b\d+\b', str(it.get("llm", "")))})
        chat.append(
            {"role": "user", "content": f"(legacy-migrated) contract #{cid} attempt {att}\n<original user content unavailable>"})
        chat.append(
            {"role": "system", "content": f"[prediction] {','.join(str(x) for x in lines)}"})
        chat.append(
            {"role": "user", "content": f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x:.2f} ; [feedback_for_next] {fb}"})
    return chat


def load_or_migrate_chat(memory_path: Path, label_name: str) -> List[Dict]:
    if not memory_path.exists():
        return [{"role": "user", "content": instruction_block(label_name)}]
    raw = json.load(open(memory_path))
    if isinstance(raw, list) and raw and "role" not in raw[0]:
        chat = migrate_legacy_metrics_array_to_chat(raw, label_name)
        json.dump(chat, open(memory_path, "w"), indent=2)
        return chat
    chat = raw if isinstance(raw, list) else []
    ensure_user_instruction(chat, label_name)
    return chat

# ---------------- REAL mode ----------------


def process_label_real(label_key: str,
                       folder_name: str,
                       contracts_root: Path,
                       results_root: Path,
                       memory_root: Path,
                       api_key: str,
                       threshold: float,
                       max_attempts: int,
                       history_turns: int):
    label_dir = contracts_root / folder_name
    if not label_dir.exists():
        print(f"[SKIP] Missing directory: {label_dir}")
        return

    results_dir = results_root / folder_name
    mem_dir = memory_root / folder_name
    ensure_dir(results_dir)
    ensure_dir(mem_dir)
    memory_path = mem_dir / "memory_full.json"

    memory_chat = load_or_migrate_chat(memory_path, folder_name)
    print(
        f"[INFO] Loaded memory chat for {label_key} ({len(memory_chat)} messages).")

    sol_files = sorted(label_dir.glob("buggy_*.sol"))
    if not sol_files:
        print(f"[WARN] No contracts found in {label_dir}")
        return

    carried_feedback_across_contracts = get_last_user_feedback(memory_chat)

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

        attempt = 0
        final_lines: List[int] = []
        final_prec = final_rec = 0.0
        final_f1x100 = 0.0

        immediate_feedback: Optional[str] = carried_feedback_across_contracts

        while True:
            attempt += 1
            messages = build_messages_for_attempt(
                memory_chat=memory_chat,
                contract_text=contract_text,
                history_turns=history_turns,
                carry_feedback=immediate_feedback
            )

            print(
                f"\n[REAL::{folder_name}] {sol_file.name} | Attempt {attempt}")
            llm_text = call_llm_messages(messages, api_key).strip()

            found_nums = re.findall(r'\b\d+\b', llm_text)
            if found_nums:
                pred_lines = sorted({int(x) for x in found_nums})
                prec, rec, TP, FP, FN = precision_recall(
                    pred_lines, truth_lines)
                f1 = f1_from_pr(prec, rec)
                f1x100 = round(100.0 * f1, 2)
                fb = feedback_from_pr(prec, rec, threshold)

                analysis_user = f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x100:.2f} ; [feedback_for_next] {fb}"
                system_content = f"[prediction] {','.join(str(x) for x in pred_lines)}"

                user_prompt_logged = messages[-1]["content"]
                append_attempt_to_chat(
                    memory_chat, user_prompt_logged, system_content, analysis_user)
                json.dump(memory_chat, open(memory_path, "w"), indent=2)

                print(f"   SYSTEM → {system_content}")
                print(
                    f"   EVAL   → P={prec:.3f}, R={rec:.3f}, F1*100={f1x100:.2f}")

                if (prec >= threshold and rec >= threshold) or attempt >= max_attempts:
                    final_lines, final_prec, final_rec, final_f1x100 = pred_lines, prec, rec, f1x100
                    carried_feedback_across_contracts = fb
                    break
                else:
                    immediate_feedback = fb
                    continue
            else:
                # Warm-up / acknowledgement
                system_content = llm_text or "I understand the patterns and I'm ready for the next contract."
                analysis_user = "[analysis] WARM-UP_ACK ; [feedback_for_next] Warm-up acknowledged."
                user_prompt_logged = messages[-1]["content"]
                append_attempt_to_chat(
                    memory_chat, user_prompt_logged, system_content, analysis_user)
                json.dump(memory_chat, open(memory_path, "w"), indent=2)
                carried_feedback_across_contracts = "Warm-up acknowledged."
                break

        out_csv = results_dir / f"{sol_file.stem}_pred.csv"
        pd.DataFrame({"predicted_lines": final_lines}
                     ).to_csv(out_csv, index=False)
        print(
            f"   FINAL → P={final_prec:.3f}, R={final_rec:.3f}, F1*100={final_f1x100:.2f} (saved {out_csv.name})")

# ---------------- TEST mode ----------------


def find_sample_pred_csv(test_pred_root: Path, i: int) -> Optional[Path]:
    for name in (f"buggy_{i}_pred.csv", f"BugLog_{i}_pred.csv"):
        cand = test_pred_root / name
        if cand.exists():
            return cand
    return None


def process_test_mode(test_root: Path,
                      test_pred_root: Path,
                      results_root: Path,
                      memory_root: Path,
                      history_turns: int,
                      threshold: float):
    label_display = "Overflow-Underflow_TEST"
    results_dir = results_root / label_display
    mem_dir = memory_root / label_display
    ensure_dir(results_dir)
    ensure_dir(mem_dir)
    memory_path = mem_dir / "memory_full.json"

    memory_chat = load_or_migrate_chat(memory_path, "Overflow-Underflow")
    print(f"[INFO][TEST] Loaded memory chat ({len(memory_chat)} messages).")

    sol_files = [test_root / f"buggy_{i}.sol" for i in range(1, 6)]
    meta_files = [test_root / f"BugLog_{i}.csv" for i in range(1, 6)]
    for p in sol_files + meta_files:
        if not p.exists():
            print(f"[ERROR][TEST] Missing file: {p}")

    # Priming (contract 1)
    if sol_files[0].exists():
        ctext1 = read_text(sol_files[0])
        injected_fb = get_last_user_feedback(memory_chat)
        user_prompt_1 = build_user_contract_prompt(ctext1, injected_fb)

        pred2_csv = find_sample_pred_csv(test_pred_root, 2)
        if pred2_csv:
            pred2_lines = parse_pred_csv_lines(pred2_csv)
            system_content = f"[prediction] {','.join(str(x) for x in pred2_lines)}"
            truth1 = read_vuln_lines_from_csv(
                meta_files[0]) if meta_files[0].exists() else []
            prec, rec, *_ = precision_recall(pred2_lines, truth1)
            f1x100 = round(100.0 * f1_from_pr(prec, rec), 2)
            fb = feedback_from_pr(prec, rec, threshold)
            analysis_user = f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x100:.2f} ; [feedback_for_next] {fb}"
        else:
            system_content = "I understand the patterns and I'm ready for the next contract."
            analysis_user = "[analysis] WARM-UP_ACK ; [feedback_for_next] Warm-up acknowledged."

        append_attempt_to_chat(memory_chat, user_prompt_1,
                               system_content, analysis_user)
        json.dump(memory_chat, open(memory_path, "w"), indent=2)
        print("[TEST] Priming complete.")
    else:
        print("[ERROR][TEST] Missing buggy_1.sol; aborting test.")
        return

    # Contracts 2..5
    for i in range(2, 6):
        sol_path = sol_files[i-1]
        meta_path = meta_files[i-1]
        if not sol_path.exists() or not meta_path.exists():
            print(f"[WARN][TEST] Skipping buggy_{i}: missing files.")
            memory_chat.append({"role": "system", "content": "[prediction] "})
            memory_chat.append(
                {"role": "user", "content": "[analysis] Missing files ; [feedback_for_next] Missing files."})
            json.dump(memory_chat, open(memory_path, "w"), indent=2)
            continue

        ctext = read_text(sol_path)
        injected_fb = get_last_user_feedback(memory_chat)
        user_prompt = build_user_contract_prompt(ctext, injected_fb)

        pred_csv = find_sample_pred_csv(test_pred_root, i)
        if not pred_csv:
            print(
                f"[WARN][TEST] Missing sample prediction CSV for contract {i}.")
            memory_chat.append({"role": "user", "content": user_prompt})
            memory_chat.append({"role": "system", "content": "[prediction] "})
            memory_chat.append(
                {"role": "user", "content": "[analysis] P=0.0000 R=0.0000 F1*100=0.00 ; [feedback_for_next] Missing sample prediction."})
            json.dump(memory_chat, open(memory_path, "w"), indent=2)
            continue

        pred_lines = parse_pred_csv_lines(pred_csv)
        truth_lines = read_vuln_lines_from_csv(meta_path)
        prec, rec, *_ = precision_recall(pred_lines, truth_lines)
        f1x100 = round(100.0 * f1_from_pr(prec, rec), 2)
        fb = feedback_from_pr(prec, rec, threshold)
        system_content = f"[prediction] {','.join(str(x) for x in pred_lines)}"
        analysis_user = f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x100:.2f} ; [feedback_for_next] {fb}"

        append_attempt_to_chat(memory_chat, user_prompt,
                               system_content, analysis_user)
        json.dump(memory_chat, open(memory_path, "w"), indent=2)

        out_csv = results_dir / f"{sol_path.stem}_pred.csv"
        pd.DataFrame({"predicted_lines": pred_lines}
                     ).to_csv(out_csv, index=False)
        print(
            f"[TEST] {sol_path.name} → P={prec:.3f}, R={rec:.3f}, F1*100={f1x100:.2f} (saved {out_csv.name})")

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
                    help="Where to save full chat memory JSONs.")
    ap.add_argument("--api_key", required=True, help="LLM API key.")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="Per-metric threshold in [0,1] applied to both precision and recall.")
    ap.add_argument("--max_attempts", type=int, default=3,
                    help="Maximum attempts per contract (first try + retries).")
    ap.add_argument("--history_turns", type=int, default=HISTORY_TURNS_DEFAULT,
                    help="How many recent SYSTEM messages to include in the next request.")
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
        # Single TEST folder
        process_test_mode(
            test_root=Path(args.test_root),
            test_pred_root=Path(args.test_pred_root),
            results_root=results_root,
            memory_root=memory_root,
            history_turns=args.history_turns,
            threshold=args.threshold
        )
        print("\n✅ TEST mode completed.")
        return

    # REAL mode → create ALL seven label folders up-front (results + memory)
    ensure_label_dirs(results_root)
    ensure_label_dirs(memory_root)

    if not args.contracts_root:
        raise SystemExit("--contracts_root is required for real mode.")
    contracts_root = Path(args.contracts_root)

    for label_key, folder_name in LABEL_FOLDERS.items():
        print(
            f"\n=== REAL MODE: Processing Label '{label_key}' → folder '{folder_name}' ===")
        process_label_real(
            label_key=label_key,
            folder_name=folder_name,
            contracts_root=contracts_root,
            results_root=results_root,
            memory_root=memory_root,
            api_key=args.api_key,
            threshold=args.threshold,
            max_attempts=args.max_attempts,
            history_turns=args.history_turns
        )

    print("\n✅ REAL mode completed.")


if __name__ == "__main__":
    main()
