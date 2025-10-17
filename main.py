#!/usr/bin/env python3
"""
smartguard_user_feedback_nodup_multitest.py

Changes vs prior:
- Feedback is NOT stored as a separate USER message. It's carried in variables and
  injected ONLY into the NEXT USER prompt. This prevents "double feedback."
- The injected banner reads "=== FEEDBACK ON YOUR LAST PREDICTION ===".
- Warm-up is still first contract per label with metadata; no predictions/metrics/CSV.
- REAL mode: pre-creates 7 label folders under results_root and memory_root.
- TEST mode: supports multiple test labels (Overflow-Underflow + tx.origin), each
  with its own *_TEST results/memory directories.

CLI (REAL):
  python smartguard_user_feedback_nodup_multitest.py \
    --mode real \
    --contracts_root /path/to/buggy_contracts \
    --results_root ./results \
    --memory_root ./memory \
    --api_key $OPENAI_API_KEY

CLI (TEST) — auto-detects these roots if present:
  test/Overflow-Underflow
  test/tx.origin
  (override with --test_roots "pathA;pathB")
"""

import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

# Optional: only needed for REAL mode
try:
    import openai  # type: ignore
except Exception:
    openai = None

# ---------------- Defaults ----------------
BASE_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
HISTORY_TURNS_DEFAULT = 4

LABEL_FOLDERS = {
    "Re-entrancy": "Re-entrancy",
    "Timestamp dep": "Timestamp-Dependency",
    "Unchecked-send": "Unchecked-Send",
    "Unhandled exp": "Unhandled-Exceptions",
    "TOD": "TOD",
    "Integer flow": "Overflow-Underflow",
    "tx.origin": "tx.origin",
}

# ---------------- Utilities ----------------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def ensure_label_dirs(base: Path) -> None:
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


def precision_recall(pred: List[int], truth: List[int]) -> Tuple[float, float, int, int, int]:
    ps, ts = set(pred), set(truth)
    TP = len(ps & ts)
    FP = len(ps - ts)
    FN = len(ts - ps)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    return prec, rec, TP, FP, FN


def f1_from_pr(p: float, r: float) -> float:
    return 2*p*r/(p+r) if (p+r) else 0.0


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

# ---------------- Messaging helpers ----------------


def instruction_block(label_name: str) -> str:
    return (
        "You are an expert in smart contract vulnerability detection.\n"
        "A Solidity smart contract will be provided; identify all vulnerabilities present in the code.\n"
        f"Focus specifically on detecting instances of {label_name}.\n"
        "Return ONLY the line numbers (comma-separated or newline-separated) when asked to predict.\n"
        "For the first contract of each label, DO NOT output predictions. Reply only: "
        "'I understand the patterns and I'm ready for the next contract'."
    )


def ensure_user_instruction(memory_chat: List[Dict], label_name: str) -> None:
    if not memory_chat or memory_chat[0].get("role") != "user":
        memory_chat.insert(
            0, {"role": "user", "content": instruction_block(label_name)})


def compress_recent_systems(memory_chat: List[Dict], last_n: int) -> List[str]:
    systems = [m.get("content", "")
               for m in memory_chat if m.get("role") == "system"]
    return systems[-last_n:] if last_n > 0 else []


def build_user_contract_prompt(contract_text: str, injected_feedback: Optional[str]) -> str:
    parts = []
    if injected_feedback:
        parts.append("=== FEEDBACK ON YOUR LAST PREDICTION ===\n" +
                     injected_feedback.strip())
    parts.append(
        "Analyze the following contract. Return only the line numbers.")
    parts.append("--- CONTRACT ---\n" + contract_text)
    return "\n\n".join(parts)


def build_user_warmup_prompt(contract_text: str, truth_lines: List[int]) -> str:
    truth_str = ",".join(map(str, truth_lines)) if truth_lines else "(none)"
    return (
        "=== WARM-UP (DO NOT PREDICT) ===\n"
        "Read and internalize the pattern. Do not output any line numbers.\n"
        "Reply only: 'I understand the patterns and I'm ready for the next contract'.\n\n"
        "=== METADATA (ground-truth lines for learning) ===\n"
        f"{truth_str}\n\n"
        "=== CONTRACT ===\n"
        f"{contract_text}"
    )


def build_messages_for_attempt(memory_chat: List[Dict],
                               user_content: str,
                               history_turns: int) -> List[Dict]:
    msgs: List[Dict] = []
    for s in compress_recent_systems(memory_chat, history_turns):
        msgs.append({"role": "system", "content": s})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def append_turn(memory_chat: List[Dict], user_prompt: str, system_msg: str, analysis_text: Optional[str]) -> None:
    memory_chat.append({"role": "user", "content": user_prompt})
    memory_chat.append({"role": "system", "content": system_msg})
    if analysis_text:
        memory_chat.append({"role": "user", "content": analysis_text})

# ---------------- LLM ----------------


def call_llm_messages(messages: List[Dict], api_key: str) -> str:
    if openai is None:
        raise RuntimeError(
            "openai package not installed; cannot call LLM in REAL mode.")
    openai.api_key = api_key
    for _ in range(3):
        try:
            resp = openai.ChatCompletion.create(
                model=BASE_MODEL, messages=messages, temperature=TEMPERATURE
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            print(f"[WARN] LLM API error: {e}, retrying...")
            time.sleep(2)
    raise RuntimeError("LLM API failed after 3 retries")

# ---------------- Persistence (legacy migration kept minimal) ----------------


def load_or_init_chat(memory_path: Path, label_name: str) -> List[Dict]:
    if not memory_path.exists():
        return [{"role": "user", "content": instruction_block(label_name)}]
    raw = json.load(open(memory_path))
    chat = raw if isinstance(raw, list) else []
    ensure_user_instruction(chat, label_name)
    return chat

# ---------------- REAL pipeline ----------------


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

    memory_chat = load_or_init_chat(memory_path, folder_name)
    print(
        f"[INFO] Loaded memory chat for {label_key} ({len(memory_chat)} messages).")

    sol_files = sorted(label_dir.glob("buggy_*.sol"))
    if not sol_files:
        print(f"[WARN] No contracts found in {label_dir}")
        return

    # Warm-up (first contract with metadata, no eval/CSV)
    warm = sol_files[0]
    m0 = re.search(r'buggy_(\d+)\.sol', warm.name)
    idx0 = int(m0.group(1)) if m0 else -1
    meta0 = label_dir / f"BugLog_{idx0}.csv"
    truth0 = read_vuln_lines_from_csv(meta0) if meta0.exists() else []
    warm_user = build_user_warmup_prompt(read_text(warm), truth0)
    warm_msgs = build_messages_for_attempt(
        memory_chat, warm_user, history_turns)
    print(
        f"\n[REAL::{folder_name}] {warm.name} | WARM-UP (no predictions expected)")
    warm_sys = call_llm_messages(warm_msgs, api_key).strip() or \
        "I understand the patterns and I'm ready for the next contract."
    append_turn(memory_chat, warm_user, warm_sys, analysis_text=None)
    json.dump(memory_chat, open(memory_path, "w"), indent=2)
    print(f"   SYSTEM (warm-up) → {warm_sys}")

    # No feedback computed for warm-up
    carried_feedback_across_contracts: Optional[str] = None

    # Normal loop (from contract #2 onward)
    for sol_file in sol_files[1:]:
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

        # feedback variables (not stored as USER message)
        immediate_feedback: Optional[str] = carried_feedback_across_contracts

        while True:
            attempt += 1
            user_prompt = build_user_contract_prompt(
                contract_text, immediate_feedback)
            messages = build_messages_for_attempt(
                memory_chat, user_prompt, history_turns)

            print(
                f"\n[REAL::{folder_name}] {sol_file.name} | Attempt {attempt}")
            llm_text = call_llm_messages(messages, api_key).strip()
            nums = re.findall(r'\b\d+\b', llm_text)

            if not nums:
                # Unexpected ack/empty; store and move on
                sys_msg = llm_text or "[prediction] "
                append_turn(memory_chat, user_prompt,
                            sys_msg, analysis_text=None)
                json.dump(memory_chat, open(memory_path, "w"), indent=2)
                print(f"   SYSTEM (no numbers) → {sys_msg}")
                break

            pred_lines = sorted({int(x) for x in nums})
            prec, rec, *_ = precision_recall(pred_lines, truth_lines)
            f1x100 = round(100.0 * f1_from_pr(prec, rec), 2)
            fb = feedback_from_pr(prec, rec, threshold)

            sys_msg = f"[prediction] {','.join(str(x) for x in pred_lines)}"
            analysis_msg = f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x100:.2f}"

            append_turn(memory_chat, user_prompt, sys_msg, analysis_msg)
            json.dump(memory_chat, open(memory_path, "w"), indent=2)

            print(f"   SYSTEM → {sys_msg}")
            print(
                f"   EVAL   → P={prec:.3f}, R={rec:.3f}, F1*100={f1x100:.2f}")

            if (prec >= threshold and rec >= threshold) or attempt >= max_attempts:
                final_lines, final_prec, final_rec, final_f1x100 = pred_lines, prec, rec, f1x100
                carried_feedback_across_contracts = fb     # used by next contract
                immediate_feedback = None                  # clear for safety
                break
            else:
                # retry same contract → carry the *new* feedback to the next attempt
                immediate_feedback = fb
                continue

        # Save final prediction CSV for this contract
        out_csv = results_dir / f"{sol_file.stem}_pred.csv"
        pd.DataFrame({"predicted_lines": final_lines}
                     ).to_csv(out_csv, index=False)
        print(
            f"   FINAL → P={final_prec:.3f}, R={final_rec:.3f}, F1*100={final_f1x100:.2f} (saved {out_csv.name})")

# ---------------- TEST pipeline (multi-label) ----------------


def process_one_test_label(test_root: Path,
                           results_root: Path,
                           memory_root: Path,
                           label_display: str,
                           history_turns: int,
                           threshold: float):
    results_dir = results_root / label_display
    mem_dir = memory_root / label_display
    ensure_dir(results_dir)
    ensure_dir(mem_dir)
    memory_path = mem_dir / "memory_full.json"

    # Infer the label name from folder name for instruction text
    label_name_for_instruction = "tx.origin" if "tx.origin" in str(
        test_root) else "Overflow-Underflow"
    memory_chat = load_or_init_chat(memory_path, label_name_for_instruction)
    print(
        f"[INFO][TEST:{label_display}] Loaded memory chat ({len(memory_chat)} messages).")

    sol_files = sorted(test_root.glob("buggy_*.sol"))
    if not sol_files:
        print(
            f"[WARN][TEST:{label_display}] No contracts found in {test_root}")
        return

    # Warm-up
    warm = sol_files[0]
    m0 = re.search(r'buggy_(\d+)\.sol', warm.name)
    idx0 = int(m0.group(1)) if m0 else -1
    meta0 = test_root / f"BugLog_{idx0}.csv"
    truth0 = read_vuln_lines_from_csv(meta0) if meta0.exists() else []
    warm_user = build_user_warmup_prompt(read_text(warm), truth0)
    msgs = build_messages_for_attempt(memory_chat, warm_user, history_turns)
    # TEST: deterministic ack
    sys_msg = "I understand the patterns and I'm ready for the next contract."
    append_turn(memory_chat, warm_user, sys_msg, analysis_text=None)
    json.dump(memory_chat, open(memory_path, "w"), indent=2)
    print(f"[TEST:{label_display}] Warm-up complete.")

    # Normal loop (no retries in TEST to keep it deterministic)
    carried_feedback: Optional[str] = None
    for sol_file in sol_files[1:]:
        m = re.search(r'buggy_(\d+)\.sol', sol_file.name)
        if not m:
            print(
                f"[SKIP][TEST:{label_display}] Non-matching filename: {sol_file.name}")
            continue
        idx = int(m.group(1))
        meta_file = test_root / f"BugLog_{idx}.csv"
        if not meta_file.exists():
            print(
                f"[WARN][TEST:{label_display}] Missing metadata for {sol_file.name}")
            continue

        contract_text = read_text(sol_file)
        truth_lines = read_vuln_lines_from_csv(meta_file)

        # Load sample predictions if present: buggy_{i}_pred.csv or BugLog_{i}_pred.csv
        pred_csv = None
        for name in (f"buggy_{idx}_pred.csv", f"BugLog_{idx}_pred.csv"):
            cand = test_root / "sample_preds" / name
            if cand.exists():
                pred_csv = cand
                break

        user_prompt = build_user_contract_prompt(
            contract_text, carried_feedback)
        if not pred_csv:
            # No sample prediction → store empty prediction (no eval saved)
            append_turn(memory_chat, user_prompt,
                        "[prediction] ", analysis_text=None)
            json.dump(memory_chat, open(memory_path, "w"), indent=2)
            print(
                f"[WARN][TEST:{label_display}] Missing sample prediction for {sol_file.name}")
            continue

        df = pd.read_csv(pred_csv)
        col = df.columns[0] if len(df.columns) else None
        pred_lines = sorted({int(x) for x in df[col].dropna().astype(
            str).str.extract(r'(\d+)')[0].dropna().astype(int)}) if col else []

        prec, rec, *_ = precision_recall(pred_lines, truth_lines)
        f1x100 = round(100.0 * f1_from_pr(prec, rec), 2)
        fb = feedback_from_pr(prec, rec, threshold)

        sys_msg = f"[prediction] {','.join(str(x) for x in pred_lines)}"
        analysis_msg = f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x100:.2f}"
        append_turn(memory_chat, user_prompt, sys_msg, analysis_msg)
        json.dump(memory_chat, open(memory_path, "w"), indent=2)

        carried_feedback = fb

        out_csv = results_dir / f"{sol_file.stem}_pred.csv"
        pd.DataFrame({"predicted_lines": pred_lines}
                     ).to_csv(out_csv, index=False)
        print(
            f"[TEST:{label_display}] {sol_file.name} → P={prec:.3f}, R={rec:.3f}, F1*100={f1x100:.2f} (saved {out_csv.name})")

# ---------------- Main ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "test"], default="real")
    ap.add_argument("--contracts_root",
                    help="Root of labeled contract folders (required in real mode).")
    ap.add_argument("--results_root", required=True,
                    help="Where to save final prediction CSVs.")
    ap.add_argument("--memory_root", required=True,
                    help="Where to save full chat memory JSONs.")
    ap.add_argument("--api_key", required=False,
                    help="LLM API key (required in real mode).")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--max_attempts", type=int, default=3)
    ap.add_argument("--history_turns", type=int, default=HISTORY_TURNS_DEFAULT)
    # TEST multi-root support (semicolon-separated)
    ap.add_argument("--test_roots", default="",
                    help="Semicolon-separated test roots (default: auto-detect).")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    ensure_dir(results_root)
    memory_root = Path(args.memory_root)
    ensure_dir(memory_root)

    if args.mode == "test":
        # auto-detect two common test roots if not provided
        if args.test_roots.strip():
            roots = [Path(p.strip())
                     for p in args.test_roots.split(";") if p.strip()]
        else:
            roots = []
            default_a = Path("test/Overflow-Underflow")
            default_b = Path("test/tx.origin")
            if default_a.exists():
                roots.append(default_a)
            if default_b.exists():
                roots.append(default_b)
            if not roots:
                print(
                    "[ERROR][TEST] No test roots found. Provide --test_roots or create test folders.")
                return

        for r in roots:
            label_display = "tx.origin_TEST" if "tx.origin" in str(
                r) else "Overflow-Underflow_TEST"
            process_one_test_label(
                test_root=r,
                results_root=results_root,
                memory_root=memory_root,
                label_display=label_display,
                history_turns=args.history_turns,
                threshold=args.threshold
            )
        print("\n✅ TEST mode completed.")
        return

    # REAL mode
    if not args.contracts_root:
        raise SystemExit("--contracts_root is required for real mode.")
    if not args.api_key:
        raise SystemExit("--api_key is required for real mode.")

    ensure_label_dirs(results_root)
    ensure_label_dirs(memory_root)

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
