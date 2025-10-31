#!/usr/bin/env python3
"""
smartguard_user_feedback_system_predictions_only_vfinal.py (refactored, compact, multi-attempt, memory-aware)

- One LLM call per attempt for predictions; OPTIONAL extra LLM call for "smart feedback" summarization (configurable).
- Compact prompt: only Top-K candidates + snippets with configurable radius.
- Attempt #2+: can focus on previous false negatives (Missed-Truth snippets).
- Block-level evaluation: selectable modes (hit/dilated/overlap).
- Line-level evaluation: optional ±tolerance (line_tolerance).
- Multi-label support with compact built-in rules.
- NEW: Memory-aware feedback:
    * Summarize recent feedbacks from memory_chat (local or via LLM) into a short, highly valuable guidance.
    * Inject that concise guidance into the next attempt (instead of dumping long history).
    * Optional pruning/distillation to prevent memory_chat from growing unbounded.

CLI (key args):
  --condense_window (int, default=5)
  --topk_candidates (int, default=40)
  --block_dilation (int, default=1)      # used in 'dilated' mode
  --block_eval {hit,dilated,overlap}     # default=dilated
  --line_tolerance (int, default=0)      # ±w tolerance for line metrics
  --early_stop {block,line,any,perfect_line,both} (default=block)
  --smart_feedback {off,local,llm} (default=llm)   # NEW
  --fb_history_k (int, default=12)                 # NEW: how many recent feedback turns to summarize
  --fb_max_chars (int, default=600)                # NEW: memory summary char limit
  --fb_rule_chars (int, default=180)               # NEW: one-line rule char limit
  --mem_max_msgs (int, default=120)                # NEW: threshold to trigger pruning
  --mem_keep_recent (int, default=24)              # NEW: how many last messages to keep after pruning
  --distill_every (int, default=10)                # NEW: prune/distill frequency (per processed contract per label)

Example:
  python .\main.py --mode real --contracts_root ".\buggy_contracts" --results_root ".\results" --memory_root ".\memory" \
    --api_key $env:OPENAI_API_KEY --threshold 0.7 --max_attempts 3 --history_turns 1 \
    --condense_window 4 --topk_candidates 28 --block_dilation 1 --block_eval dilated --line_tolerance 0 --early_stop any \
    --smart_feedback llm --fb_history_k 12 --fb_max_chars 600 --fb_rule_chars 180 --mem_max_msgs 120 --mem_keep_recent 24 --distill_every 10

"""

import re
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
import pandas as pd

# Logging/Tee dependencies
import sys
import atexit
from datetime import datetime
from dataclasses import dataclass  # <-- used
from collections import Counter  # NEW

# Optional: only needed for REAL mode calls
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

# ---------------- Defaults ----------------
#BASE_MODEL = "gpt-4o"  # Models: GPT-4o/GPT-5"


TEMPERATURE = None
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

# ---------------- Logging: simple Tee ----------------
class _Tee:
    """Duplicate writes to both the console stream and a log file."""
    def __init__(self, stream, logfile_handle):
        self.stream = stream
        self.log = logfile_handle

    def write(self, data):
        try:
            self.stream.write(data)
        except Exception:
            pass
        try:
            self.log.write(data)
        except Exception:
            pass

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

# ---------------- Utility ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def ensure_label_dirs(base: Path) -> None:
    for folder_name in LABEL_FOLDERS.values():
        ensure_dir(base / folder_name)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

# ---------------- Truth loaders ----------------
def read_vuln_lines_from_csv(meta_path: Path) -> List[int]:
    df = pd.read_csv(meta_path)
    cols_lower = {c.lower() for c in df.columns}
    if {"loc", "length"} <= cols_lower:
        loc_col = next(c for c in df.columns if c.lower() == "loc")
        len_col = next(c for c in df.columns if c.lower() == "length")
        lines = set()
        for _, row in df.iterrows():
            try:
                start = int(row[loc_col]); L = int(row[len_col])
                for x in range(start, start + L): lines.add(x)
            except Exception:
                continue
        return sorted(lines)
    cols = [c for c in df.columns if re.search(r'line|loc|linen', c, re.I)]
    if not cols and df.shape[1] >= 1:
        cols = [df.columns[0]]
    lines = set()
    for c in cols:
        for v in df[c].dropna().astype(str):
            m = re.search(r'(\d+)', v)
            if m: lines.add(int(m.group(1)))
    return sorted(lines)

def read_truth(meta_path: Path) -> Dict[str, List[int]]:
    df = pd.read_csv(meta_path)
    cols_lower = {c.lower() for c in df.columns}
    block_lines: Set[int] = set()
    point_lines: Set[int] = set()
    if {"loc", "length"} <= cols_lower:
        loc_col = next(c for c in df.columns if c.lower() == "loc")
        len_col = next(c for c in df.columns if c.lower() == "length")
        for _, row in df.iterrows():
            try:
                start = int(row[loc_col]); L = int(row[len_col])
                for x in range(start, start + L): block_lines.add(x)
            except Exception:
                continue
    line_cols = [c for c in df.columns if c.lower() == "line"]
    for lc in line_cols:
        for v in df[lc].dropna().astype(str):
            m = re.search(r'(\d+)', v)
            if m: point_lines.add(int(m.group(1)))
    return {"block_lines": sorted(block_lines), "point_lines": sorted(point_lines)}

# ---------------- Evaluation helpers ----------------
def _dilate_lines(lines: List[int], w: int) -> Set[int]:
    if w <= 0:
        return set(lines)
    out: Set[int] = set()
    for l in lines:
        for k in range(l - w, l + w + 1):
            if k > 0:
                out.add(k)
    return out

def precision_recall(pred: List[int], truth: List[int]) -> Tuple[float, float, int, int, int]:
    pred_set, truth_set = set(pred), set(truth)
    TP = len(pred_set & truth_set)
    FP = len(pred_set - truth_set)
    FN = len(truth_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    return prec, rec, TP, FP, FN

def precision_recall_with_tol(pred: List[int], truth: List[int], tol: int) -> Tuple[float, float, int, int, int]:
    """
    Line-level PR with ±tol tolerance by dilating predictions (and only predictions).
    If you prefer symmetric dilation, dilate truth as well.
    """
    pred_set = _dilate_lines(pred, tol) if tol > 0 else set(pred)
    truth_set = set(truth)
    TP = len(pred_set & truth_set)
    FP = len(pred_set - truth_set)
    FN = len(truth_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec  = TP / (TP + FN) if (TP + FN) else 0.0
    return prec, rec, TP, FP, FN

def f1_from_pr(prec: float, rec: float) -> float:
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)

def accuracy_from_counts(tp: int, fp: int, fn: int, total_lines: int) -> float:
    tn = max(total_lines - tp - fp - fn, 0)
    return (tp + tn) / total_lines if total_lines > 0 else 0.0

@dataclass
class BlockMetrics:
    prec: float; rec: float; f1: float
    tp: int; fp: int; fn: int
    acc_label: str; acc_value: float   # 'HitRate' when applicable; otherwise ('None', 0.0)

def compute_block_metrics(
    pred_lines: List[int],
    truth_block: List[int],
    mode: str,            # 'hit' | 'dilated' | 'overlap'
    dilation: int
) -> BlockMetrics:
    P = set(pred_lines)
    T = set(truth_block)

    if mode == "hit":
        TP = len(P & T)
        FP = len(P - T)
        FN = len(T - P)
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec  = TP / (TP + FN) if (TP + FN) else 0.0
        f1   = f1_from_pr(prec, rec)
        # HitRate over predictions (same numeric value as precision under this definition)
        return BlockMetrics(prec, rec, f1, TP, FP, FN, "HitRate", prec)

    if mode == "dilated":
        Pd = _dilate_lines(pred_lines, dilation)
        Td = set(truth_block)  # dilate only predictions to reward near hits
        TP = len(Pd & Td)
        FP = len(Pd - Td)
        FN = len(Td - Pd)
        prec = TP / (TP + FP) if (TP + FP) else 0.0
        rec  = TP / (TP + FN) if (TP + FN) else 0.0
        f1   = f1_from_pr(prec, rec)
        return BlockMetrics(prec, rec, f1, TP, FP, FN, "HitRate", prec)

    # mode == 'overlap' : set-overlap proportion (block-as-set)
    inter = len(P & T)
    prec = inter / len(P) if len(P) else 0.0
    rec  = inter / len(T) if len(T) else 0.0
    f1   = f1_from_pr(prec, rec)
    # No sensible 'accuracy' here; mark as none
    return BlockMetrics(prec, rec, f1, inter, len(P) - inter, len(T) - inter, "None", 0.0)

# ---------------- Compact rules per label ----------------
_COMPACT_RULES: Dict[str, Dict[str, List[str]]] = {
    "Re-entrancy": {
        "MUST": [
            "Flag low-level value transfers (.call{value}(), .send(), .transfer) especially after state changes."
        ],
        "INCLUDE": [
            "Mark function signature line if body does external call without a reentrancy guard.",
            "Track state writes (balances/flags) that precede external calls."
        ],
        "NEVER": [
            "Exclude pure events/logs and comments.",
            "Exclude arithmetic-only lines unrelated to external calls."
        ],
    },
    "Timestamp-Dependency": {
        "MUST": ["Flag usages of block.timestamp/now affecting control flow or payouts."],
        "INCLUDE": ["Mark conditions (require/if/loop) driven by timestamp."],
        "NEVER": ["Exclude constant or unused timestamp declarations."],
    },
    "Unchecked-Send": {
        "MUST": ["Flag send/transfer/call.value with unchecked return or missing revert."],
        "INCLUDE": ["Check for missing require(success) after external send."],
        "NEVER": ["Exclude lines already handling success/failure robustly."],
    },
    "Unhandled-Exceptions": {
        "MUST": ["Flag external calls whose return values are ignored without revert path."],
        "INCLUDE": ["call(), delegatecall(), staticcall() without require/handling."],
        "NEVER": ["Exclude try/catch with proper handling."],
    },
    "TOD": {
        "MUST": ["Flag order-dependent read/write patterns around external calls."],
        "INCLUDE": ["State read → external call → state write depending on the read."],
        "NEVER": ["Exclude lines irrelevant to ordering or external interactions."],
    },
    "Overflow-Underflow": {
        "MUST": ["Flag arithmetic updates on balances/allowances without checks."],
        "INCLUDE": ["Unguarded add/sub/mul on critical state variables."],
        "NEVER": ["Exclude compiler-checked arithmetic (>=0.8) unless inside unchecked."],
    },
    "tx.origin": {
        "MUST": ["Flag any use of tx.origin for auth or critical branching."],
        "INCLUDE": ["require/if consuming tx.origin."],
        "NEVER": ["Exclude comments or lookalike identifiers."],
    },
}

def _mode_from_pr(prec: float, rec: float, threshold: float = 0.7) -> str:
    bad_p = prec < threshold
    bad_r = rec < threshold
    if bad_p and bad_r: return "balance"
    if bad_p and not bad_r: return "tighten"
    if not bad_p and bad_r: return "broaden"
    return "stable"

def _snip_line(code_lines: List[str], lno: int, max_len: int = 90) -> str:
    if 1 <= lno <= len(code_lines):
        s = code_lines[lno-1].strip()
        return (s[:max_len] + "…") if len(s) > max_len else s
    return ""

def _pick_k(items: List[int], k: int) -> List[int]:
    return items[:k] if len(items) > k else items

def build_compact_guidance(
    label_name: str,
    contract_text: str,
    pred_lines: List[int],
    truth_block: List[int],
    truth_point: List[int],
    prec_block: float,
    rec_block: float,
    k: int = 2,
    max_chars: int = 900
) -> str:
    code_lines = contract_text.splitlines()
    pred_set, truth_set = set(pred_lines), set(truth_block)
    fp = sorted(pred_set - truth_set)
    fn = sorted(truth_set - pred_set)
    fp_sel = _pick_k(fp, k)
    fn_sel = _pick_k(fn, k)
    fp_snips = [f"L{ln}: {_snip_line(code_lines, ln)}" for ln in fp_sel]
    fn_snips = [f"L{ln}: {_snip_line(code_lines, ln)}" for ln in fn_sel]
    rules = _COMPACT_RULES.get(label_name, {
        "MUST": ["Focus on lines directly implementing the labeled vulnerability."],
        "INCLUDE": ["Prefer lines near external calls or critical state changes."],
        "NEVER": ["Exclude comments, events, and boilerplate."],
    })
    mode = _mode_from_pr(prec_block, rec_block)
    parts: List[str] = []
    parts.append("=== COMPACT HINTS ===")
    parts.append(f"Label: {label_name}")
    parts.append(f"Mode: {mode}")
    parts.append("MUST: " + " ".join(f"- {r}" for r in rules.get("MUST", [])))
    if rules.get("INCLUDE"): parts.append("INCLUDE: " + " ".join(f"- {r}" for r in rules.get("INCLUDE", [])))
    if rules.get("NEVER"):   parts.append("NEVER: " + " ".join(f"- {r}" for r in rules.get("NEVER", [])))
    if fp_snips: parts.append("FalsePositives (avoid): " + " | ".join(fp_snips))
    if fn_snips: parts.append("MissedCandidates (consider): " + " | ".join(fn_snips))
    parts.append("FORMAT: Return only comma-separated integers (e.g., 12,27). No words, no ranges, no JSON.")
    text = "\n".join(parts)
    return text[:max_chars-3] + "..." if len(text) > max_chars else text

# ---------------- Candidate extraction + snippets ----------------
# Comprehensive coverage of: call{value:...}(), call.value(...), send/transfer and even call(...)
_RE_LOWLEVEL_ANY = re.compile(
    r"\.call\s*\("                          # .call(...)
    r"|\.call\s*\{[^}]*value\s*:"           # .call{value:...}
    r"|\.call\.value\s*\("                  # .call.value(...)
    r"|\.send\s*\("                         # .send(...)
    r"|\.transfer\s*\(",                    # .transfer(...)
    re.I
)
_RE_TXORIGIN = re.compile(r"\btx\.origin\b", re.I)
_RE_TIMESTAMP = re.compile(r"\b(block\.timestamp|now)\b", re.I)
_RE_ARITH = re.compile(r"(\+|-|\*|/|<<|>>)", re.I)

def extract_candidates(contract_text: str, label_name: str) -> List[int]:
    lines = contract_text.splitlines()
    out: List[int] = []
    for i, s in enumerate(lines, start=1):
        L = s.lower()
        if "re-entrancy" in label_name.lower():
            if _RE_LOWLEVEL_ANY.search(s) or ("withdraw" in L or "claim" in L or "refund" in L):
                out.append(i)
        elif "tx.origin" in label_name.lower():
            if _RE_TXORIGIN.search(s): out.append(i)
        elif "timestamp" in label_name.lower():
            if _RE_TIMESTAMP.search(s): out.append(i)
        elif "unchecked-send" in label_name.lower() or "unhandled" in label_name.lower():
            if _RE_LOWLEVEL_ANY.search(s): out.append(i)
        elif "overflow" in label_name.lower() or "integer" in label_name.lower():
            if _RE_ARITH.search(s) and ("balance" in L or "allowance" in L or "=" in s):
                out.append(i)
        elif "tod" in label_name.lower():
            if _RE_LOWLEVEL_ANY.search(s) or ("order" in L or "front" in L):
                out.append(i)
        else:
            if _RE_LOWLEVEL_ANY.search(s) or _RE_TXORIGIN.search(s) or _RE_TIMESTAMP.search(s) or _RE_ARITH.search(s):
                out.append(i)
    return sorted(set(out))

def score_candidate_line(s: str, label_name: str) -> int:
    score = 0
    s_low = s.lower()
    if "re-entrancy" in label_name.lower():
        for kw in [".call", ".send", ".transfer"]:
            if kw in s_low: score += 4
        if "balance" in s_low or "flag" in s_low or "=" in s: score += 1
    elif "tx.origin" in label_name.lower():
        if "tx.origin" in s_low: score += 5
    elif "timestamp" in label_name.lower():
        if "block.timestamp" in s_low or " now" in s_low: score += 4
    elif "unchecked-send" in label_name.lower() or "unhandled" in label_name.lower():
        for kw in [".send(", ".transfer(", "call{value", ".call("]:
            if kw in s_low: score += 4
    elif "overflow" in label_name.lower() or "integer" in label_name.lower():
        for kw in ["+", "-", "*", "/", "<<", ">>"]:
            if kw in s_low: score += 1
        if "balance" in s_low or "allowance" in s_low: score += 2
    elif "tod" in label_name.lower():
        if ".call" in s_low or ".send" in s_low or ".transfer" in s_low: score += 3
        if "read" in s_low or "write" in s_low: score += 1
    else:
        if _RE_LOWLEVEL_ANY.search(s): score += 3
        if _RE_TXORIGIN.search(s): score += 3
        if _RE_TIMESTAMP.search(s): score += 2
        if _RE_ARITH.search(s): score += 1
    return score

def rank_candidates(contract_text: str, label_name: str, topk: int) -> List[int]:
    lines = contract_text.splitlines()
    cands = extract_candidates(contract_text, label_name)
    scored = [(i, score_candidate_line(lines[i-1], label_name)) for i in cands]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [i for i,_ in scored[:topk]]

def slice_around(lines: List[str], centers: List[int], radius: int = 5, add_line_numbers: bool = True) -> Tuple[str, List[int]]:
    seen: Set[int] = set()
    chunks: List[str] = []
    covered: List[int] = []
    for c in sorted(set(centers)):
        start = max(1, c - radius); end = min(len(lines), c + radius)
        block = []
        for idx in range(start, end+1):
            if add_line_numbers:
                block.append(f"{idx:>4}: {lines[idx-1]}")
            else:
                block.append(lines[idx-1])
            covered.append(idx)
        key = (start, end)
        if key in seen: continue
        seen.add(key)
        chunks.append("\n".join(block))
    return ("\n\n".join(chunks), covered)

# ---------------- Chat helpers ----------------
def instruction_block(label_name: str) -> str:
    return (
        "You are an expert in smart contract vulnerability detection.\n"
        "A Solidity smart contract will be provided; identify all vulnerabilities present in the code.\n"
        f"Focus specifically on detecting instances of {label_name}.\n"
        "Return ONLY the line numbers.\n"
        "Output format constraint: ONLY digits separated by commas (e.g., 12,27,41). "
        "No words, no ranges, no JSON, no brackets, no explanations.\n"
        "When ready, you may first reply 'I understand the patterns and I'm ready for the next contract' or directly give predictions."
    )

def ensure_user_instruction(memory_chat: List[Dict], label_name: str) -> None:
    if not memory_chat or memory_chat[0].get("role") != "user":
        memory_chat.insert(0, {"role": "user", "content": instruction_block(label_name)})

def compress_recent_systems(memory_chat: List[Dict], last_n: int) -> List[str]:
    systems = [m.get("content", "") for m in memory_chat if m.get("role") == "system"]
    return systems[-last_n:] if last_n > 0 else []

def get_last_user_feedback(memory_chat: List[Dict]) -> Optional[str]:
    for m in reversed(memory_chat):
        if m.get("role") == "user":
            txt = str(m.get("content", "")).strip()
            if txt.startswith("[feedback_for_next]"):
                return txt.replace("[feedback_for_next]", "", 1).strip()
    return None

def build_user_contract_prompt(
    contract_text: str,
    injected_feedback: Optional[str],
    label_name: str,
    attempt_index: int,
    last_pred: Optional[List[int]],
    truth_block: Optional[List[int]],
    condense_window: int,
    topk_candidates: int
) -> str:
    lines = contract_text.splitlines()
    parts: List[str] = []

    # Guidance / feedback (compact hints)
    if injected_feedback:
        parts.append("=== PREVIOUS FEEDBACK (for this attempt) ===\n" + injected_feedback.strip())

    # Rank Top-K candidates and build compact context
    rank = rank_candidates(contract_text, label_name, topk_candidates)
    cand_list = ",".join(str(x) for x in rank) if rank else ""

    if rank:
        code_cand, _ = slice_around(lines, rank, radius=condense_window)
        parts.append("=== CANDIDATE SNIPPETS (Top-K) ===\n" + code_cand)

    if attempt_index >= 2 and last_pred and truth_block:
        fn = sorted(set(truth_block) - set(last_pred))
        if fn:
            code_fn, _ = slice_around(lines, fn[:min(30, len(fn))], radius=max(2, condense_window//2))
            parts.append("=== MISSED-TRUTH SNIPPETS (focus) ===\n" + code_fn)

    # Explicit gating
    parts.append("You MUST choose line numbers ONLY from CandidateLines below.")
    parts.append("FORMAT: Return only comma-separated integers (e.g., 12,27). No words, no ranges, no JSON.")
    parts.append("CandidateLines: " + cand_list)

    # Intentionally we do not send the entire contract to keep the context compact and targeted
    parts.append("Analyze the snippets and output only the matching line numbers.")

    return "\n\n".join(parts)

def build_messages_for_attempt(
    memory_chat: List[Dict],
    contract_text: str,
    history_turns: int,
    carry_feedback: Optional[str],
    label_name: str,
    attempt_index: int,
    last_pred: Optional[List[int]],
    truth_block: Optional[List[int]],
    condense_window: int,
    topk_candidates: int
) -> List[Dict]:
    msgs: List[Dict] = []
    for s in compress_recent_systems(memory_chat, history_turns):
        msgs.append({"role": "system", "content": s})
    user_prompt = build_user_contract_prompt(
        contract_text=contract_text,
        injected_feedback=carry_feedback,
        label_name=label_name,
        attempt_index=attempt_index,
        last_pred=last_pred,
        truth_block=truth_block,
        condense_window=condense_window,
        topk_candidates=topk_candidates
    )
    msgs.append({"role": "user", "content": user_prompt})
    return msgs

def append_attempt_to_chat(
    memory_chat: List[Dict],
    user_prompt: str,
    system_content: str,
    analysis_and_feedback_user: str,
    store_minimal_prompt: bool = False  # NEW (default False to avoid breaking old behavior)
) -> None:
    """
    Append attempt to memory. Optionally store a minimalized user prompt to reduce memory bloat.
    """
    if store_minimal_prompt:
        minimal = "[contract_prompt elided] " + re.sub(r"\s+", " ", user_prompt).strip()[:160]
        memory_chat.append({"role": "user", "content": minimal})
    else:
        memory_chat.append({"role": "user", "content": user_prompt})
    memory_chat.append({"role": "system", "content": system_content})
    memory_chat.append({"role": "user", "content": analysis_and_feedback_user})

# ---------------- LLM call ----------------
def call_llm_messages(messages: List[Dict], api_key: str) -> str:
    """
    Robust LLM caller with:
      - Explicit handling of 429 rate limits (parse "try again in Xms" and backoff)
      - Exponential backoff for transient errors (5xx)
      - Fallback retry without temperature if the server rejects temperature
      - Optional model fallback to a lighter variant (e.g., gpt-4o-mini) after repeated 429s
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed; cannot call LLM in REAL mode.")

    client = OpenAI(api_key=api_key)

    # Optional: if BASE_MODEL is a heavier 4o, allow fallback to a lighter one on repeated 429s
    model = BASE_MODEL
    fallback_model = "gpt-4o-mini" if ("gpt-4o" in str(BASE_MODEL) and "mini" not in str(BASE_MODEL)) else None

    max_retries = 6
    backoff = 0.75  # seconds; grows on each retry
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"model": model, "messages": messages}
            if TEMPERATURE is not None:
                kwargs["temperature"] = TEMPERATURE

            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Empty completion content")
            return text

        except Exception as e:
            last_err = e
            msg = str(e)

            # Retry without temperature if the API complains about it
            if "param': 'temperature'" in msg or "Unsupported value" in msg:
                try:
                    resp = client.chat.completions.create(model=model, messages=messages)
                    text = (resp.choices[0].message.content or "").strip()
                    if text:
                        return text
                except Exception as e2:
                    print(f"[WARN] retry without temperature failed: {e2}")
                    msg = str(e2)  # continue handling with updated msg

            # Handle explicit rate limiting (429); respect "try again in Xms" if present
            if ("rate_limit_exceeded" in msg) or ("Rate limit" in msg) or (" 429" in msg):
                m = re.search(r"try again in\s+(\d+)ms", msg, re.I)
                sleep_s = backoff
                if m:
                    try:
                        sleep_s = max(sleep_s, float(m.group(1)) / 1000.0)
                    except Exception:
                        pass
                print(f"[WARN] 429 rate limit. Sleeping {sleep_s:.2f}s (attempt {attempt}/{max_retries})")
                time.sleep(sleep_s)
                backoff = min(backoff * 1.8, 8.0)

                # After a couple of rate-limit hits, optionally switch to a lighter model
                if fallback_model and attempt >= 2 and model != fallback_model:
                    print(f"[INFO] Switching model to {fallback_model} due to repeated 429.")
                    model = fallback_model
                continue

            # Transient server-side errors (5xx)
            if any(code in msg for code in (" 502", " 503", " 504")):
                sleep_s = backoff
                print(f"[WARN] Server error {msg[:60]}... Sleeping {sleep_s:.2f}s (attempt {attempt}/{max_retries})")
                time.sleep(sleep_s)
                backoff = min(backoff * 1.6, 6.0)
                continue

            # Other errors: generic retry with backoff
            sleep_s = backoff
            print(f"[WARN] LLM API error: {e}; retrying in {sleep_s:.2f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)
            backoff = min(backoff * 1.5, 5.0)

    raise RuntimeError(f"LLM API failed after {max_retries} retries: {last_err}")

# ---------------- Migration loader ----------------
def migrate_legacy_metrics_array_to_chat(raw: List[Dict], label_name: str) -> List[Dict]:
    def instruction_block_inner(label_name_inner: str) -> str:
        return (
            "You are an expert in smart contract vulnerability detection.\n"
            "A Solidity smart contract will be provided; identify all vulnerabilities present in the code.\n"
            f"Focus specifically on detecting instances of {label_name_inner}.\n"
            "Return ONLY the line numbers.\n"
            "Output format constraint: ONLY digits separated by commas (e.g., 12,27,41). "
            "No words, no ranges, no JSON, no brackets, no explanations.\n"
            "When ready, you may first reply 'I understand the patterns and I'm ready for the next contract' or directly give predictions."
        )
    chat: List[Dict] = [{"role": "user", "content": instruction_block_inner(label_name)}]
    for it in raw:
        cid = int(it.get("contract_id", -1)); att = int(it.get("attempt", 1))
        prec = float(it.get("precision", 0.0)); rec = float(it.get("recall", 0.0))
        f1x = float(it.get("per_contract_score", 0.0))
        fb = str(it.get("feedback", "")) or "No feedback available."
        lines = sorted({int(x) for x in re.findall(r'\b\d+\b', str(it.get("llm", "")))})
        chat.append({"role": "user", "content": f"(legacy-migrated) contract #{cid} attempt {att}\n<original user content unavailable>"})
        chat.append({"role": "system", "content": f"[prediction] {','.join(str(x) for x in lines)}"})
        chat.append({"role": "user", "content": f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x:.2f} ; [feedback_for_next] {fb}"})
    return chat

def load_or_migrate_chat(memory_path: Path, label_name: str) -> List[Dict]:
    def _instr(label: str) -> str:
        return (
            "You are an expert in smart contract vulnerability detection.\n"
            "A Solidity smart contract will be provided; identify all vulnerabilities present in the code.\n"
            f"Focus specifically on detecting instances of {label}.\n"
            "Return ONLY the line numbers.\n"
            "Output format constraint: ONLY digits separated by commas (e.g., 12,27,41). "
            "No words, no ranges, no JSON, no brackets, no explanations.\n"
            "When ready, you may first reply 'I understand the patterns and I'm ready for the next contract' or directly give predictions."
        )
    if not memory_path.exists():
        return [{"role": "user", "content": _instr(label_name)}]
    raw = json.load(open(memory_path, "r", encoding="utf-8"))
    if isinstance(raw, list) and raw and "role" not in raw[0]:
        chat = migrate_legacy_metrics_array_to_chat(raw, label_name)
        json.dump(chat, open(memory_path, "w", encoding="utf-8"), indent=2)
        return chat
    chat = raw if isinstance(raw, list) else []
    if not chat or chat[0].get("role") != "user":
        chat.insert(0, {"role": "user", "content": _instr(label_name)})
    return chat

# ---------------- Helper (TEST): parse prediction csv ----------------
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

# ---------------- Early-stop policy ----------------
def _should_stop(policy: str,
                 prec_b: float, rec_b: float,
                 prec_l: float, rec_l: float,
                 threshold: float,
                 has_line_truth: bool) -> bool:
    """
    Early-stopping decision policy:
      - block:         stop if BLOCK P>=th && R>=th
      - line:          stop if LINE  P>=th && R>=th (only if line-truth exists)
      - any:           stop if (block) OR (line) meets threshold
      - perfect_line:  stop if LINE P==1.0 && R==1.0 (only if line-truth exists)
      - both:          stop if (block) AND (line) meet threshold
    """
    if policy == "block":
        return (prec_b >= threshold and rec_b >= threshold)
    if policy == "line":
        return (has_line_truth and prec_l >= threshold and rec_l >= threshold)
    if policy == "any":
        return ((prec_b >= threshold and rec_b >= threshold) or
                (has_line_truth and prec_l >= threshold and rec_l >= threshold))
    if policy == "perfect_line":
        return (has_line_truth and prec_l == 1.0 and rec_l == 1.0)
    if policy == "both":
        return ((prec_b >= threshold and rec_b >= threshold) and
                (has_line_truth and prec_l >= threshold and rec_l >= threshold))
    return False

# ---------------- Best-attempt selection ----------------
@dataclass
class AttemptResult:
    attempt_idx: int
    pred_lines: List[int]
    # block metrics
    b_prec: float; b_rec: float; b_f1: float; b_hitrate: float
    # line metrics
    l_prec: float; l_rec: float; l_f1: float; l_acc: float
    # messages (only the selected attempt will be persisted)
    user_prompt_logged: str
    system_content: str
    analysis_user: str
    # buffered console report (for later printing)
    console_report: str

def _select_best_attempt(attempts: List[AttemptResult]) -> AttemptResult:
    """
    1) Strict dominance on both Block F1 and Line F1.
    2) Otherwise: max average F1 ((b_f1 + l_f1)/2).
    3) Tie-breakers: b_f1, l_f1, b_rec, l_rec, lower attempt_idx.
    """
    if not attempts:
        raise RuntimeError("No attempts available for selection.")
    # strict dominance
    for a in attempts:
        if all((a.b_f1 > b.b_f1 and a.l_f1 > b.l_f1) or (a is b) for b in attempts):
            return a
    # fallback by keys
    def _key(x: AttemptResult):
        avg = (x.b_f1 + x.l_f1) / 2.0
        return (avg, x.b_f1, x.l_f1, x.b_rec, x.l_rec, -x.attempt_idx)
    return sorted(attempts, key=_key, reverse=True)[0]

# ---------------- NEW: Memory summarization & smart feedback helpers ----------------
def extract_feedback_strings(memory_chat: List[Dict], k: int = 200) -> List[str]:
    """Extract last k '[feedback_for_next]' user notes from memory."""
    out: List[str] = []
    for m in reversed(memory_chat):
        if m.get("role") == "user":
            txt = str(m.get("content", ""))
            if "[feedback_for_next]" in txt:
                fb = txt.split("[feedback_for_next]", 1)[1].strip()
                if fb:
                    out.append(fb)
                    if len(out) >= k:
                        break
    return list(reversed(out))

def summarize_feedback_local(feedbacks: List[str], fb_chars: int = 600) -> str:
    """Local summarization: unique + frequency ordered + char limit."""
    if not feedbacks:
        return ""
    norm = [re.sub(r"\s+", " ", f.strip()) for f in feedbacks if f.strip()]
    cnt = Counter(norm)
    lines = [f"- {t}" for (t, _) in cnt.most_common()]
    text = "=== MEMORY SUMMARY ===\n" + "\n".join(lines)
    return (text[:fb_chars-3] + "...") if len(text) > fb_chars else text

def build_error_profile_for_rule(pred_lines: List[int], truth_block: List[int], code_lines: List[str], k:int=2) -> Dict[str,Any]:
    P, T = set(pred_lines), set(truth_block or [])
    tp = sorted(P & T); fp = sorted(P - T); fn = sorted(T - P)
    def snips(L: List[int]) -> List[str]:
        out = []
        for l in L[:k]:
            s = code_lines[l-1].strip() if 1 <= l <= len(code_lines) else ""
            out.append(f"L{l}:{s[:100]}")
        return out
    return {
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "fp_snips": snips(fp), "fn_snips": snips(fn)
    }

def make_one_line_rule_local(label: str, err: Dict[str,Any], rule_chars:int=180) -> str:
    base = {
      "Re-entrancy": "Select only external value-transfers preceded by state-write; ignore events/comments and arithmetic-only lines.",
      "Timestamp-Dependency": "Flag timestamp usages that gate control/payouts; ignore unused or constant timestamps.",
      "Unchecked-Send": "Keep only sends without require/handling; drop lines that already check return.",
      "Unhandled-Exceptions": "Mark call/delegatecall/staticcall with ignored return; drop try/catch or require-handled cases.",
      "TOD": "Focus read→external call→dependent write; ignore unrelated ordering mentions.",
      "Overflow-Underflow": "Keep arithmetic updates on balances/allowances without checks; ignore safe math or >=0.8 unless 'unchecked'.",
      "tx.origin": "Flag only auth/critical branching using tx.origin; ignore lookalikes and comments.",
    }.get(label, "Focus on exact vulnerability lines; avoid boilerplate/events/comments.")
    if err.get("fp", err.get("fp_count", err.get("fp", 0))) > 0 and err.get("fn", err.get("fn_count", err.get("fn", 0))) == 0:
        hint = base
    elif err.get("fn", err.get("fn_count", err.get("fn", 0))) > 0 and err.get("fp", err.get("fp_count", err.get("fp", 0))) == 0:
        hint = "Broaden minimally: include adjacent control lines or near-calls matching missed pattern."
    else:
        hint = base
    return hint[:rule_chars]

def make_one_line_rule_llm(api_key: str, model: str, label: str, memory_summary: str,
                           err: Dict[str,Any], rule_chars:int=180) -> str:
    if OpenAI is None:
        return make_one_line_rule_local(label, err, rule_chars)
    sys_msg = ("You are a Solidity security coach. Output ONE imperative rule (<=%d chars) "
               "to improve the NEXT attempt; prioritize PRECISION, preserve RECALL.") % rule_chars
    user_msg = (f"[label]={label}\n"
                f"{memory_summary}\n"
                f"[current] tp={err.get('tp',0)} fp={err.get('fp',0)} fn={err.get('fn',0)}\n"
                f"FP={'; '.join(err.get('fp_snips',[]))}\n"
                f"FN={'; '.join(err.get('fn_snips',[]))}\n"
                "Return ONE sentence. No bullets.")
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":sys_msg},
                      {"role":"user","content":user_msg}],
            # max_tokens=64, temperature=0
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"\s+", " ", text)[:rule_chars]
        return text if len(text) >= 8 else make_one_line_rule_local(label, err, rule_chars)
    except Exception as e:
        print(f"[WARN] smart-feedback LLM error: {e}")
        return make_one_line_rule_local(label, err, rule_chars)

# ---------------- REAL mode ----------------
def process_label_real(
    label_key: str,
    folder_name: str,
    contracts_root: Path,
    results_root: Path,
    memory_root: Path,
    api_key: str,
    threshold: float,
    max_attempts: int,
    history_turns: int,
    condense_window: int,
    topk_candidates: int,
    block_dilation: int,
    early_stop: str,
    block_eval: str,
    line_tolerance: int,
    # NEW smart feedback & memory controls
    smart_feedback: str,
    fb_history_k: int,
    fb_max_chars: int,
    fb_rule_chars: int,
    mem_max_msgs: int,
    mem_keep_recent: int,
    distill_every: int,
    contract_counter_state: Dict[str, int],
    limit_contracts: Optional[int] = None  # NEW: maximum number of contracts to process for this label in this run
):
    label_dir = contracts_root / folder_name
    if not label_dir.exists():
        print(f"[SKIP] Missing directory: {label_dir}")
        return

    results_dir = results_root / folder_name
    mem_dir = memory_root / folder_name
    ensure_dir(results_dir); ensure_dir(mem_dir)
    memory_path = mem_dir / "memory_full.json"

    memory_chat = load_or_migrate_chat(memory_path, folder_name)
    print(f"[INFO] Loaded memory chat for {label_key} ({len(memory_chat)} messages).")

    sol_files = sorted(label_dir.glob("buggy_*.sol"))
    if not sol_files:
        print(f"[WARN] No contracts found in {label_dir}")
        return

    # NEW: If a limit is provided, only process the first N contracts of this label.
    if limit_contracts is not None and limit_contracts > 0:
        sol_files = sol_files[:limit_contracts]

    carried_feedback_across_contracts = get_last_user_feedback(memory_chat)
    folder_rows: List[Dict[str, Any]] = []

    # per-label counter for distillation cadence
    label_key_for_counter = f"__count__::{folder_name}"
    if label_key_for_counter not in contract_counter_state:
        contract_counter_state[label_key_for_counter] = 0

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
        total_lines = len(contract_text.splitlines())

        truth_all = read_truth(meta_file)
        truth_block = truth_all["block_lines"]
        truth_point = truth_all["point_lines"]

        attempt = 0
        last_numeric_pred: List[int] = []
        last_llm_text: str = ""
        final_lines: List[int] = []
        final_prec_b = final_rec_b = 0.0
        final_f1x100_b = 0.0

        # Build short memory summary ONCE per contract (from memory_chat)
        recent_feedbacks = extract_feedback_strings(memory_chat, k=fb_history_k)
        memory_summary = summarize_feedback_local(recent_feedbacks, fb_max_chars)

        immediate_feedback: Optional[str] = carried_feedback_across_contracts if carried_feedback_across_contracts else memory_summary

        # collect attempts here; we will select best at the end
        attempt_results: List[AttemptResult] = []
        numeric_attempts: List[AttemptResult] = []

        while True:
            attempt += 1
            messages = build_messages_for_attempt(
                memory_chat=memory_chat,
                contract_text=contract_text,
                history_turns=history_turns,
                carry_feedback=immediate_feedback,
                label_name=folder_name,
                attempt_index=attempt,
                last_pred=last_numeric_pred if last_numeric_pred else None,
                truth_block=truth_block,
                condense_window=condense_window,
                topk_candidates=topk_candidates
            )

            # Buffer header (will decide later)
            header = f"\n[REAL::{folder_name}] {sol_file.name} | Attempt {attempt}"

            llm_text = call_llm_messages(messages, api_key).strip()
            last_llm_text = llm_text

            found_nums = re.findall(r'\b\d+\b', llm_text)
            if found_nums:
                pred_lines = sorted({int(x) for x in found_nums})

                # ---- Block-level evaluation (configurable) ----
                b = compute_block_metrics(
                    pred_lines=pred_lines,
                    truth_block=truth_block,
                    mode=block_eval,                 # 'hit' | 'dilated' | 'overlap'
                    dilation=block_dilation
                )
                prec_b, rec_b, f1_b = b.prec, b.rec, b.f1
                f1x100_b = round(100.0 * f1_b, 2)
                hitrate_b = b.acc_value if b.acc_label == "HitRate" else None

                # ---- Line-level evaluation (with optional tolerance) ----
                if truth_point:
                    prec_l, rec_l, TP_l, FP_l, FN_l = precision_recall_with_tol(pred_lines, truth_point, line_tolerance)
                    f1_l = f1_from_pr(prec_l, rec_l)
                    f1x100_l = round(100.0 * f1_l, 2)
                    acc_l = accuracy_from_counts(TP_l, FP_l, FN_l, total_lines)
                else:
                    prec_l = rec_l = f1_l = acc_l = 0.0
                    TP_l = FP_l = FN_l = 0
                    f1x100_l = 0.0

                last_numeric_pred = pred_lines[:]

                # Base compact feedback (candidate-centric)
                fb_local_compact = build_compact_guidance(
                    label_name=folder_name,
                    contract_text=contract_text,
                    pred_lines=pred_lines,
                    truth_block=truth_block,
                    truth_point=truth_point,
                    prec_block=prec_b,
                    rec_block=rec_b,
                    k=2,
                    max_chars=min(900, fb_max_chars)
                )

                # Error profile for one-line rule
                err_prof = build_error_profile_for_rule(pred_lines, truth_block, contract_text.splitlines(), k=2)

                # One-line rule from memory + current errors
                if smart_feedback == "llm":
                    fb_rule = make_one_line_rule_llm(api_key, BASE_MODEL, folder_name, memory_summary, err_prof, rule_chars=fb_rule_chars)
                elif smart_feedback == "local":
                    fb_rule = make_one_line_rule_local(folder_name, err_prof, rule_chars=fb_rule_chars)
                else:
                    fb_rule = ""

                # Compose final feedback injected to next attempt
                fb_parts = []
                if memory_summary:
                    fb_parts.append(memory_summary)
                if fb_rule:
                    fb_parts.append("=== NEXT RULE ===\n" + fb_rule)
                fb_parts.append(fb_local_compact)
                fb = "\n\n".join(fb_parts)

                system_content = f"[prediction] {','.join(str(x) for x in pred_lines)}"
                analysis_user = (
                    f"[analysis] "
                    f"[BLOCK/{block_eval}] P={prec_b:.4f} R={rec_b:.4f} F1*100={f1x100_b:.2f}"
                    + (f" HitRate*100={100*hitrate_b:.2f}" if hitrate_b is not None else "")
                    + f" ; [LINE±{line_tolerance}] P={prec_l:.4f} R={rec_l:.4f} F1*100={f1x100_l:.2f} Acc*100={100*acc_l:.2f} ; "
                    f"[feedback_for_next] {fb}"
                )
                user_prompt_logged = messages[-1]["content"]

                console_report = (
                    f"{header}\n"
                    f"   SYSTEM → {system_content}\n"
                    f"   EVAL   → [BLOCK/{block_eval}] P={prec_b:.3f}, R={rec_b:.3f}, F1*100={f1x100_b:.2f}"
                    + (f", HitRate*100={100*hitrate_b:.2f}" if hitrate_b is not None else "")
                    + f" | [LINE±{line_tolerance}] P={prec_l:.3f}, R={rec_l:.3f}, F1*100={f1x100_l:.2f}, Acc*100={100*acc_l:.2f}"
                )

                ar = AttemptResult(
                    attempt_idx=attempt,
                    pred_lines=pred_lines,
                    b_prec=prec_b, b_rec=rec_b, b_f1=f1_b, b_hitrate=hitrate_b if hitrate_b is not None else 0.0,
                    l_prec=prec_l, l_rec=rec_l, l_f1=f1_l, l_acc=acc_l,
                    user_prompt_logged=user_prompt_logged,
                    system_content=system_content,
                    analysis_user=analysis_user,
                    console_report=console_report
                )
                attempt_results.append(ar)
                numeric_attempts.append(ar)

                # Early stop decision
                stop_now = _should_stop(
                    policy=early_stop,
                    prec_b=prec_b, rec_b=rec_b,
                    prec_l=prec_l, rec_l=rec_l,
                    threshold=threshold,
                    has_line_truth=bool(truth_point)
                )

                if stop_now or attempt >= max_attempts:
                    carried_feedback_across_contracts = fb_rule or fb_local_compact or memory_summary
                    break
                else:
                    immediate_feedback = fb
                    continue

            else:
                # Warm-up / malformed output attempt (no numeric predictions)
                system_content = llm_text or "I understand the patterns and I'm ready for the next contract."
                analysis_user = "[analysis] WARM-UP_ACK ; [feedback_for_next] Warm-up acknowledged."
                user_prompt_logged = messages[-1]["content"]

                console_report = (
                    f"{header}\n"
                    f"   SYSTEM → {system_content}\n"
                    f"   EVAL   → No numeric predictions."
                )
                ar = AttemptResult(
                    attempt_idx=attempt,
                    pred_lines=[],
                    b_prec=0.0, b_rec=0.0, b_f1=0.0, b_hitrate=0.0,
                    l_prec=0.0, l_rec=0.0, l_f1=0.0, l_acc=0.0,
                    user_prompt_logged=user_prompt_logged,
                    system_content=system_content,
                    analysis_user=analysis_user,
                    console_report=console_report
                )
                attempt_results.append(ar)

                if attempt < max_attempts:
                    immediate_feedback = (
                        "Return only comma-separated integers (e.g., 12,27). "
                        "No words, no ranges, no JSON, no brackets."
                    )
                    continue
                else:
                    break

        # ---- Selection & persistence ----
        if numeric_attempts:
            best = _select_best_attempt(numeric_attempts)
            final_lines = best.pred_lines[:]
            final_prec_b, final_rec_b = best.b_prec, best.b_rec
            final_f1x100_b = round(100.0 * best.b_f1, 2)

            # Print selected attempt to both console and log
            print(best.console_report)
            # Print other attempts to console ONLY
            for ar in attempt_results:
                if ar is best:
                    continue
                try:
                    sys.__stdout__.write(ar.console_report + "\n")
                except Exception:
                    print(ar.console_report)

            # Persist ONLY the best attempt into memory chat & disk (store minimalized prompt to prevent bloat)
            append_attempt_to_chat(memory_chat, best.user_prompt_logged, best.system_content, best.analysis_user, store_minimal_prompt=True)
            json.dump(memory_chat, open(memory_path, "w"), indent=2)

            # Add to folder summary
            folder_rows.append({
                "filename": sol_file.name,
                "BlockDetection.P": best.b_prec,
                "BlockDetection.Recall": best.b_rec,
                "BlockDetection.F1-Score": best.b_f1,
                "BlockDetection.HitRate": best.b_hitrate if block_eval in ("hit", "dilated") else None,
                "LineDetection.P": best.l_prec,
                "LineDetection.Recall": best.l_rec,
                "LineDetection.F1-Score": best.l_f1,
                "LineDetection.Accuracy": best.l_acc,
            })

            out_csv = results_dir / f"{sol_file.stem}_pred.csv"
            pd.DataFrame({"predicted_lines": final_lines}).to_csv(out_csv, index=False)
            print(f"   FINAL → [BLOCK/{block_eval}] P={final_prec_b:.3f}, R={final_rec_b:.3f}, F1*100={final_f1x100_b:.2f} (saved {out_csv.name}) | Selected Attempt = {best.attempt_idx}")

        else:
            # No numeric predictions at all
            for ar in attempt_results:
                try:
                    sys.__stdout__.write(ar.console_report + "\n")
                except Exception:
                    print(ar.console_report)

            folder_rows.append({
                "filename": sol_file.name,
                "BlockDetection.P": 0.0,
                "BlockDetection.Recall": 0.0,
                "BlockDetection.F1-Score": 0.0,
                "BlockDetection.HitRate": 0.0 if block_eval in ("hit", "dilated") else None,
                "LineDetection.P": 0.0,
                "LineDetection.Recall": 0.0,
                "LineDetection.F1-Score": 0.0,
                "LineDetection.Accuracy": 0.0,
            })
            print(f"   FINAL → No numeric predictions to save for {sol_file.stem}. Skipping file.")

        # ---- PRUNE + DISTILL (memory bloat control), done per contract with cadence ----
        contract_counter_state[label_key_for_counter] += 1
        should_distill_now = (contract_counter_state[label_key_for_counter] % max(1, distill_every) == 0)
        if len(memory_chat) > mem_max_msgs or should_distill_now:
            try:
                # Build a fresh summary of all feedbacks
                fb_all = extract_feedback_strings(memory_chat, k=max(fb_history_k, 200))
                mem_sum_text = summarize_feedback_local(fb_all, fb_max_chars)
                distilled_note = {"role": "user", "content": "[feedback_for_next] " + mem_sum_text} if mem_sum_text else None

                # Preserve the first instruction message if exists
                seed = memory_chat[0:1] if memory_chat and memory_chat[0].get("role") == "user" else []
                recent = memory_chat[-mem_keep_recent:] if mem_keep_recent > 0 else []

                new_mem = seed + ([distilled_note] if distilled_note else []) + recent
                memory_chat[:] = new_mem
                json.dump(memory_chat, open(memory_path, "w"), indent=2)
                print(f"[INFO] Memory distilled/pruned for label '{folder_name}'. Kept {len(memory_chat)} messages.")
            except Exception as e:
                print(f"[WARN] Memory distillation failed: {e}")

    try:
        if folder_rows:
            df_folder = pd.DataFrame(folder_rows)
            # Order columns cleanly
            cols = [
                "filename",
                "BlockDetection.P", "BlockDetection.Recall", "BlockDetection.F1-Score", "BlockDetection.HitRate",
                "LineDetection.P", "LineDetection.Recall", "LineDetection.F1-Score", "LineDetection.Accuracy",
            ]
            # keep only existing
            cols = [c for c in cols if c in df_folder.columns]
            df_folder = df_folder[cols]
            folder_csv = results_dir / f"{folder_name}_result.csv"
            df_folder.to_csv(folder_csv, index=False)
            # Macro averages (simple means)
            def _mean_safe(series_name: str) -> float:
                if series_name not in df_folder.columns: return 0.0
                return float(pd.to_numeric(df_folder[series_name], errors="coerce").mean())
            macro_block_p = _mean_safe("BlockDetection.P")
            macro_block_r = _mean_safe("BlockDetection.Recall")
            macro_block_f1 = _mean_safe("BlockDetection.F1-Score")
            macro_line_p  = _mean_safe("LineDetection.P")
            macro_line_r  = _mean_safe("LineDetection.Recall")
            macro_line_f1 = _mean_safe("LineDetection.F1-Score")
            print(f"[INFO] Saved folder summary → {folder_csv}")
            print(f"[SUMMARY::{folder_name}] MACRO Block P/R/F1 = {macro_block_p:.3f}/{macro_block_r:.3f}/{macro_block_f1:.3f} | "
                  f"MACRO Line P/R/F1 = {macro_line_p:.3f}/{macro_line_r:.3f}/{macro_line_f1:.3f}")
    except Exception as e:
        print(f"[WARN] Could not save folder summary: {e}")

# ---------------- TEST mode ----------------
def find_sample_pred_csv(test_pred_root: Path, i: int) -> Optional[Path]:
    for name in (f"buggy_{i}_pred.csv", f"BugLog_{i}_pred.csv"):
        cand = test_pred_root / name
        if cand.exists(): return cand
    return None

def process_test_mode(
    test_root: Path,
    test_pred_root: Path,
    results_root: Path,
    memory_root: Path,
    history_turns: int,
    threshold: float
):
    label_display = "Overflow-Underflow_TEST"
    results_dir = results_root / label_display
    mem_dir = memory_root / label_display
    ensure_dir(results_dir); ensure_dir(mem_dir)
    memory_path = mem_dir / "memory_full.json"

    memory_chat = load_or_migrate_chat(memory_path, "Overflow-Underflow")
    print(f"[INFO][TEST] Loaded memory chat ({len(memory_chat)} messages).")

    sol_files = [test_root / f"buggy_{i}.sol" for i in range(1, 6)]
    meta_files = [test_root / f"BugLog_{i}.csv" for i in range(1, 6)]
    for p in sol_files + meta_files:
        if not p.exists(): print(f"[ERROR][TEST] Missing file: {p}")

    if sol_files[0].exists():
        ctext1 = read_text(sol_files[0])
        injected_fb = get_last_user_feedback(memory_chat)
        user_prompt_1 = build_user_contract_prompt(
            contract_text=ctext1,
            injected_feedback=injected_fb,
            label_name="Overflow-Underflow",
            attempt_index=1,
            last_pred=None,
            truth_block=None,
            condense_window=5,
            topk_candidates=40
        )

        pred2_csv = find_sample_pred_csv(test_pred_root, 2)
        if pred2_csv:
            pred2_lines = parse_pred_csv_lines(pred2_csv)
            system_content = f"[prediction] {','.join(str(x) for x in pred2_lines)}"
            truth1 = read_vuln_lines_from_csv(meta_files[0]) if meta_files[0].exists() else []
            prec, rec, *_ = precision_recall(pred2_lines, truth1)
            f1x100 = round(100.0 * f1_from_pr(prec, rec), 2)
            fb = "Use candidate-centric hints."
            analysis_user = f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x100:.2f} ; [feedback_for_next] {fb}"
        else:
            system_content = "I understand the patterns and I'm ready for the next contract."
            analysis_user = "[analysis] WARM-UP_ACK ; [feedback_for_next] Warm-up acknowledged."

        append_attempt_to_chat(memory_chat, user_prompt_1, system_content, analysis_user, store_minimal_prompt=True)
        json.dump(memory_chat, open(memory_path, "w"), indent=2)
        print("[TEST] Priming complete.")
    else:
        print("[ERROR][TEST] Missing buggy_1.sol; aborting test.")
        return

    for i in range(2, 6):
        sol_path = sol_files[i-1]; meta_path = meta_files[i-1]
        if not sol_path.exists() or not meta_path.exists():
            print(f"[WARN][TEST] Skipping buggy_{i}: missing files.")
            memory_chat.append({"role": "system", "content": "[prediction] "})
            memory_chat.append({"role": "user", "content": "[analysis] Missing files ; [feedback_for_next] Missing files."})
            json.dump(memory_chat, open(memory_path, "w"), indent=2)
            continue

        pred_csv = find_sample_pred_csv(test_pred_root, i)
        if not pred_csv:
            print(f"[WARN][TEST] Missing sample prediction CSV for contract {i}.")
            memory_chat.append({"role": "user", "content": f"Analyze buggy_{i}..."})
            memory_chat.append({"role": "system", "content": "[prediction] "})
            memory_chat.append({"role": "user", "content": "[analysis] P=0.0000 R=0.0000 F1*100=0.00 ; [feedback_for_next] Missing sample prediction."})
            json.dump(memory_chat, open(memory_path, "w"), indent=2)
            continue

        pred_lines = parse_pred_csv_lines(pred_csv)
        truth_lines = read_vuln_lines_from_csv(meta_path)
        prec, rec, *_ = precision_recall(pred_lines, truth_lines)
        f1x100 = round(100.0 * f1_from_pr(prec, rec), 2)
        fb = "Use candidate-centric hints."
        system_content = f"[prediction] {','.join(str(x) for x in pred_lines)}"
        analysis_user = f"[analysis] P={prec:.4f} R={rec:.4f} F1*100={f1x100:.2f} ; [feedback_for_next] {fb}"
        memory_chat.append({"role": "user", "content": f"Analyze buggy_{i}..."})
        memory_chat.append({"role": "system", "content": system_content})
        memory_chat.append({"role": "user", "content": analysis_user})
        json.dump(memory_chat, open(memory_path, "w"), indent=2)

        out_csv = results_dir / f"{sol_path.stem}_pred.csv"
        pd.DataFrame({"predicted_lines": pred_lines}).to_csv(out_csv, index=False)
        print(f"[TEST] {sol_path.name} → P={prec:.3f}, R={rec:.3f}, F1*100={f1x100:.2f} (saved {out_csv.name})")

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "test"], default="real",
                    help="Run mode: real (7 labels) or test (5-contract demo).")
    ap.add_argument("--contracts_root", help="Root of labeled contract folders (required in real mode).")
    ap.add_argument("--results_root", required=True, help="Where to save final prediction CSVs.")
    ap.add_argument("--memory_root", required=True, help="Where to save full chat memory JSONs.")
    ap.add_argument("--api_key", required=True, help="LLM API key.")
    ap.add_argument("--threshold", type=float, default=0.7, help="Precision/Recall threshold.")
    ap.add_argument("--max_attempts", type=int, default=3, help="Maximum attempts per contract.")  # <-- 3 attempts
    ap.add_argument("--history_turns", type=int, default=HISTORY_TURNS_DEFAULT, help="How many recent SYSTEM msgs to include.")
    # Compact params
    ap.add_argument("--condense_window", type=int, default=5, help="Snippet radius for compact context.")
    ap.add_argument("--topk_candidates", type=int, default=40, help="Max number of candidate lines to include.")
    # Block eval params
    ap.add_argument("--block_dilation", type=int, default=1, help="Dilation window for BLOCK metrics (used in 'dilated' mode).")
    ap.add_argument("--block_eval", choices=["hit", "dilated", "overlap"], default="dilated",
                    help="Block eval mode: 'hit' (exact), 'dilated' (±w tolerance on predictions), 'overlap' (set-overlap).")
    # Line eval tolerance
    ap.add_argument("--line_tolerance", type=int, default=0, help="±w tolerance for line-level matching (0 = exact).")
    # Early stop
    ap.add_argument(
        "--early_stop",
        choices=["block", "line", "any", "perfect_line", "both"],
        default="block",
        help="Stopping policy: block (default), line, any, perfect_line, both."
    )
    # NEW: smart feedback & memory controls
    ap.add_argument("--smart_feedback", choices=["off","local","llm"], default="llm",
                    help="How to build feedback for next attempts.")
    ap.add_argument("--fb_history_k", type=int, default=12,
                    help="How many recent [feedback_for_next] entries to summarize from memory.")
    ap.add_argument("--fb_max_chars", type=int, default=600,
                    help="Max characters for memory summary injected into next attempt.")
    ap.add_argument("--fb_rule_chars", type=int, default=180,
                    help="Max characters for one-line rule injected into next attempt.")
    ap.add_argument("--mem_max_msgs", type=int, default=120,
                    help="If memory_chat grows beyond this, prune/distill.")
    ap.add_argument("--mem_keep_recent", type=int, default=24,
                    help="After pruning/distill, keep last N messages (plus seed + distilled summary).")
    ap.add_argument("--distill_every", type=int, default=10,
                    help="Run prune/distill every N processed contracts per label.")

    # TEST paths
    ap.add_argument("--test_root", default="test/Overflow-Underflow", help="Test folder with 5 contracts + metadata.")
    ap.add_argument("--test_pred_root", default="test/Overflow-Underflow/sample_preds", help="Folder with sample predictions.")

    # NEW: selection & limit controls
    ap.add_argument("--label_index", type=int, default=None,
                    help="Pick a vulnerability class by number (1..7). If omitted in real mode, you will be prompted.")
    ap.add_argument("--limit_contracts", type=int, default=50,
                    help="Max number of contracts to process for the selected label (default=50).")
    ap.add_argument("--all_labels", action="store_true",
                    help="Process ALL labels (legacy behavior). If set, ignores --label_index prompt behavior.")

    args = ap.parse_args()

    results_root = Path(args.results_root)
    memory_root = Path(args.memory_root)
    ensure_dir(results_root); ensure_dir(memory_root)

    # Logging setup (only in real mode)
    log_fp = None
    if args.mode == "real":
        logs_dir = results_root / "log"
        ensure_dir(logs_dir)
        log_path = logs_dir / "run.log"
        log_fp = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
        start_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fp.write("\n" + "="*80 + "\n")
        log_fp.write(f"[RUN START] {start_stamp}\n")
        log_fp.write(f"mode=real results_root={results_root} memory_root={memory_root}\n")
        if args.contracts_root:
            log_fp.write(f"contracts_root={args.contracts_root}\n")
        log_fp.write(
            "params: "
            f"threshold={args.threshold} max_attempts={args.max_attempts} "
            f"history_turns={args.history_turns} condense_window={args.condense_window} "
            f"topk_candidates={args.topk_candidates} block_dilation={args.block_dilation} "
            f"block_eval={args.block_eval} line_tolerance={args.line_tolerance} "
            f"early_stop={args.early_stop} smart_feedback={args.smart_feedback} "
            f"fb_history_k={args.fb_history_k} fb_max_chars={args.fb_max_chars} fb_rule_chars={args.fb_rule_chars} "
            f"mem_max_msgs={args.mem_max_msgs} mem_keep_recent={args.mem_keep_recent} distill_every={args.distill_every}\n"
        )
        log_fp.write("-"*80 + "\n")

        # tee stdout/stderr
        sys.stdout = _Tee(sys.stdout, log_fp)
        sys.stderr = _Tee(sys.stderr, log_fp)

        @atexit.register
        def _close_log():
            try:
                if log_fp and not log_fp.closed:
                    log_fp.write(f"[RUN END]   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    log_fp.flush()
                    log_fp.close()
            except Exception:
                pass

    if args.mode == "test":
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

    ensure_label_dirs(results_root); ensure_label_dirs(memory_root)
    if not args.contracts_root:
        raise SystemExit("--contracts_root is required for real mode.")
    contracts_root = Path(args.contracts_root)

    # shared counter state for distill cadence
    contract_counter_state: Dict[str, int] = {}

    # --- NEW: Interactive/specified label selection (default), while preserving legacy "all labels" path. ---
    labels_list = list(LABEL_FOLDERS.items())  # [(display_name, folder_name_on_disk), ...]

    if args.all_labels:
        # Legacy behavior: iterate all labels (kept to avoid removing existing code paths).
        for label_key, folder_name in LABEL_FOLDERS.items():
            print(f"\n=== REAL MODE: Processing Label '{label_key}' → folder '{folder_name}' ===")
            process_label_real(
                label_key=label_key,
                folder_name=folder_name,
                contracts_root=contracts_root,
                results_root=results_root,
                memory_root=memory_root,
                api_key=args.api_key,
                threshold=args.threshold,
                max_attempts=args.max_attempts,
                history_turns=args.history_turns,
                condense_window=args.condense_window,
                topk_candidates=args.topk_candidates,
                block_dilation=args.block_dilation,
                early_stop=args.early_stop,
                block_eval=args.block_eval,
                line_tolerance=args.line_tolerance,
                # NEW
                smart_feedback=args.smart_feedback,
                fb_history_k=args.fb_history_k,
                fb_max_chars=args.fb_max_chars,
                fb_rule_chars=args.fb_rule_chars,
                mem_max_msgs=args.mem_max_msgs,
                mem_keep_recent=args.mem_keep_recent,
                distill_every=args.distill_every,
                contract_counter_state=contract_counter_state,
                limit_contracts=args.limit_contracts  # still honored in all-labels mode
            )
        print("\nREAL mode completed.")
        return

    # Default behavior: process ONLY one selected label.
    if args.label_index is None:
        print("\nSelect a vulnerability class to process (1..7):")
        for i, (disp, fold) in enumerate(labels_list, start=1):
            print(f"  {i}) {disp}  ->  {fold}")
        while True:
            try:
                sel = int(input("Enter 1..7: ").strip())
                if 1 <= sel <= len(labels_list):
                    label_key, folder_name = labels_list[sel - 1]
                    break
                else:
                    print("Please enter a number between 1 and 7.")
            except Exception:
                print("Invalid input. Try again.")
    else:
        if 1 <= args.label_index <= len(labels_list):
            label_key, folder_name = labels_list[args.label_index - 1]
        else:
            raise SystemExit("--label_index must be between 1 and 7.")

    print(f"\n=== REAL MODE: Processing ONLY Label '{label_key}' → folder '{folder_name}' ===")
    process_label_real(
        label_key=label_key,
        folder_name=folder_name,
        contracts_root=contracts_root,
        results_root=results_root,
        memory_root=memory_root,
        api_key=args.api_key,
        threshold=args.threshold,
        max_attempts=args.max_attempts,
        history_turns=args.history_turns,
        condense_window=args.condense_window,
        topk_candidates=args.topk_candidates,
        block_dilation=args.block_dilation,
        early_stop=args.early_stop,
        block_eval=args.block_eval,
        line_tolerance=args.line_tolerance,
        # NEW
        smart_feedback=args.smart_feedback,
        fb_history_k=args.fb_history_k,
        fb_max_chars=args.fb_max_chars,
        fb_rule_chars=args.fb_rule_chars,
        mem_max_msgs=args.mem_max_msgs,
        mem_keep_recent=args.mem_keep_recent,
        distill_every=args.distill_every,
        contract_counter_state=contract_counter_state,
        limit_contracts=args.limit_contracts  # NEW: cap to 50 by default
    )
    print("\nREAL mode completed for the selected label.")

if __name__ == "__main__":
    main()
