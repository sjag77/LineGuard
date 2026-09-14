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
import subprocess
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

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

# ---------------- Defaults ----------------
BASE_MODEL = "gpt-4o"  # overridden by --model at runtime; Models: GPT-4o/GPT-5/claude-opus-5/gemini-2.0-flash/...
PROVIDER = "openai"    # overridden by --provider at runtime: 'openai' | 'anthropic' | 'gemini' | 'groq' | 'openrouter'

DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o",
    "anthropic": "claude-opus-5",
    "claude_cli": "claude-sonnet-5",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
}

# 'gemini', 'groq', 'openrouter' are OpenAI-wire-compatible: same client, different base_url.
PROVIDER_BASE_URL = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Rough $/1M tokens, for cost reporting only (approximate, update as pricing changes).
_COST_PER_M_TOKENS = {
    "gpt-4o":        {"in": 2.50,  "out": 10.00},
    "gpt-4o-mini":    {"in": 0.15,  "out": 0.60},
    "gpt-5":          {"in": 5.00,  "out": 15.00},
    "claude-opus-5":  {"in": 5.00,  "out": 25.00},
    "claude-sonnet-5": {"in": 2.00, "out": 10.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "gemini-2.0-flash": {"in": 0.0, "out": 0.0},  # free tier
    "llama-3.3-70b-versatile": {"in": 0.0, "out": 0.0},  # free tier
    "meta-llama/llama-3.3-70b-instruct:free": {"in": 0.0, "out": 0.0},  # OpenRouter free tier
}

TEMPERATURE = None
HISTORY_TURNS_DEFAULT = 4

# Per-request wall-clock cap. Without this the SDK default (600s) x the retry loop
# lets a single stalled call block a run for ~an hour.
REQUEST_TIMEOUT_S = 120.0
# Predictions are just comma-separated line numbers; the cap also stops reasoning
# models from emitting unbounded thinking tokens.
MAX_OUTPUT_TOKENS = 1024

# ---------------- Claude CLI support ----------------
CLI_EMPTY_DIR = str(Path(__file__).resolve().parent / ".cli_empty")
os.makedirs(CLI_EMPTY_DIR, exist_ok=True)


class UsageLimitReached(RuntimeError):
    """Raised when the Claude subscription usage limit is hit; the run stops cleanly and can be resumed."""


def _looks_like_usage_limit(data: Dict[str, Any], blob: str) -> bool:
    status = data.get("api_error_status")
    if status in (429, "429"):
        return True
    return any(s in blob for s in ("usage limit", "limit reached", "rate limit", "5-hour limit", "limit will reset"))


# ---------------- Usage / cost tracking (Reviewer#2 Concern #10) ----------------
_USAGE_LOG: List[Dict[str, Any]] = []

def _log_usage(label: str, contract: str, attempt: int, call_type: str,
               provider: str, model: str, prompt_tokens: int, completion_tokens: int,
               elapsed_s: float, is_fallback: bool = False) -> None:
    rates = _COST_PER_M_TOKENS.get(model, {"in": 0.0, "out": 0.0})
    cost = (prompt_tokens / 1e6) * rates["in"] + (completion_tokens / 1e6) * rates["out"]
    _USAGE_LOG.append({
        "label": label, "contract": contract, "attempt": attempt, "call_type": call_type,
        "provider": provider, "model": model,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "elapsed_s": round(elapsed_s, 4), "is_fallback": is_fallback,
        "est_cost_usd": round(cost, 6),
    })

def dump_usage_log(csv_path: Path) -> Optional[Dict[str, Any]]:
    """Writes the full call-level usage log and returns an aggregate summary dict."""
    if not _USAGE_LOG:
        return None
    df = pd.DataFrame(_USAGE_LOG)
    df.to_csv(csv_path, index=False)
    summary = {
        "num_calls": len(df),
        "num_fallback_calls": int(df["is_fallback"].sum()),
        "total_prompt_tokens": int(df["prompt_tokens"].sum()),
        "total_completion_tokens": int(df["completion_tokens"].sum()),
        "total_tokens": int(df["total_tokens"].sum()),
        "total_elapsed_s": float(df["elapsed_s"].sum()),
        "avg_elapsed_s_per_call": float(df["elapsed_s"].mean()),
        "est_total_cost_usd": float(df["est_cost_usd"].sum()),
        "num_contracts": df["contract"].nunique(),
    }
    if summary["num_contracts"] > 0:
        summary["avg_cost_usd_per_contract"] = summary["est_total_cost_usd"] / summary["num_contracts"]
        summary["avg_calls_per_contract"] = summary["num_calls"] / summary["num_contracts"]
    return summary

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

def build_self_consistency_guidance(
    label_name: str,
    contract_text: str,
    pred_lines: List[int],
    ranked_candidates: List[int],
    prev_pred_lines: Optional[List[int]],
    k: int = 2,
    max_chars: int = 900
) -> str:
    """
    Non-oracle counterpart to build_compact_guidance (Reviewer#2 Concern #1).
    Builds feedback WITHOUT any ground-truth access: it only uses (a) the fixed
    per-label rules, (b) ranked candidate lines the model did NOT select (self
    only, no truth), and (c) agreement/disagreement with the model's own previous
    attempt, as a self-consistency signal. This is what would realistically be
    available at real audit time.
    """
    code_lines = contract_text.splitlines()
    pred_set = set(pred_lines)
    unselected = [c for c in ranked_candidates if c not in pred_set]
    unselected_sel = _pick_k(unselected, k)
    unselected_snips = [f"L{ln}: {_snip_line(code_lines, ln)}" for ln in unselected_sel]

    rules = _COMPACT_RULES.get(label_name, {
        "MUST": ["Focus on lines directly implementing the labeled vulnerability."],
        "INCLUDE": ["Prefer lines near external calls or critical state changes."],
        "NEVER": ["Exclude comments, events, and boilerplate."],
    })

    parts: List[str] = ["=== COMPACT HINTS (non-oracle: no ground truth used) ==="]
    parts.append(f"Label: {label_name}")
    parts.append("MUST: " + " ".join(f"- {r}" for r in rules.get("MUST", [])))
    if rules.get("INCLUDE"): parts.append("INCLUDE: " + " ".join(f"- {r}" for r in rules.get("INCLUDE", [])))
    if rules.get("NEVER"):   parts.append("NEVER: " + " ".join(f"- {r}" for r in rules.get("NEVER", [])))
    if unselected_snips:
        parts.append("HighRankedButNotSelected (double-check these): " + " | ".join(unselected_snips))
    if prev_pred_lines is not None:
        agree = sorted(pred_set & set(prev_pred_lines))
        disagree_new = sorted(pred_set - set(prev_pred_lines))
        disagree_dropped = sorted(set(prev_pred_lines) - pred_set)
        parts.append(f"SelfConsistency: agree_with_prev={len(agree)} newly_added={len(disagree_new)} dropped={len(disagree_dropped)}")
        if disagree_new or disagree_dropped:
            parts.append("Re-examine lines that changed between attempts before finalizing.")
    parts.append("FORMAT: Return only comma-separated integers (e.g., 12,27). No words, no ranges, no JSON.")
    text = "\n".join(parts)
    return text[:max_chars-3] + "..." if len(text) > max_chars else text

def _jaccard(a: List[int], b: List[int]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

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
    topk_candidates: int,
    use_pruning: bool = True,   # Reviewer#2 Concern #6 / Reviewer#3 Concern #3 (ablation)
    oracle: bool = True         # Reviewer#2 Concern #1 (non-oracle mode)
) -> str:
    lines = contract_text.splitlines()
    parts: List[str] = []

    # Guidance / feedback (compact hints)
    if injected_feedback:
        parts.append("=== PREVIOUS FEEDBACK (for this attempt) ===\n" + injected_feedback.strip())

    if use_pruning:
        # Rank Top-K candidates and build compact context (semantic pruning component)
        rank = rank_candidates(contract_text, label_name, topk_candidates)
        cand_list = ",".join(str(x) for x in rank) if rank else ""

        if rank:
            code_cand, _ = slice_around(lines, rank, radius=condense_window)
            parts.append("=== CANDIDATE SNIPPETS (Top-K) ===\n" + code_cand)

        # MISSED-TRUTH is an oracle-derived hint: only usable when oracle=True
        if oracle and attempt_index >= 2 and last_pred and truth_block:
            fn = sorted(set(truth_block) - set(last_pred))
            if fn:
                code_fn, _ = slice_around(lines, fn[:min(30, len(fn))], radius=max(2, condense_window//2))
                parts.append("=== MISSED-TRUTH SNIPPETS (focus) ===\n" + code_fn)

        # Explicit gating
        parts.append("You MUST choose line numbers ONLY from CandidateLines below.")
        parts.append("FORMAT: Return only comma-separated integers (e.g., 12,27). No words, no ranges, no JSON.")
        parts.append("CandidateLines: " + cand_list)
        parts.append("Analyze the snippets and output only the matching line numbers.")
    else:
        # Ablation: no semantic pruning — send the full numbered contract instead of Top-K snippets.
        numbered = "\n".join(f"{i:>4}: {l}" for i, l in enumerate(lines, start=1))
        parts.append("=== FULL CONTRACT (no pruning) ===\n" + numbered)
        parts.append("FORMAT: Return only comma-separated integers (e.g., 12,27). No words, no ranges, no JSON.")
        parts.append("Analyze the full contract above and output only the matching line numbers.")

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
    topk_candidates: int,
    use_pruning: bool = True,
    oracle: bool = True
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
        topk_candidates=topk_candidates,
        use_pruning=use_pruning,
        oracle=oracle
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
def _messages_to_anthropic(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Converts this project's OpenAI-style message list (roles: 'user', 'system' —
    where mid-conversation 'system' entries are actually prior model outputs stashed
    for memory purposes) into Anthropic's format: a single system string plus a
    strictly alternating user/assistant message list.
    """
    sys_parts: List[str] = []
    conv: List[Dict] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "system" and i == 0:
            sys_parts.append(content)
            continue
        mapped_role = "assistant" if role == "system" else "user"
        if conv and conv[-1]["role"] == mapped_role:
            conv[-1]["content"] += "\n\n" + content
        else:
            conv.append({"role": mapped_role, "content": content})
    if not conv or conv[0]["role"] != "user":
        conv.insert(0, {"role": "user", "content": "(context continues)"})
    return ("\n\n".join(sys_parts), conv)

def call_llm_messages(
    messages: List[Dict],
    api_key: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    usage_ctx: Optional[Dict[str, Any]] = None,  # {"label","contract","attempt","call_type"}
) -> str:
    """
    Robust LLM caller with:
      - Explicit handling of 429 rate limits (parse "try again in Xms" and backoff)
      - Exponential backoff for transient errors (5xx)
      - Fallback retry without temperature if the server rejects temperature
      - Optional model fallback to a lighter variant (e.g., gpt-4o-mini) after repeated 429s
      - Provider switch between OpenAI and Anthropic (Reviewer#2 Concern #3: model/decoding
        versioning is now explicit and logged; see usage_ctx/_log_usage)
    """
    provider = (provider or PROVIDER or "openai").lower()
    model = model or BASE_MODEL
    ctx = usage_ctx or {}
    max_retries = 6
    backoff = 0.75  # seconds; grows on each retry
    last_err = None

    if provider == "claude_cli":
        # LOCAL FUNCTIONAL TESTING ONLY. Shells out to the Claude Code CLI in print mode.
        # Not suitable for reported results: the CLI applies its own system prompt and
        # harness, the model snapshot and decoding parameters are not controllable or
        # observable, and no token usage is returned.
        # Fixed, documented system prompt replaces the CLI default; all built-in tools and MCP
        # servers are disabled and the call runs from an empty directory, so no project
        # instructions or tool schemas enter the context.
        if ctx.get("call_type", "prediction") != "prediction" and messages and messages[0].get("role") == "system":
            # Auxiliary calls (e.g. the feedback-rule coach) carry their own system message.
            system_prompt = messages[0]["content"]
            history = []
        else:
            system_prompt = ctx.get("system_prompt") or instruction_block(ctx.get("label", "the specified vulnerability"))
            history = [m["content"] for m in messages[:-1] if m.get("role") == "system"]
        prompt_parts = [f"Previous prediction: {h}" for h in history] + [messages[-1]["content"]]
        prompt = "\n\n".join(prompt_parts)
        cmd = ["claude", "-p", "--output-format", "json", "--no-session-persistence",
               "--strict-mcp-config", "--tools", "", "--system-prompt", system_prompt]
        if model:
            cmd += ["--model", model]
        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                      timeout=REQUEST_TIMEOUT_S, cwd=CLI_EMPTY_DIR)
                elapsed = time.time() - t0
                raw = (proc.stdout or "").strip()
                try:
                    data = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    data = {}
                blob = f"{raw} {proc.stderr}".lower()
                if _looks_like_usage_limit(data, blob):
                    raise UsageLimitReached((data.get("result") or proc.stderr or raw).strip()[:300])
                if proc.returncode != 0 or data.get("is_error"):
                    raise RuntimeError(f"claude CLI error (exit {proc.returncode}): {(data.get('result') or proc.stderr).strip()[:200]}")
                text = (data.get("result") or "").strip()
                if not text:
                    raise RuntimeError("Empty CLI result")
                usage = data.get("usage") or {}
                resolved = next(iter(data.get("modelUsage") or {}), model or "cli-default")
                _log_usage(
                    label=ctx.get("label", "unknown"), contract=ctx.get("contract", "unknown"),
                    attempt=ctx.get("attempt", 0), call_type=ctx.get("call_type", "prediction"),
                    provider=provider, model=resolved,
                    prompt_tokens=int(usage.get("input_tokens", 0)) + int(usage.get("cache_creation_input_tokens", 0))
                                  + int(usage.get("cache_read_input_tokens", 0)),
                    completion_tokens=int(usage.get("output_tokens", 0)),
                    elapsed_s=elapsed, is_fallback=False,
                )
                if _USAGE_LOG and data.get("total_cost_usd") is not None:
                    _USAGE_LOG[-1]["est_cost_usd"] = round(float(data["total_cost_usd"]), 6)
                return text
            except UsageLimitReached:
                raise
            except subprocess.TimeoutExpired as e:
                last_err = e
                print(f"[WARN] claude CLI timed out after {REQUEST_TIMEOUT_S}s (attempt {attempt}/{max_retries})")
            except Exception as e:
                last_err = e
                print(f"[WARN] claude CLI error: {e}; retrying in {backoff:.2f}s (attempt {attempt}/{max_retries})")
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 5.0)
        raise RuntimeError(f"claude CLI failed after {max_retries} retries: {last_err}")

    if provider == "anthropic":
        if anthropic is None:
            raise RuntimeError("anthropic package not installed; cannot call LLM with provider=anthropic.")
        # api_key=None lets the SDK resolve credentials itself: ANTHROPIC_API_KEY,
        # ANTHROPIC_AUTH_TOKEN, or an 'ant auth login' OAuth profile on disk.
        client = (anthropic.Anthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S) if api_key
                  else anthropic.Anthropic(timeout=REQUEST_TIMEOUT_S))
        system_str, conv = _messages_to_anthropic(messages)
        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                kwargs = {"model": model, "max_tokens": MAX_OUTPUT_TOKENS, "messages": conv}
                if system_str:
                    kwargs["system"] = system_str
                resp = client.messages.create(**kwargs)
                elapsed = time.time() - t0
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
                if not text:
                    raise RuntimeError("Empty completion content")
                usage = getattr(resp, "usage", None)
                _log_usage(
                    label=ctx.get("label", "unknown"), contract=ctx.get("contract", "unknown"),
                    attempt=ctx.get("attempt", 0), call_type=ctx.get("call_type", "prediction"),
                    provider=provider, model=model,
                    prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                    elapsed_s=elapsed, is_fallback=False,
                )
                return text
            except Exception as e:
                last_err = e
                msg = str(e)
                if ("429" in msg) or ("rate_limit" in msg.lower()) or ("overloaded" in msg.lower()):
                    sleep_s = backoff
                    print(f"[WARN] Anthropic rate limit/overload. Sleeping {sleep_s:.2f}s (attempt {attempt}/{max_retries})")
                    time.sleep(sleep_s)
                    backoff = min(backoff * 1.8, 8.0)
                    continue
                sleep_s = backoff
                print(f"[WARN] Anthropic API error: {e}; retrying in {sleep_s:.2f}s (attempt {attempt}/{max_retries})")
                time.sleep(sleep_s)
                backoff = min(backoff * 1.5, 5.0)
        raise RuntimeError(f"Anthropic API failed after {max_retries} retries: {last_err}")

    # ---- provider in {'openai','gemini','groq'} — all OpenAI-wire-compatible ----
    if OpenAI is None:
        raise RuntimeError("openai package not installed; cannot call LLM in REAL mode.")

    base_url = PROVIDER_BASE_URL.get(provider)  # None for 'openai' -> SDK default
    client = (OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_S) if base_url
              else OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT_S))

    # Optional: if the model is a heavier 4o, allow fallback to a lighter one on repeated 429s
    fallback_model = "gpt-4o-mini" if ("gpt-4o" in str(model) and "mini" not in str(model)) else None
    used_fallback = False

    # GPT-5 / o-series reject 'max_tokens' and require 'max_completion_tokens'; switched on demand.
    token_param = "max_tokens"

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {"model": model, "messages": messages, token_param: MAX_OUTPUT_TOKENS}
            if TEMPERATURE is not None:
                kwargs["temperature"] = TEMPERATURE
            if seed is not None and provider == "openai":  # 'seed' isn't a Gemini/Groq-compat param
                kwargs["seed"] = seed

            t0 = time.time()
            resp = client.chat.completions.create(**kwargs)
            elapsed = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("Empty completion content")
            usage = getattr(resp, "usage", None)
            _log_usage(
                label=ctx.get("label", "unknown"), contract=ctx.get("contract", "unknown"),
                attempt=ctx.get("attempt", 0), call_type=ctx.get("call_type", "prediction"),
                provider=provider, model=model,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                elapsed_s=elapsed, is_fallback=used_fallback,
            )
            return text

        except Exception as e:
            last_err = e
            msg = str(e)

            # GPT-5/o-series: swap max_tokens -> max_completion_tokens and retry immediately.
            if token_param == "max_tokens" and "max_completion_tokens" in msg:
                print("[INFO] Switching to 'max_completion_tokens' for this model.")
                token_param = "max_completion_tokens"
                continue

            # Retry without temperature if the API complains about it
            if "param': 'temperature'" in msg or "Unsupported value" in msg:
                try:
                    t0 = time.time()
                    resp = client.chat.completions.create(
                        model=model, messages=messages, **{token_param: MAX_OUTPUT_TOKENS})
                    elapsed = time.time() - t0
                    text = (resp.choices[0].message.content or "").strip()
                    if text:
                        usage = getattr(resp, "usage", None)
                        _log_usage(
                            label=ctx.get("label", "unknown"), contract=ctx.get("contract", "unknown"),
                            attempt=ctx.get("attempt", 0), call_type=ctx.get("call_type", "prediction"),
                            provider=provider, model=model,
                            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                            elapsed_s=elapsed, is_fallback=used_fallback,
                        )
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
                    used_fallback = True
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

def _should_stop_non_oracle(cur_pred: List[int], prev_pred: Optional[List[int]],
                             convergence_threshold: float = 0.9) -> bool:
    """
    Truth-free stopping rule (Reviewer#2 Concern #1): stop once the model's
    prediction set has converged across consecutive attempts (self-consistency),
    since no metric-derived label is available at real audit time.
    """
    if prev_pred is None:
        return False
    return _jaccard(cur_pred, prev_pred) >= convergence_threshold

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

def _select_best_attempt_non_oracle(attempts: List[AttemptResult]) -> AttemptResult:
    """
    Truth-free counterpart to _select_best_attempt (Reviewer#2 Concern #1).
    Selects the attempt with the highest self-consistency (mean Jaccard
    agreement with every other attempt's prediction set); ties broken toward
    the later attempt (more feedback rounds seen), since no ground-truth
    F1 is available at real audit time.
    """
    if not attempts:
        raise RuntimeError("No attempts available for selection.")
    if len(attempts) == 1:
        return attempts[0]

    def _agreement(a: AttemptResult) -> float:
        others = [b for b in attempts if b is not a]
        return sum(_jaccard(a.pred_lines, b.pred_lines) for b in others) / len(others)

    scored = [(_agreement(a), a.attempt_idx, a) for a in attempts]
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]

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
                           err: Dict[str,Any], rule_chars:int=180,
                           provider: Optional[str] = None, seed: Optional[int] = None,
                           usage_ctx: Optional[Dict[str, Any]] = None) -> str:
    sys_msg = ("You are a Solidity security coach. Output ONE imperative rule (<=%d chars) "
               "to improve the NEXT attempt; prioritize PRECISION, preserve RECALL.") % rule_chars
    user_msg = (f"[label]={label}\n"
                f"{memory_summary}\n"
                f"[current] tp={err.get('tp',0)} fp={err.get('fp',0)} fn={err.get('fn',0)}\n"
                f"FP={'; '.join(err.get('fp_snips',[]))}\n"
                f"FN={'; '.join(err.get('fn_snips',[]))}\n"
                "Return ONE sentence. No bullets.")
    try:
        text = call_llm_messages(
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
            api_key=api_key, provider=provider, model=model, seed=seed,
            usage_ctx={**(usage_ctx or {}), "call_type": "feedback_rule"},
        )
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
    limit_contracts: Optional[int] = None,  # NEW: maximum number of contracts to process for this label in this run
    # NEW: provider/model + reproducibility (Reviewer#2 Concerns #2, #3, #9)
    provider: str = "openai",
    model: Optional[str] = None,
    llm_seed: Optional[int] = None,
    # NEW: oracle vs non-oracle assessment (Reviewer#2 Concern #1)
    oracle: bool = True,
    # NEW: ablation switches (Reviewer#2 Concern #6 / Reviewer#3 Concern #3)
    use_pruning: bool = True,
    use_feedback: bool = True,
    # First N contracts run in oracle mode to warm the memory; they are excluded from reported metrics.
    warmup_contracts: int = 0,
    # Skip contracts that already have a saved result row (continue a run cut off by a usage limit).
    resume: bool = False,
):
    oracle_default = oracle
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

    sol_files = sorted(label_dir.glob("buggy_*.sol"),
                       key=lambda p: int(re.search(r'(\d+)', p.stem).group(1)))
    rows_dir = results_dir / "rows"
    ensure_dir(rows_dir)
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

    for pos, sol_file in enumerate(sol_files):
        m = re.search(r'buggy_(\d+)\.sol', sol_file.name)
        if not m:
            print(f"[SKIP] Non-matching filename: {sol_file.name}")
            continue
        idx = int(m.group(1))
        is_warmup = pos < warmup_contracts
        phase = "warmup" if is_warmup else "eval"
        oracle = oracle_default or is_warmup
        row_path = rows_dir / f"{sol_file.stem}.json"
        if resume and row_path.exists():
            folder_rows.append(json.load(open(row_path, encoding="utf-8")))
            print(f"[RESUME] {sol_file.name} already done ({phase}); skipping.")
            continue
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

        prev_pred_for_consistency: Optional[List[int]] = None

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
                topk_candidates=topk_candidates,
                use_pruning=use_pruning,
                oracle=oracle
            )

            # Buffer header (will decide later)
            header = f"\n[REAL::{folder_name}] {sol_file.name} | Attempt {attempt}"

            llm_text = call_llm_messages(
                messages, api_key, provider=provider, model=model, seed=llm_seed,
                usage_ctx={"label": folder_name, "contract": sol_file.name, "attempt": attempt, "call_type": "prediction"}
            ).strip()
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

                if not use_feedback:
                    # Ablation: feedback component disabled entirely.
                    fb_local_compact = ""
                    fb_rule = ""
                    fb = ""
                elif oracle:
                    # Base compact feedback (candidate-centric, ground-truth informed)
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
                    if smart_feedback == "llm":
                        fb_rule = make_one_line_rule_llm(
                            api_key, model or BASE_MODEL, folder_name, memory_summary, err_prof, rule_chars=fb_rule_chars,
                            provider=provider, seed=llm_seed,
                            usage_ctx={"label": folder_name, "contract": sol_file.name, "attempt": attempt}
                        )
                    elif smart_feedback == "local":
                        fb_rule = make_one_line_rule_local(folder_name, err_prof, rule_chars=fb_rule_chars)
                    else:
                        fb_rule = ""
                    fb_parts = []
                    if memory_summary:
                        fb_parts.append(memory_summary)
                    if fb_rule:
                        fb_parts.append("=== NEXT RULE ===\n" + fb_rule)
                    fb_parts.append(fb_local_compact)
                    fb = "\n\n".join(fb_parts)
                else:
                    # Non-oracle (Reviewer#2 Concern #1): feedback built with NO ground-truth access —
                    # only self-consistency across attempts + fixed per-label rules.
                    rank_now = rank_candidates(contract_text, folder_name, topk_candidates) if use_pruning else []
                    fb_local_compact = build_self_consistency_guidance(
                        label_name=folder_name,
                        contract_text=contract_text,
                        pred_lines=pred_lines,
                        ranked_candidates=rank_now,
                        prev_pred_lines=prev_pred_for_consistency,
                        k=2,
                        max_chars=min(900, fb_max_chars)
                    )
                    fb_rule = ""
                    fb_parts = []
                    if memory_summary:
                        fb_parts.append(memory_summary)
                    fb_parts.append(fb_local_compact)
                    fb = "\n\n".join(fb_parts)

                system_content = f"[prediction] {','.join(str(x) for x in pred_lines)}"
                if oracle:
                    analysis_user = (
                        f"[analysis] "
                        f"[BLOCK/{block_eval}] P={prec_b:.4f} R={rec_b:.4f} F1*100={f1x100_b:.2f}"
                        + (f" HitRate*100={100*hitrate_b:.2f}" if hitrate_b is not None else "")
                        + f" ; [LINE±{line_tolerance}] P={prec_l:.4f} R={rec_l:.4f} F1*100={f1x100_l:.2f} Acc*100={100*acc_l:.2f} ; "
                        f"[feedback_for_next] {fb}"
                    )
                else:
                    # Non-oracle: no score derived from the ground truth is written to memory.
                    analysis_user = f"[analysis] non-oracle ; [feedback_for_next] {fb}"
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
                if not use_feedback:
                    # Single-shot / no-feedback ablation: never iterate.
                    stop_now = True
                elif oracle:
                    stop_now = _should_stop(
                        policy=early_stop,
                        prec_b=prec_b, rec_b=rec_b,
                        prec_l=prec_l, rec_l=rec_l,
                        threshold=threshold,
                        has_line_truth=bool(truth_point)
                    )
                else:
                    # Non-oracle: truth-free convergence criterion (Reviewer#2 Concern #1)
                    stop_now = _should_stop_non_oracle(pred_lines, prev_pred_for_consistency)

                prev_pred_for_consistency = pred_lines[:]

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

                if use_feedback and attempt < max_attempts:
                    immediate_feedback = (
                        "Return only comma-separated integers (e.g., 12,27). "
                        "No words, no ranges, no JSON, no brackets."
                    )
                    continue
                else:
                    break

        # ---- Selection & persistence ----
        if numeric_attempts:
            best = _select_best_attempt(numeric_attempts) if oracle else _select_best_attempt_non_oracle(numeric_attempts)
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
            row = {
                "filename": sol_file.name,
                "phase": phase,
                "selected_attempt": best.attempt_idx,
                "attempts_run": len(attempt_results),
                "BlockDetection.P": best.b_prec,
                "BlockDetection.Recall": best.b_rec,
                "BlockDetection.F1-Score": best.b_f1,
                "BlockDetection.HitRate": best.b_hitrate if block_eval in ("hit", "dilated") else None,
                "LineDetection.P": best.l_prec,
                "LineDetection.Recall": best.l_rec,
                "LineDetection.F1-Score": best.l_f1,
                "LineDetection.Accuracy": best.l_acc,
                "predicted_lines": final_lines,
            }
            folder_rows.append(row)
            json.dump(row, open(row_path, "w", encoding="utf-8"), indent=1)

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

            row = {
                "filename": sol_file.name,
                "phase": phase,
                "selected_attempt": None,
                "attempts_run": len(attempt_results),
                "BlockDetection.P": 0.0,
                "BlockDetection.Recall": 0.0,
                "BlockDetection.F1-Score": 0.0,
                "BlockDetection.HitRate": 0.0 if block_eval in ("hit", "dilated") else None,
                "LineDetection.P": 0.0,
                "LineDetection.Recall": 0.0,
                "LineDetection.F1-Score": 0.0,
                "LineDetection.Accuracy": 0.0,
                "predicted_lines": [],
            }
            folder_rows.append(row)
            json.dump(row, open(row_path, "w", encoding="utf-8"), indent=1)
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
                "filename", "phase", "selected_attempt", "attempts_run",
                "BlockDetection.P", "BlockDetection.Recall", "BlockDetection.F1-Score", "BlockDetection.HitRate",
                "LineDetection.P", "LineDetection.Recall", "LineDetection.F1-Score", "LineDetection.Accuracy",
            ]
            # keep only existing
            cols = [c for c in cols if c in df_folder.columns]
            df_folder = df_folder[cols]
            folder_csv = results_dir / f"{folder_name}_result.csv"
            df_folder.to_csv(folder_csv, index=False)
            # Reported metrics use evaluation contracts only; warm-up contracts are excluded.
            df_eval = df_folder[df_folder["phase"] == "eval"] if "phase" in df_folder.columns else df_folder
            n_eval = len(df_eval)
            n_warmup = len(df_folder) - n_eval
            # Macro averages (simple means)
            def _mean_safe(series_name: str) -> float:
                if series_name not in df_eval.columns or df_eval.empty: return 0.0
                return float(pd.to_numeric(df_eval[series_name], errors="coerce").mean())
            macro_block_p = _mean_safe("BlockDetection.P")
            macro_block_r = _mean_safe("BlockDetection.Recall")
            macro_block_f1 = _mean_safe("BlockDetection.F1-Score")
            macro_line_p  = _mean_safe("LineDetection.P")
            macro_line_r  = _mean_safe("LineDetection.Recall")
            macro_line_f1 = _mean_safe("LineDetection.F1-Score")
            print(f"[INFO] Saved folder summary → {folder_csv}")
            print(f"[SUMMARY::{folder_name}] eval contracts={n_eval} (warm-up excluded={n_warmup}) | "
                  f"MACRO Block P/R/F1 = {macro_block_p:.3f}/{macro_block_r:.3f}/{macro_block_f1:.3f} | "
                  f"MACRO Line P/R/F1 = {macro_line_p:.3f}/{macro_line_r:.3f}/{macro_line_f1:.3f}")

            # ---- Usage/cost log for this label run (Reviewer#2 Concern #10) ----
            usage_csv = results_dir / f"{folder_name}_usage.csv"
            usage_summary = dump_usage_log(usage_csv)
            _USAGE_LOG.clear()
            if usage_summary:
                print(f"[INFO] Saved usage/cost log → {usage_csv}")
                print(f"[USAGE::{folder_name}] calls={usage_summary['num_calls']} "
                      f"tokens={usage_summary['total_tokens']} "
                      f"est_cost_usd={usage_summary['est_total_cost_usd']:.4f} "
                      f"avg_s/call={usage_summary['avg_elapsed_s_per_call']:.2f}")

            return {
                "label": folder_name, "provider": provider, "model": model or BASE_MODEL,
                "oracle": oracle_default, "warmup_contracts": warmup_contracts,
                "n_eval": n_eval, "n_warmup": n_warmup,
                "use_pruning": use_pruning, "use_feedback": use_feedback,
                "macro_block_p": macro_block_p, "macro_block_r": macro_block_r, "macro_block_f1": macro_block_f1,
                "macro_line_p": macro_line_p, "macro_line_r": macro_line_r, "macro_line_f1": macro_line_f1,
                "usage_summary": usage_summary,
            }
    except Exception as e:
        print(f"[WARN] Could not save folder summary: {e}")
    return None

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
    ap.add_argument("--api_key", default=None,
                    help="LLM API key. Optional for provider=anthropic: if omitted, the SDK resolves "
                         "credentials itself (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an "
                         "'ant auth login' profile). Required for the OpenAI-compatible providers.")
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

    # NEW: provider/model selection (Reviewer#2 Concern #3 — explicit, logged model versioning)
    ap.add_argument("--provider", choices=["openai", "anthropic", "claude_cli", "gemini", "groq", "openrouter"], default="openai",
                    help="LLM provider. 'anthropic' enables Claude models (e.g. claude-opus-5); "
                         "'gemini'/'groq'/'openrouter' use their free-tier, OpenAI-wire-compatible endpoints.")
    ap.add_argument("--model", default=None,
                    help="Model id (e.g. gpt-4o, gpt-5, claude-opus-5, gemini-2.0-flash, "
                         "meta-llama/llama-3.3-70b-instruct:free). "
                         "Defaults per-provider.")

    # NEW: oracle vs non-oracle evaluation (Reviewer#2 Concern #1)
    ap.add_argument("--oracle", choices=["on", "off"], default="on",
                    help="'on' (legacy/ceiling): feedback+selection use ground truth. "
                         "'off': feedback+selection are truth-free, as at real audit time.")

    # NEW: ablation controls (Reviewer#2 Concern #6 / Reviewer#3 Concern #3)
    ap.add_argument("--ablation", choices=["full", "single_shot", "pruning_only", "feedback_only"], default="full",
                    help="full=pruning+feedback (baseline); single_shot=no pruning,no feedback,1 attempt; "
                         "pruning_only=pruning,no feedback,1 attempt; feedback_only=no pruning,feedback,multi-attempt.")
    ap.add_argument("--use_pruning", choices=["on", "off"], default=None,
                    help="Override the pruning component independent of --ablation preset.")
    ap.add_argument("--use_feedback", choices=["on", "off"], default=None,
                    help="Override the feedback component independent of --ablation preset.")

    # NEW: statistical stability across repeated runs (Reviewer#2 Concern #9)
    ap.add_argument("--warmup_contracts", type=int, default=0,
                    help="Run the first N contracts of each label in oracle mode to warm the memory; "
                         "they are excluded from the reported metrics. Remaining contracts use --oracle.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip contracts that already have a saved result row (continue after a usage-limit stop).")
    ap.add_argument("--num_runs", type=int, default=1,
                    help="Repeat the selected label run this many times (independent memory/results per run) "
                         "to report mean/std across runs.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Decoding seed passed to the provider when supported (OpenAI 'seed' param), for determinism controls.")

    args = ap.parse_args()

    # Resolve provider/model globals used by call_llm_messages/make_one_line_rule_llm defaults.
    global BASE_MODEL, PROVIDER
    PROVIDER = args.provider
    BASE_MODEL = args.model or DEFAULT_MODEL_BY_PROVIDER.get(args.provider, "gpt-4o")

    # Resolve ablation preset -> (use_pruning, use_feedback, max_attempts), then let explicit
    # --use_pruning/--use_feedback overrides win.
    _ABLATION_PRESETS = {
        "full":          {"use_pruning": True,  "use_feedback": True,  "max_attempts": args.max_attempts},
        "single_shot":   {"use_pruning": False, "use_feedback": False, "max_attempts": 1},
        "pruning_only":  {"use_pruning": True,  "use_feedback": False, "max_attempts": 1},
        "feedback_only": {"use_pruning": False, "use_feedback": True,  "max_attempts": args.max_attempts},
    }
    preset = _ABLATION_PRESETS[args.ablation]
    resolved_use_pruning = preset["use_pruning"] if args.use_pruning is None else (args.use_pruning == "on")
    resolved_use_feedback = preset["use_feedback"] if args.use_feedback is None else (args.use_feedback == "on")
    resolved_max_attempts = preset["max_attempts"]
    resolved_oracle = (args.oracle == "on")

    # Only the Anthropic SDK can resolve credentials on its own (env vars or an OAuth profile).
    if args.api_key is None and args.provider not in ("anthropic", "claude_cli"):
        raise SystemExit(f"--api_key is required for provider={args.provider}.")

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
            f"provider={PROVIDER} model={BASE_MODEL} seed={args.seed} "
            f"oracle={resolved_oracle} ablation={args.ablation} use_pruning={resolved_use_pruning} "
            f"use_feedback={resolved_use_feedback} num_runs={args.num_runs} "
            f"threshold={args.threshold} max_attempts={resolved_max_attempts} "
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

    def _run_one_label(label_key: str, folder_name: str, this_results_root: Path, this_memory_root: Path,
                        counter_state: Dict[str, int]) -> Optional[Dict[str, Any]]:
        return process_label_real(
            label_key=label_key,
            folder_name=folder_name,
            contracts_root=contracts_root,
            results_root=this_results_root,
            memory_root=this_memory_root,
            api_key=args.api_key,
            threshold=args.threshold,
            max_attempts=resolved_max_attempts,
            history_turns=args.history_turns,
            condense_window=args.condense_window,
            topk_candidates=args.topk_candidates,
            block_dilation=args.block_dilation,
            early_stop=args.early_stop,
            block_eval=args.block_eval,
            line_tolerance=args.line_tolerance,
            smart_feedback=args.smart_feedback,
            fb_history_k=args.fb_history_k,
            fb_max_chars=args.fb_max_chars,
            fb_rule_chars=args.fb_rule_chars,
            mem_max_msgs=args.mem_max_msgs,
            mem_keep_recent=args.mem_keep_recent,
            distill_every=args.distill_every,
            contract_counter_state=counter_state,
            limit_contracts=args.limit_contracts,
            provider=PROVIDER,
            model=BASE_MODEL,
            llm_seed=args.seed,
            oracle=resolved_oracle,
            use_pruning=resolved_use_pruning,
            use_feedback=resolved_use_feedback,
            warmup_contracts=args.warmup_contracts,
            resume=args.resume,
        )

    def _run_with_stability(labels_to_run: List[Tuple[str, str]]):
        run_rows: List[Dict[str, Any]] = []
        for run_idx in range(1, args.num_runs + 1):
            if args.num_runs > 1:
                run_results_root = results_root / f"run_{run_idx}"
                run_memory_root = memory_root / f"run_{run_idx}"
                ensure_label_dirs(run_results_root); ensure_label_dirs(run_memory_root)
                print(f"\n--- Stability run {run_idx}/{args.num_runs} (Reviewer#2 Concern #9) ---")
            else:
                run_results_root, run_memory_root = results_root, memory_root
            counter_state: Dict[str, int] = {}
            for label_key, folder_name in labels_to_run:
                print(f"\n=== REAL MODE: Processing Label '{label_key}' → folder '{folder_name}' "
                      f"[provider={PROVIDER} model={BASE_MODEL} oracle={resolved_oracle} ablation={args.ablation}] ===")
                res = _run_one_label(label_key, folder_name, run_results_root, run_memory_root, counter_state)
                if res:
                    res["run_idx"] = run_idx
                    run_rows.append(res)

        if args.num_runs > 1 and run_rows:
            df = pd.DataFrame(run_rows)
            stability_csv = results_root / "stability_summary.csv"
            df.to_csv(stability_csv, index=False)
            agg = df.groupby("label")[["macro_block_f1", "macro_line_f1"]].agg(["mean", "std"])
            print(f"\n[INFO] Saved per-run results → {stability_csv}")
            print(f"[STABILITY] across {args.num_runs} runs (mean ± std):\n{agg}")

    if args.all_labels:
        _run_with_stability(labels_list)
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

    _run_with_stability([(label_key, folder_name)])
    print("\nREAL mode completed for the selected label.")

if __name__ == "__main__":
    try:
        main()
    except UsageLimitReached as e:
        print(f"\n[STOP] Claude usage limit reached: {e}\n"
              "Completed contracts are saved. Rerun the same command with --resume after the limit resets.")
        sys.exit(75)
