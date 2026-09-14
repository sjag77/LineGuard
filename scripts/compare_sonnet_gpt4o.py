#!/usr/bin/env python3
"""Build a per-contract comparison of the Sonnet run against the recorded GPT-4o run.

GPT-4o metrics come from the October execution logs; Sonnet metrics from a
results_root produced by main.py. Emits JSON consumed by the HTML report.
"""
import json, re, sys
from pathlib import Path
from collections import Counter

LABEL = "Re-entrancy"


def parse_log(log_path: Path, label: str) -> dict:
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    pat = (rf'\[REAL::{re.escape(label)}\] (buggy_\d+)\.sol \| Attempt (\d+)\n'
           r'[^\n]*\n'
           r'\s+EVAL\s+→ \[BLOCK/dilated\] P=([\d.]+), R=([\d.]+), F1\*100=([\d.]+)[^|]*'
           r'\| \[LINE±0\] P=([\d.]+), R=([\d.]+), F1\*100=([\d.]+), Acc\*100=([\d.]+)[^\n]*\n'
           r'\s+FINAL')
    out = {}
    for name, att, bp, br, bf, lp, lr, lf, la in re.findall(pat, txt):
        out[name] = dict(attempt=int(att), b_p=float(bp), b_r=float(br), b_f1=float(bf) / 100,
                         l_p=float(lp), l_r=float(lr), l_f1=float(lf) / 100, l_acc=float(la) / 100)
    return out


def macro(rows: dict, key: str) -> float:
    return sum(r[key] for r in rows.values()) / len(rows) if rows else 0.0


def main():
    gpt_log, sonnet_log, out_path = map(Path, sys.argv[1:4])
    gpt = parse_log(gpt_log, LABEL)
    son = parse_log(sonnet_log, LABEL)
    shared = sorted(set(gpt) & set(son), key=lambda s: int(s.split('_')[1]))

    per = []
    for c in shared:
        g, s = gpt[c], son[c]
        per.append({
            "contract": c,
            "gpt": {k: round(g[k], 4) for k in ("b_p", "b_r", "b_f1", "l_p", "l_r", "l_f1")} | {"attempt": g["attempt"]},
            "son": {k: round(s[k], 4) for k in ("b_p", "b_r", "b_f1", "l_p", "l_r", "l_f1")} | {"attempt": s["attempt"]},
            "d_l_f1": round(s["l_f1"] - g["l_f1"], 4),
            "d_b_f1": round(s["b_f1"] - g["b_f1"], 4),
        })

    keys = ("b_p", "b_r", "b_f1", "l_p", "l_r", "l_f1")
    summary = {
        "n_shared": len(shared),
        "n_gpt": len(gpt), "n_sonnet": len(son),
        "gpt_macro": {k: round(macro({c: gpt[c] for c in shared}, k), 4) for k in keys},
        "son_macro": {k: round(macro({c: son[c] for c in shared}, k), 4) for k in keys},
        "gpt_attempts": dict(sorted(Counter(gpt[c]["attempt"] for c in shared).items())),
        "son_attempts": dict(sorted(Counter(son[c]["attempt"] for c in shared).items())),
        "line_f1_wins": sum(1 for p in per if p["d_l_f1"] > 0),
        "line_f1_losses": sum(1 for p in per if p["d_l_f1"] < 0),
        "line_f1_ties": sum(1 for p in per if p["d_l_f1"] == 0),
        "block_f1_wins": sum(1 for p in per if p["d_b_f1"] > 0),
        "block_f1_losses": sum(1 for p in per if p["d_b_f1"] < 0),
        "block_f1_ties": sum(1 for p in per if p["d_b_f1"] == 0),
    }
    json.dump({"summary": summary, "per_contract": per}, open(out_path, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
