#!/usr/bin/env python3
"""Tuned configuration vs the currently reported one, on the same contracts.

Current = results_sonnet_v2 (topk=40, original TOD patterns, the numbers in the paper).
Tuned   = results_tune      (topk=100, extended TOD extraction).
Reports macro line-level P/R/F1 and missed injected bugs under the per-bug rule, plus the
paired per-contract F1 difference. Line-level P/R/F1 are recomputed here from the stored predictions rather than read from the
row files, so both configurations are scored under the same convention even when they were
produced before the blank-line fix. Blank ground-truth lines are excluded by default
(they carry no statement); --keep-blank scores them as annotated, as the original run did.

Usage: compare_tuning.py [first last] [--keep-blank] [--tuned=results_tune2]
"""
import csv, json, sys
from pathlib import Path

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

SKIP_BLANK = "--keep-blank" not in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST, LAST = (int(args[0]), int(args[1])) if len(args) >= 2 else (6, 15)
LABELS = ["Re-entrancy", "TOD"]
TUNED = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tuned=")), "results_tune")
SRC = {"current (paper)": Path("results_sonnet_v2"), "tuned": Path(TUNED)}


def truth_lines(label, n):
    """Annotated vulnerable lines, blank lines excluded unless --keep-blank."""
    src = open(f"buggy_contracts/{label}/buggy_{n}.sol", errors="ignore").read().splitlines()
    out = set()
    for r in list(csv.reader(open(f"buggy_contracts/{label}/BugLog_{n}.csv", errors="ignore")))[1:]:
        if len(r) >= 3 and r[0].strip().isdigit():
            ln = int(r[0])
            if SKIP_BLANK and (ln > len(src) or not src[ln - 1].strip()):
                continue
            out.add(ln)
    return out


def prf(pred, truth):
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def spans(label, n):
    src = open(f"buggy_contracts/{label}/buggy_{n}.sol", errors="ignore").read().splitlines()
    out = {}
    for r in list(csv.reader(open(f"buggy_contracts/{label}/BugLog_{n}.csv", errors="ignore")))[1:]:
        if len(r) >= 3 and r[0].strip().isdigit():
            ln = int(r[0])
            if SKIP_BLANK and (ln > len(src) or not src[ln - 1].strip()):
                continue
            out.setdefault((r[1], r[2]), set()).add(ln)
    return [v for v in out.values() if v]


out = {}
for label in LABELS:
    rows = {}
    for name, root in SRC.items():
        rows[name] = {}
        for n in range(FIRST, LAST + 1):
            p = root / label / "rows" / f"buggy_{n}.json"
            if p.exists():
                rows[name][n] = json.load(open(p))
    common = sorted(set.intersection(*(set(v) for v in rows.values())))
    if not common:
        print(f"\n{label}: no contracts finished in both configurations yet")
        continue
    print(f"\n{label}: {len(common)} contracts ({common[0]}-{common[-1]})")
    print(f"{'config':<18}{'P':>7}{'R':>7}{'F1':>7}{'missed bugs':>14}{'calls':>7}")
    res = {}
    scored = {}
    for name in SRC:
        missed = total = 0
        vals = []
        for n in common:
            pred = set(rows[name][n]["predicted_lines"])
            vals.append(prf(pred, truth_lines(label, n)))
            for s in spans(label, n):
                total += 1
                missed += not (s & pred)
        scored[name] = [v[2] for v in vals]
        P, R, F1 = (sum(v[i] for v in vals) / len(vals) for i in range(3))
        calls = sum(rows[name][n].get("attempts_run", 0) for n in common)
        res[name] = dict(P=P, R=R, F1=F1, missed=missed, bugs=total)
        print(f"{name:<18}{P:>7.3f}{R:>7.3f}{F1:>7.3f}{f'{missed}/{total}':>14}{calls:>7}")
    a, b = scored["tuned"], scored["current (paper)"]
    diffs = [x - y for x, y in zip(a, b)]
    nz = [d for d in diffs if d]
    pv = float(wilcoxon(a, b).pvalue) if wilcoxon and len(nz) >= 6 else float("nan")
    print(f"  dF1 {res['tuned']['F1']-res['current (paper)']['F1']:+.3f}   "
          f"missed {res['current (paper)']['missed']} -> {res['tuned']['missed']}   "
          f"better/worse {sum(d>0 for d in diffs)}/{sum(d<0 for d in diffs)}   "
          f"p {'n/a' if pv != pv else f'{pv:.4f}'}")
    out[label] = res
json.dump(out, open(f"reports/tuning_{Path(TUNED).name}_{FIRST}_{LAST}{'_noblank' if SKIP_BLANK else '_keepblank'}.json", "w"), indent=1)
