#!/usr/bin/env python3
"""Recompute the reported per-category metrics from stored per-contract predictions.

Every label is scored by the same rules, so results produced by different runs can be
combined: line-level precision/recall/F1 are macro averages over the 45 evaluation
contracts, and missed bugs use the per-bug rule (a bug counts as detected when at least
one of its annotated lines is predicted). Blank ground-truth lines are excluded, matching
the scoring fix in main.py; --keep-blank reproduces the earlier convention.

Labels default to results_sonnet_v2; override per label as Label=path, e.g.
  paper_metrics.py Re-entrancy=results_tuned_final TOD=results_tuned_final
"""
import csv, json, sys
from pathlib import Path

KEEP_BLANK = "--keep-blank" in sys.argv
LABELS = ["Re-entrancy", "Timestamp-Dependency", "Unchecked-Send", "Unhandled-Exceptions",
          "TOD", "Overflow-Underflow", "tx.origin"]
roots = {L: Path("results_sonnet_v2") for L in LABELS}
for a in sys.argv[1:]:
    if "=" in a and not a.startswith("--"):
        k, v = a.split("=", 1)
        roots[k] = Path(v)


def annotations(label, n):
    """(line-level truth, list of per-bug line sets) for one contract."""
    src = open(f"buggy_contracts/{label}/buggy_{n}.sol", errors="ignore").read().splitlines()
    spans, lines = {}, set()
    for r in list(csv.reader(open(f"buggy_contracts/{label}/BugLog_{n}.csv", errors="ignore")))[1:]:
        if len(r) >= 3 and r[0].strip().isdigit():
            ln = int(r[0])
            if not KEEP_BLANK and (ln > len(src) or not src[ln - 1].strip()):
                continue
            spans.setdefault((r[1], r[2]), set()).add(ln)
            lines.add(ln)
    return lines, [v for v in spans.values() if v]


print(f"{'Category':<22}{'P':>7}{'R':>7}{'F1':>7}{'missed/bugs':>14}{'n':>4}  source")
tot = {"P": 0.0, "R": 0.0, "F1": 0.0, "missed": 0, "bugs": 0}
out = {}
for L in LABELS:
    vals, missed, bugs, n_ok = [], 0, 0, 0
    for n in range(6, 51):
        p = roots[L] / L / "rows" / f"buggy_{n}.json"
        if not p.exists():
            continue
        pred = set(json.load(open(p))["predicted_lines"])
        truth, spans = annotations(L, n)
        tp = len(pred & truth)
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(truth) if truth else 0.0
        vals.append((prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0))
        bugs += len(spans)
        missed += sum(not (s & pred) for s in spans)
        n_ok += 1
    if not vals:
        print(f"{L:<22}{'no rows':>25}")
        continue
    P, R, F1 = (sum(v[i] for v in vals) / len(vals) for i in range(3))
    for k, v in zip(("P", "R", "F1"), (P, R, F1)):
        tot[k] += v
    tot["missed"] += missed
    tot["bugs"] += bugs
    out[L] = dict(P=P, R=R, F1=F1, missed=missed, bugs=bugs, contracts=n_ok, source=str(roots[L]))
    print(f"{L:<22}{P:>7.3f}{R:>7.3f}{F1:>7.3f}{f'{missed}/{bugs}':>14}{n_ok:>4}  {roots[L]}")
k = len(out)
print(f"{'MACRO':<22}{tot['P']/k:>7.3f}{tot['R']/k:>7.3f}{tot['F1']/k:>7.3f}"
      f"{f'{tot ['missed']}/{tot['bugs']}':>14}")
json.dump(out, open(f"reports/paper_metrics{'_keepblank' if KEEP_BLANK else ''}.json", "w"), indent=1)
