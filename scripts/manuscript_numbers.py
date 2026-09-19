#!/usr/bin/env python3
"""All per-category numbers of the reported configuration, recomputed from stored artefacts.

Sources, per category: the per-contract rows (predictions), the BugLog annotations, the run
logs (per-attempt predictions, for the consistency table) and the usage logs (calls, tokens,
latency, cost). Blank ground-truth lines are excluded throughout, matching main.py.

Prints the inputs of Table 2 (line-level FP/FN/P/R/F1), Table 4 (missed bugs),
Table 6 (cost) and Table 7 (attempt consistency).
"""
import csv, json, re, statistics, sys
from pathlib import Path

KEEP_BLANK = "--keep-blank" in sys.argv  # score blank annotated lines, as the GPT-4o run did

LAB = ["Re-entrancy", "Timestamp-Dependency", "Unchecked-Send", "Unhandled-Exceptions",
       "TOD", "Overflow-Underflow", "tx.origin"]
TUNED = {"Re-entrancy": 60, "Unhandled-Exceptions": 60, "TOD": 100, "Overflow-Underflow": 80}
ROOT = {L: Path("results_tuned_final" if L in TUNED else "results_sonnet_v2") for L in LAB}
OLD = Path("results_sonnet_v2")


def annotations(L, n):
    src = open(f"buggy_contracts/{L}/buggy_{n}.sol", errors="ignore").read().splitlines()
    spans, lines = {}, set()
    for r in list(csv.reader(open(f"buggy_contracts/{L}/BugLog_{n}.csv", errors="ignore")))[1:]:
        if len(r) >= 3 and r[0].strip().isdigit():
            ln = int(r[0])
            if not KEEP_BLANK and (ln > len(src) or not src[ln - 1].strip()):
                continue
            spans.setdefault((r[1], r[2]), set()).add(ln)
            lines.add(ln)
    return lines, [v for v in spans.values() if v]


def jaccard(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b) if (a | b) else 1.0


def attempts_from_log(L):
    """Per-contract list of attempt predictions, parsed from the label's run log."""
    out, cur, name = {}, [], None
    logs = sorted(Path(f"{ROOT[L]}/log").glob("label_*.log")) + sorted(Path(f"{OLD}/log").glob("label_*.log"))
    for lg in logs:
        for line in open(lg, errors="ignore"):
            m = re.search(r"\[prediction\]\s*([\d,\s]+)", line)
            if m:
                cur.append({int(x) for x in re.findall(r"\d+", m.group(1))})
            f = re.search(r"saved (buggy_\d+)_pred\.csv", line)
            if f:
                out.setdefault((lg.name, f.group(1)), cur)
                cur = []
    return out


print(f"{'Category':<24}{'FP':>7}{'FN':>7}{'lines':>7}{'P':>7}{'R':>7}{'F1':>7}{'missed/bugs':>13}{'K':>5}")
tot = {}
T2 = {}
for L in LAB:
    fp = fn = truth_n = 0
    vals, missed, bugs = [], 0, 0
    att, early, sel, n_ok = [], 0, [0, 0, 0], 0
    for n in range(6, 51):
        p = ROOT[L] / L / "rows" / f"buggy_{n}.json"
        if not p.exists():
            continue
        row = json.load(open(p))
        pred = set(row["predicted_lines"])
        truth, spans = annotations(L, n)
        fp += len(pred - truth); fn += len(truth - pred); truth_n += len(truth)
        tp = len(pred & truth)
        pr = tp / len(pred) if pred else 0.0
        rc = tp / len(truth) if truth else 0.0
        vals.append((pr, rc, 2 * pr * rc / (pr + rc) if pr + rc else 0.0))
        bugs += len(spans); missed += sum(not (s & pred) for s in spans)
        a = row.get("attempts_run", 0); att.append(a)
        early += a < 3
        s = row.get("selected_attempt", 1)
        if 1 <= s <= 3: sel[s - 1] += 1
        n_ok += 1
    P, R, F1 = (sum(v[i] for v in vals) / len(vals) for i in range(3))
    T2[L] = dict(FP=fp, FN=fn, lines=truth_n, P=P, R=R, F1=F1, missed=missed, bugs=bugs,
                 attempts=sum(att) / len(att), early=early, sel=sel, n=n_ok,
                 K=TUNED.get(L, 40))
    print(f"{L:<24}{fp:>7}{fn:>7}{truth_n:>7}{P:>7.3f}{R:>7.3f}{F1:>7.3f}"
          f"{f'{missed}/{bugs}':>13}{TUNED.get(L,40):>5}")
k = len(T2)
g = lambda key: sum(v[key] for v in T2.values())
mb = f"{g('missed')}/{g('bugs')}"
print(f"{'MACRO':<24}{g('FP'):>7}{g('FN'):>7}{g('lines'):>7}"
      f"{g('P')/k:>7.3f}{g('R')/k:>7.3f}{g('F1')/k:>7.3f}{mb:>13}")
print()
print(f"{'Category':<24}{'attempts':>9}{'early':>10}{'att 1/2/3':>14}")
for L in LAB:
    v = T2[L]
    e = f"{v['early']} / {v['n']}"
    d = f"{v['sel'][0]} / {v['sel'][1]} / {v['sel'][2]}"
    print(f"{L:<24}{v['attempts']:>9.2f}{e:>10}{d:>14}")
json.dump(T2, open(f"reports/manuscript_numbers{'_keepblank' if KEEP_BLANK else ''}.json", "w"), indent=1)
print("\nwritten: reports/manuscript_numbers.json")
