#!/usr/bin/env python3
"""Summarise the ablation smoke test: per label and configuration, line-level P/R/F1 (macro over
evaluated contracts), missed injected bugs (a bug is detected if any of its annotated lines is
predicted), model calls and tokens. Only contracts completed by every configuration are compared.

Usage: compare_ablation.py [first last] [cfg,cfg,...]
  default: 6 15 over all four configurations. Contracts are intersected across the
  configurations compared, so restrict the set (e.g. single_shot,pruning_only,full)
  to use the full 45-contract range where feedback_only has not been run.
"""
import csv, json, sys
from pathlib import Path

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

FIRST, LAST = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) >= 3 else (6, 15)
LABELS = ["Re-entrancy", "Timestamp-Dependency", "Unchecked-Send", "Unhandled-Exceptions", "TOD", "Overflow-Underflow", "tx.origin"]
ONLY = sys.argv[3].split(",") if len(sys.argv) >= 4 else None
CONFIGS = {"full": Path("results_sonnet_v2"),
           "pruning_only": Path("results_ablation_smoke/pruning_only"),
           "feedback_only": Path("results_ablation_smoke/feedback_only"),
           "single_shot": Path("results_ablation_smoke/single_shot")}
if ONLY:
    CONFIGS = {k: v for k, v in CONFIGS.items() if k in ONLY}
ORDER = [c for c in ["single_shot", "pruning_only", "feedback_only", "full"] if c in CONFIGS]


def bugs(label, n):
    spans = {}
    with open(f"buggy_contracts/{label}/BugLog_{n}.csv", newline="", encoding="utf-8", errors="ignore") as fh:
        for r in list(csv.reader(fh))[1:]:
            if len(r) >= 3 and r[0].strip().isdigit():
                spans.setdefault((r[1].strip(), r[2].strip()), set()).add(int(r[0]))
    return list(spans.values())


def usage(root, label, contracts):
    """Calls and tokens for `contracts`, plus how many of them the log actually covers.

    main.py rewrites <label>_usage.csv at the end of each label run with only that
    process's calls, and writes nothing if the run stops at the usage limit. After a
    --resume the log therefore covers only the contracts processed in the last run, so
    counts are reported as incomplete rather than as measurements."""
    calls = toks = 0
    seen = set()
    p = root / label / f"{label}_usage.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["contract"] in contracts:
                calls += 1
                toks += int(r["total_tokens"] or 0)
                seen.add(r["contract"])
    return calls, toks, len(seen)


out = {}
for label in LABELS:
    rows = {}
    for cfg, root in CONFIGS.items():
        rows[cfg] = {}
        for n in range(FIRST, LAST + 1):
            p = root / label / "rows" / f"buggy_{n}.json"
            if p.exists():
                rows[cfg][n] = json.load(open(p))
    common = sorted(set.intersection(*(set(v) for v in rows.values())))
    print(f"\n{label}: {len(common)} contracts completed by all configurations {common}")
    if not common:
        continue
    names = {f"buggy_{n}.sol" for n in common}
    print(f"{'config':<14}{'P':>7}{'R':>7}{'F1':>7}{'missed':>12}{'calls':>7}{'tokens':>10}{'dF1 vs single':>15}")
    base = None
    res = {}
    for cfg in ORDER:
        m = lambda k: sum(rows[cfg][n][f"LineDetection.{k}"] for n in common) / len(common)
        missed = total = 0
        for n in common:
            pred = set(rows[cfg][n]["predicted_lines"])
            for lines in bugs(label, n):
                total += 1
                missed += not (lines & pred)
        calls, toks, covered = usage(CONFIGS[cfg], label, names)
        partial = "*" if covered < len(common) else " " 
        f1 = m("F1-Score")
        base = f1 if base is None else base
        res[cfg] = dict(P=m("P"), R=m("Recall"), F1=f1, missed=missed, bugs=total,
                        calls=calls, tokens=toks, usage_contracts=covered,
                        usage_complete=covered == len(common))
        print(f"{cfg:<14}{m('P'):>7.3f}{m('Recall'):>7.3f}{f1:>7.3f}{f'{missed}/{total}':>12}"
              f"{calls:>6}{partial}{toks:>10}{partial}{f1-base:>+14.3f}")
    # Paired per-contract comparison: same contracts in every configuration, so the
    # difference is taken contract by contract and tested with a signed-rank test.
    per = {c: [rows[c][n]["LineDetection.F1-Score"] for n in common] for c in res}
    print(f"  {'paired vs single_shot':<24}{'median dF1':>12}{'better':>9}{'worse':>7}{'p':>10}")
    for c in [c for c in ORDER if c != "single_shot"]:
        diffs = [a - b for a, b in zip(per[c], per["single_shot"])]
        nz = [d for d in diffs if d != 0]
        med = sorted(diffs)[len(diffs) // 2] if diffs else 0.0
        if wilcoxon is None or len(nz) < 6:
            pv = float("nan")  # too few non-tied pairs for a meaningful test
        else:
            pv = float(wilcoxon(per[c], per["single_shot"], zero_method="wilcox").pvalue)
        res[c]["median_dF1"] = med
        res[c]["n_better"] = sum(d > 0 for d in diffs)
        res[c]["n_worse"] = sum(d < 0 for d in diffs)
        res[c]["wilcoxon_p"] = pv
        print(f"  {c:<24}{med:>+12.3f}{res[c]['n_better']:>9}{res[c]['n_worse']:>7}"
              f"{('  n/a' if pv != pv else f'{pv:>10.4f}')}")

    s = res["single_shot"]["F1"]
    for c in [c for c in ORDER if c != "single_shot"]:
        print(f"  {c} - single_shot: {res[c]['F1']-s:+.3f} F1")
    out[label] = {"contracts": common, "configs": res}
Path("reports").mkdir(exist_ok=True)
print("\n* call/token counts incomplete: the usage log covers fewer contracts than compared "
      "(a resumed or limit-interrupted run). Detection metrics are unaffected.")
json.dump(out, open(f"reports/ablation_{FIRST}_{LAST}_{'_'.join(ORDER)}.json", "w"), indent=1)
