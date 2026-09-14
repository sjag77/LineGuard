#!/usr/bin/env python3
"""Per-contract false negatives of the six SolidiFI baseline tools.

Re-implements the false-negative rule of SolidiFI's inspection.py (Ghaleb and
Pattabiraman, ISSTA 2020) on the tool reports published in the SolidiFI-benchmark
repository: an injected bug (loc, length) counts as detected only if the tool
reports an alert whose type maps to that bug category on a line in
[loc, loc + length). Totals over contracts 1..50 must reproduce the published
numbers before any subset is trusted.

Usage: baseline_fn.py <SolidiFI-benchmark/results> <out.json> [first_contract last_contract]
"""
import csv, glob, json, os, re, sys

TOOL_BUGS = {
    "Oyente": ["Re-entrancy", "Timestamp-Dependency", "Unhandled-Exceptions", "TOD", "Overflow-Underflow"],
    "Securify": ["Re-entrancy", "Unchecked-Send", "Unhandled-Exceptions", "TOD"],
    "Mythril": ["Re-entrancy", "Timestamp-Dependency", "Unchecked-Send", "Unhandled-Exceptions", "Overflow-Underflow", "tx.origin"],
    "Smartcheck": ["Re-entrancy", "Timestamp-Dependency", "Unhandled-Exceptions", "Overflow-Underflow", "tx.origin"],
    "Manticore": ["Re-entrancy", "Overflow-Underflow"],
    "Slither": ["Re-entrancy", "Timestamp-Dependency", "Unhandled-Exceptions", "tx.origin"],
}
CODES = {
    "Securify": {"Unhandled-Exceptions": ["UnhandledException"], "TOD": ["TODAmount", "TODReceiver", "TODTransfer"],
                 "Unchecked-Send": ["UnrestrictedEtherFlow"], "Re-entrancy": ["DAOConstantGas", "DAO"]},
    "Mythril": {"Unhandled-Exceptions": ["Unchecked Call Return Value"],
                "Timestamp-Dependency": ["Dependence on predictable environment variable"],
                "Overflow-Underflow": ["Integer Underflow", "Integer Overflow"], "tx.origin": ["Use of tx.origin"],
                "Unchecked-Send": ["Unprotected Ether Withdrawal"],
                "Re-entrancy": ["External Call To User-Supplied Address", "External Call To Fixed Address",
                                "State change after external call"]},
    "Slither": {"Unhandled-Exceptions": ["unchecked-send", "unchecked-lowlevel"], "Timestamp-Dependency": ["timestamp"],
                "tx.origin": ["tx-origin"],
                "Re-entrancy": ["reentrancy-benign", "reentrancy-eth", "reentrancy-unlimited-gas", "reentrancy-no-eth"]},
    "Smartcheck": {"Unhandled-Exceptions": ["SOLIDITY_UNCHECKED_CALL"],
                   "Timestamp-Dependency": ["SOLIDITY_EXACT_TIME", "VYPER_TIMESTAMP_DEPENDENCE"],
                   "Overflow-Underflow": ["SOLIDITY_UINT_CANT_BE_NEGATIVE"], "tx.origin": ["SOLIDITY_TX_ORIGIN"],
                   "Re-entrancy": ["SOLIDITY_ETRNANCY"]},
    "Oyente": {"Unhandled-Exceptions": ["Callstack Depth Attack Vulnerability"],
               "Timestamp-Dependency": ["Timestamp Dependency"], "TOD": ["Transaction-Ordering Dependency"],
               "Re-entrancy": ["Re-Entrancy Vulnerability"], "Overflow-Underflow": ["Integer Overflow", "Integer Underflow"]},
    "Manticore": {"Re-entrancy": ["Potential reentrancy vulnerability", "Reachable ether leak to sender"],
                  "Overflow-Underflow": ["Unsigned integer overflow at ADD instruction", "Signed integer overflow at ADD instruction",
                                         "Unsigned integer overflow at SUB instruction", "Signed integer overflow at SUB instruction"]},
}


def _read(path):
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def alerts_securify(d, cs):
    out = []
    for m in re.finditer(r"Violation\S*\s+for\s+(.+?)\s+in\s+contract.*?\bat\s+\S*?\((\d+)\)", _read(f"{d}/buggy_{cs}.sol.txt"), re.S):
        out.append((int(m.group(2)), m.group(1).strip()))
    return out


def alerts_mythril(d, cs):
    txt = _read(f"{d}/buggy_{cs}.sol.txt")
    out = []
    for m in re.finditer(r"^==== (.+?) ====\s*$(.*?)(?=^==== |\Z)", txt, re.S | re.M):
        lm = re.search(r"sol:(\d+)", m.group(2))
        if lm:
            out.append((int(lm.group(1)), m.group(1).strip()))
    return out


def alerts_smartcheck(d, cs):
    return [(int(m.group(2)), m.group(1).strip())
            for m in re.finditer(r"ruleId:\s*(\S+).*?line:\s*(\d+)", _read(f"{d}/buggy_{cs}.sol.txt"), re.S)]


def alerts_oyente(d, cs):
    out = []
    for f in glob.glob(f"{d}/buggy_{cs}.sol_*.json") + glob.glob(f"{d}/buggy_{cs}.sol:*.json"):
        for m in re.finditer(r"sol:(\d+):\d+:\s*Warning:\s*(.*?)\.(?:\\n|\s*\")", _read(f)):
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def alerts_manticore(d, cs):
    out = []
    for f in glob.glob(f"{d}/buggy_{cs}.*.txt"):
        for m in re.finditer(r"^- (.+?) -\s*$.*?Solidity snippet:\s*\n\s*(\d+)\s", _read(f), re.S | re.M):
            out.append((int(m.group(2)), m.group(1).strip()))
    return out


_slither_last_line = [None]


def alerts_slither(d, cs):
    data = json.loads(_read(f"{d}/buggy_{cs}.sol.json"))
    entries = []

    def walk(o):
        if isinstance(o, dict):
            if "description" in o and "check" in o:
                entries.append((o["check"], o["description"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(data)
    out = []
    for check, desc in entries:
        found = re.findall(r"(?<=sol#)[0-9]*(?=\))", desc)
        if found and found[0]:
            _slither_last_line[0] = int(found[0])
        if _slither_last_line[0] is not None:
            # inspection.py keeps the previous line when a description carries no line reference
            out.append((_slither_last_line[0], check))
    return out


PARSERS = {"Securify": alerts_securify, "Mythril": alerts_mythril, "Smartcheck": alerts_smartcheck,
           "Oyente": alerts_oyente, "Manticore": alerts_manticore, "Slither": alerts_slither}


def bug_log(path):
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.reader(fh))[1:]
    return [(int(r[0]), int(r[1])) for r in rows if len(r) >= 2 and r[0].strip().isdigit()]


def main():
    results_root, out_path = sys.argv[1], sys.argv[2]
    first, last = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) >= 5 else (1, 50)
    report = {}
    for tool, bugs in TOOL_BUGS.items():
        report[tool] = {}
        for bug in bugs:
            base = os.path.join(results_root, tool, "analyzed_buggy_contracts", bug)
            per = {}
            for cs in range(1, 51):
                injected = bug_log(f"{base}/BugLog_{cs}.csv")
                try:
                    alerts = PARSERS[tool](f"{base}/results", cs)
                except (FileNotFoundError, json.JSONDecodeError):
                    alerts = []
                codes = set(CODES[tool][bug])
                fn = sum(1 for loc, length in injected
                         if not any(loc <= line < loc + length and typ in codes for line, typ in alerts))
                per[cs] = {"injected": len(injected), "fn": fn}
            sel = [c for c in range(first, last + 1)]
            report[tool][bug] = {
                "per_contract": per,
                "total_fn_all": sum(per[c]["fn"] for c in range(1, 51)),
                "total_injected_all": sum(per[c]["injected"] for c in range(1, 51)),
                "subset": [first, last],
                "total_fn_subset": sum(per[c]["fn"] for c in sel),
                "total_injected_subset": sum(per[c]["injected"] for c in sel),
            }
    json.dump(report, open(out_path, "w"), indent=1)
    for tool, bugs in report.items():
        for bug, r in bugs.items():
            print(f"{tool:<11}{bug:<22} all50 {r['total_fn_all']:>5} ({r['total_injected_all']:>5})"
                  f"   contracts {first}-{last}: {r['total_fn_subset']:>5} ({r['total_injected_subset']:>5})")


if __name__ == "__main__":
    main()
