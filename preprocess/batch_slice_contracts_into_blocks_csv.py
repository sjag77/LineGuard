#!/usr/bin/env python3
"""
batch_slice_contracts_into_blocks_csv.py

Batch processes multiple labeled directories of Solidity contracts + metadata, and writes
per-contract CSVs of fully bracketed blocks (functions + imaginary fillers), with vulnerable
lines assigned to the correct block.

Directory layout (example):
  <INPUT_ROOT>/
    Re-entrancy/
      buggy_1.sol
      BugLog_1.csv
      buggy_2.sol
      BugLog_2.csv
      ...
    Timestamp-Dependency/
      buggy_1.sol
      BugLog_1.csv
      ...
    Unchecked-Send/
    Unhandled-Exceptions/
    TOD/
    Overflow-Underflow/
    tx.origin/

Output layout (mirrors labels):
  <OUTPUT_ROOT>/
    Re-entrancy/
      buggy_1_blocks.csv
      buggy_2_blocks.csv
      ...
    Timestamp-Dependency/
      ...
  + master index: <OUTPUT_ROOT>/_index.csv

Usage:
  python batch_slice_contracts_into_blocks_csv.py <INPUT_ROOT> <OUTPUT_ROOT>

Notes:
- Only start/end line indexes + vulnerability mapping are written (no function code).
- Completely covers the contract from line 1..N using:
    - function blocks (brace-matched)
    - imaginary blocks (to fill gaps)
- Expects pairs: buggy_<i>.sol + BugLog_<i>.csv (case-sensitive by default).
"""

import re
import sys
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd

# --------------------------
# CONFIG: label folders
# --------------------------
LABEL_FOLDERS = {
    "Re-entrancy": "Re-entrancy",
    "Timestamp dep": "Timestamp-Dependency",
    "Unchecked-send": "Unchecked-Send",
    "Unhandled exp": "Unhandled-Exceptions",
    "TOD": "TOD",
    "Integer flow": "Overflow-Underflow",
    "tx.origin": "tx.origin",
}

# --------------------------
# Helpers from single-file flow
# --------------------------


def read_file_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_vulnerable_lines_from_csv_or_text(path: Path) -> List[int]:
    """
    Reads a CSV (or raw text) and extracts integer line numbers.
    Works even if line numbers appear as comma/semicolon/space/pipe-separated lists or as a single column.
    """
    text = path.read_text(encoding="utf-8")
    vuln_set = set()

    # Try CSV first
    try:
        df = pd.read_csv(path)
        # Candidate columns likely containing line numbers
        cand = [c for c in df.columns if re.search(r'line|loc|linen', c, re.I)]
        if not cand and df.shape[1] == 1:
            cand = [df.columns[0]]

        for c in cand:
            # Normalize to strings and split on common separators
            for v in df[c].dropna().astype(str):
                for part in re.split(r'[;, \|]+', v.strip()):
                    if not part:
                        continue
                    if part.isdigit():
                        vuln_set.add(int(part))
                    else:
                        m = re.search(r'(\d+)', part)
                        if m:
                            vuln_set.add(int(m.group(1)))

        if vuln_set:
            return sorted(vuln_set)
    except Exception:
        # Fall through to text parsing
        pass

    # Fallback: extract all integers from raw text
    for m in re.finditer(r'\d+', text):
        vuln_set.add(int(m.group(0)))

    return sorted(vuln_set)


def find_function_blocks(lines: List[str]) -> List[Dict]:
    """
    Finds function code blocks by matching braces starting from 'function' declarations.
    Returns: list of dicts {name, start, end}, 1-based inclusive line numbers.
    """
    func_blocks = []
    func_pat = re.compile(r'\bfunction\b')
    name_pat = re.compile(r'function\s+([A-Za-z_][A-Za-z0-9_]*)')
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        if func_pat.search(line):
            start_idx = i
            brace_pos = None
            # find opening brace for the function body
            for j in range(i, min(n, i + 40)):
                if '{' in lines[j]:
                    brace_pos = j
                    break
                # interface or abstract function without body
                if ';' in lines[j]:
                    brace_pos = None
                    break
            if brace_pos is None:
                i += 1
                continue

            # match braces from the opening '{'
            brace_count = 0
            end_idx = brace_pos
            for k in range(brace_pos, n):
                brace_count += lines[k].count('{')
                brace_count -= lines[k].count('}')
                if brace_count == 0:
                    end_idx = k
                    break

            header_text = " ".join(lines[start_idx:brace_pos + 1])
            m = name_pat.search(header_text)
            fname = m.group(1) if m else f'function_at_{start_idx+1}'

            func_blocks.append({
                "name": fname,
                "start": start_idx + 1,
                "end": end_idx + 1
            })
            i = end_idx + 1
        else:
            i += 1

    return func_blocks


def build_full_block_partition(file_lines: List[str], function_blocks: List[Dict]) -> List[Dict]:
    """
    Merge function blocks + imaginary filler blocks to fully cover [1..len(file_lines)].
    Produces non-overlapping, ordered blocks.
    """
    n_lines = len(file_lines)
    funcs = sorted(function_blocks, key=lambda x: x['start'])
    blocks, cur, block_id = [], 1, 1

    for f in funcs:
        if f['start'] > cur:
            blocks.append({
                'block_id': block_id,
                'block_type': 'imaginary',
                'name': f'block_{block_id}_imag',
                'start': cur,
                'end': f['start'] - 1
            })
            block_id += 1
        blocks.append({
            'block_id': block_id,
            'block_type': 'function',
            'name': f['name'],
            'start': f['start'],
            'end': f['end']
        })
        block_id += 1
        cur = f['end'] + 1

    if cur <= n_lines:
        blocks.append({
            'block_id': block_id,
            'block_type': 'imaginary',
            'name': f'block_{block_id}_imag',
            'start': cur,
            'end': n_lines
        })

    if not funcs and not blocks:
        blocks.append({
            'block_id': 1,
            'block_type': 'imaginary',
            'name': 'block_1_imag',
            'start': 1,
            'end': n_lines
        })

    return blocks


def assign_vulns_to_blocks(blocks: List[Dict], vuln_lines: List[int]) -> List[Dict]:
    vuln_set = set(vuln_lines)
    for b in blocks:
        s, e = b['start'], b['end']
        b_vulns = sorted([ln for ln in vuln_set if s <= ln <= e])
        b['vulnerable_lines_in_block'] = ",".join(
            map(str, b_vulns)) if b_vulns else ""
    return blocks


def process_single_contract(sol_file: Path, meta_file: Path, out_csv: Path) -> Tuple[int, int]:
    """
    Process one pair (solidity, metadata) -> write block CSV.
    Returns: (#blocks_written, #vuln_lines_detected)
    """
    lines = read_file_lines(sol_file)
    vuln_lines = parse_vulnerable_lines_from_csv_or_text(meta_file)
    func_blocks = find_function_blocks(lines)
    blocks = build_full_block_partition(lines, func_blocks)
    blocks = assign_vulns_to_blocks(blocks, vuln_lines)

    df = pd.DataFrame([{
        "block_id": b['block_id'],
        "block_type": b['block_type'],
        "name": b['name'],
        "start_line": b['start'],
        "end_line": b['end'],
        "vulnerable_lines_in_block": b['vulnerable_lines_in_block']
    } for b in blocks])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return len(df), len(vuln_lines)

# --------------------------
# Batch driver
# --------------------------


def main(input_root: str, output_root: str):
    in_root = Path(input_root).resolve()
    out_root = Path(output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Master index rows
    index_rows = []
    contract_re = re.compile(r'buggy_(\d+)\.sol\Z')

    for label, folder_name in LABEL_FOLDERS.items():
        src_dir = in_root / folder_name
        dst_dir = out_root / folder_name
        if not src_dir.exists():
            print(f"[WARN] Missing input folder for '{label}': {src_dir}")
            continue

        # Find all buggy_*.sol files
        sol_files = sorted(
            [p for p in src_dir.glob("buggy_*.sol") if p.is_file()])

        if not sol_files:
            print(f"[WARN] No contracts found in {src_dir}")
            continue

        print(
            f"==> Processing label '{label}' in {src_dir} ({len(sol_files)} contracts)")

        for sol_file in sol_files:
            m = contract_re.search(sol_file.name)
            if not m:
                print(f"  [SKIP] Non-matching filename: {sol_file.name}")
                continue
            idx = m.group(1)
            meta_file = src_dir / f"BugLog_{idx}.csv"
            if not meta_file.exists():
                print(
                    f"  [WARN] Missing metadata for {sol_file.name}: {meta_file.name}")
                continue

            out_csv = dst_dir / f"buggy_{idx}_blocks.csv"

            try:
                n_blocks, n_vulns = process_single_contract(
                    sol_file, meta_file, out_csv)
                print(
                    f"  [OK] {sol_file.name} -> {out_csv.name}  (#blocks={n_blocks}, #vuln_lines={n_vulns})")
                index_rows.append({
                    "label": label,
                    "label_folder": folder_name,
                    "contract_id": int(idx),
                    "sol_path": str(sol_file),
                    "meta_path": str(meta_file),
                    "out_csv": str(out_csv),
                    "num_blocks": n_blocks,
                    "num_vuln_lines_in_metadata": n_vulns
                })
            except Exception as e:
                print(f"  [ERR] Failed {sol_file.name}: {e}")
                index_rows.append({
                    "label": label,
                    "label_folder": folder_name,
                    "contract_id": int(idx),
                    "sol_path": str(sol_file),
                    "meta_path": str(meta_file),
                    "out_csv": str(out_csv),
                    "num_blocks": 0,
                    "num_vuln_lines_in_metadata": 0,
                    "error": str(e)
                })

    # Write master index
    if index_rows:
        index_df = pd.DataFrame(index_rows).sort_values(
            by=["label", "contract_id"])
        index_csv = out_root / "_index.csv"
        index_df.to_csv(index_csv, index=False)
        print(f"\n==> Wrote master index: {index_csv} (rows={len(index_df)})")
    else:
        print("\n[WARN] No results to index. Check input paths and file naming.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_slice_contracts_into_blocks_csv.py <INPUT_ROOT> <OUTPUT_ROOT>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
