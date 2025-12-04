#!/usr/bin/env python3
"""
report.py  –  Summarise correction quality for one logical dataset.
Usage
-----
python report.py  [-o report.txt]  result_file1.txt  [result_file2.txt …]
Each argument is an NDJSON file produced by parallel_correction.py.
All files are concatenated in-memory to form a single UB dataset.
Metrics written:
  • UBs_correct      – 
  • UBs_total        – total 
  • Matches_correct  – sum of correctly labelled matches across snapshots
  • Matches_total    – sum of all matches
  • Precision / Recall / F1  – micro scores over matches
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any
def load_one_file(path: Path) -> List[Dict[str, Any]]:
    """Return list of UBs from an NDJSON file."""
    ubs = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                ubs.append(json.loads(line))
    return ubs
def dominant_true_label(ub: dict) -> str:
    """Most frequent RHS value in the *clean* data."""
    c = Counter()
    for snap_map in ub.get("values_by_snapshot", {}).values():
        c.update(snap_map)
    return c.most_common(1)[0][0] if c else ""
def predicted_label(ub: dict) -> str:
    """The label chosen by the cleaner (first key of injected snapshot 0)."""
    injected = ub.get("values_by_snapshot_injected", {})
    if not injected:
        return ""
    snap = sorted(injected, key=int)[0]
    first_key = next(iter(injected[snap]), "")
    return first_key
def evaluate(ubs: List[dict]) -> Dict[str, float]:
    ubs_total = len(ubs)
    ubs_correct = 0
    matches_total = 0
    matches_correct = 0
    for ub in ubs:
        true_label = dominant_true_label(ub)
        pred_label = predicted_label(ub)
        if pred_label == true_label and true_label:
            ubs_correct += 1
        for snap_map in ub.get("values_by_snapshot", {}).values():
            cnt = sum(snap_map.values())
            matches_total += cnt
            if pred_label == true_label:
                matches_correct += cnt
    precision = matches_correct / matches_total if matches_total else 0.0
    recall = precision  
    f1 = precision  
    return dict(
        UBs_correct=ubs_correct,
        UBs_total=ubs_total,
        Matches_correct=matches_correct,
        Matches_total=matches_total,
        Precision=precision,
        Recall=recall,
        F1=f1,
    )
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_files", nargs="+", help="NDJSON correction outputs")
    ap.add_argument("-o", "--output", default="report.txt", help="report file")
    args = ap.parse_args()
    all_ubs: List[dict] = []
    for p in args.input_files:
        all_ubs.extend(load_one_file(Path(p)))
    summary = evaluate(all_ubs)
    with Path(args.output).open("w") as fh:
        for k, v in summary.items():
            fh.write(f"{k}\t{v}\n")
    print(f"Wrote summary to {args.output}")
if __name__ == "__main__":
    main()
