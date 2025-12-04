#!/usr/bin/env python3
"""
injector.py – inject synthetic errors into TGFD Uniform Blocks while
guaranteeing that every snapshot keeps exactly the same total 
Supports
• uniform, normal (μ/σ) or zipfian (α) temporal distributions
• any number of error labels (--num-errors N)
• input that may already contain previous errors
Example
-------
python injector.py 0.30 uniform --input inject.txt --output ub_injected.txt
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
try:
    from scipy.stats import norm as _scipy_norm  
    _PPF = _scipy_norm.ppf
except Exception:  
    from math import sqrt, erfinv  
    _SQRT2 = sqrt(2.0)
    def _PPF(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:  
        return mu + sigma * _SQRT2 * erfinv(2 * p - 1)
Snapshot = str
CounterMap = Dict[Snapshot, Counter]
def _to_counter_map(raw: Dict[str, Dict[str, int]]) -> CounterMap:
    return {snap: Counter(cnts) for snap, cnts in raw.items()}
def _deep_counter_clone(src: CounterMap) -> CounterMap:
    return {snap: Counter(c) for snap, c in src.items()}
_inv_uniform = lambda: (lambda p: p)
def _inv_normal(mu: float, sigma: float):
    return lambda p: max(0.0, min(1.0, _PPF(p, mu, sigma)))
def _inv_zipf(alpha: float):
    if alpha <= 1.0:
        raise ValueError("Zipf α must be > 1.")
    return lambda p: p ** (1.0 / (1.0 - alpha))
def build_occ_index(ubs: List[dict]) -> Dict[Snapshot, List[Tuple[int, str]]]:
    idx: Dict[Snapshot, List[Tuple[int, str]]] = defaultdict(list)
    for ub_idx, ub in enumerate(ubs):
        for snap, counter in ub["values_by_snapshot_injected"].items():
            for val, cnt in counter.items():
                if str(val).startswith("error"):
                    continue
                idx[snap].extend([(ub_idx, val)] * cnt)
    return idx
def allocate_errors(percentage: float, snap_freq: Counter, inv_cdf) -> Counter:
    total = sum(snap_freq.values())
    wanted = max(1, math.floor(total * percentage))
    snaps = sorted(snap_freq.keys(), key=int)
    cum, run = [], 0
    for s in snaps:
        run += snap_freq[s]
        cum.append(run)
    out = Counter()
    for _ in range(wanted):
        frac = inv_cdf(random.random())
        global_idx = int(frac * (total - 1))
        lo, hi = 0, len(cum) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if cum[mid] > global_idx:
                hi = mid - 1
            else:
                lo = mid + 1
        out[snaps[lo]] += 1
    return out
def split_into_labels(err_counts: Counter, num_labels: int) -> Dict[Snapshot, Counter]:
    plan: Dict[Snapshot, Counter] = {}
    w = [1.0 / r for r in range(1, num_labels + 1)]
    denom = sum(w)
    for snap, n in err_counts.items():
        frac = [n * wi / denom for wi in w]
        alloc = [math.floor(x) for x in frac]
        rem = n - sum(alloc)
        if rem:
            order = sorted(
                range(num_labels),
                key=lambda i: (frac[i] - alloc[i]),
                reverse=True,
            )
            for i in order[:rem]:
                alloc[i] += 1
        plan[snap] = Counter({f"error{i+1}": c for i, c in enumerate(alloc) if c})
    return plan
def inject_errors(
    ubs: List[dict],
    occ_index: Dict[Snapshot, List[Tuple[int, str]]],
    plan: Dict[Snapshot, Counter],
) -> None:
    rng = random.Random()
    for snap, want in plan.items():
        pool = occ_index.get(snap, [])
        needed = min(sum(want.values()), len(pool))
        if needed == 0:
            continue
        rng.shuffle(pool)
        cursor = 0
        for label, cnt in want.items():
            for _ in range(cnt):
                if cursor >= needed:
                    break
                ub_idx, val = pool[cursor]
                cursor += 1
                counter = ubs[ub_idx]["values_by_snapshot_injected"][snap]
                if not isinstance(counter, Counter):
                    counter = Counter(counter)
                    ubs[ub_idx]["values_by_snapshot_injected"][snap] = counter
                counter[val] -= 1
                if counter[val] == 0:
                    del counter[val]
                counter[label] += 1
def _rebalance_snapshot(orig: Counter, inj: Counter) -> None:
    diff = sum(orig.values()) - sum(inj.values())
    if diff == 0:
        return
    if diff > 0:                                       
        pick = next((k for k in inj if k.startswith("error")), None)
        pick = pick or (next(iter(orig)) if orig else None)
        if pick is None:
            return                                     
        inj[pick] += diff
    else:                                              
        diff = -diff
        keys = sorted(inj, key=lambda k: (not k.startswith("error"), -inj[k]))
        for k in keys:
            if diff == 0:
                break
            take = min(inj[k], diff)
            inj[k] -= take
            diff -= take
            if inj[k] == 0:
                del inj[k]
def verify_and_fix(ubs: List[dict]) -> None:
    for ub in ubs:
        for snap, orig_cnts in ub["values_by_snapshot"].items():
            inj_cnts = ub["values_by_snapshot_injected"].setdefault(
                snap, Counter()
            )
            if not isinstance(inj_cnts, Counter):
                inj_cnts = Counter(inj_cnts)
                ub["values_by_snapshot_injected"][snap] = inj_cnts
            _rebalance_snapshot(orig_cnts, inj_cnts)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("percentage", type=float, help="Fraction of matches to corrupt.")
    ap.add_argument("distribution", choices=("uniform", "normal", "zipfian"))
    ap.add_argument("--num-errors", type=int, default=1, dest="num_labels")
    ap.add_argument("--input", default="inject.txt")
    ap.add_argument("--output", default="ub_injected.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=2.0)
    args = ap.parse_args()
    random.seed(args.seed)
    try:
        with Path(args.input).open() as f:
            ubs = [json.loads(line) for line in f if line.strip()]
    except Exception as exc:
        sys.exit(f"Cannot read {args.input}: {exc}")
    if not ubs:
        sys.exit("Input contained no UBs – aborting.")
    for ub in ubs:
        ub["values_by_snapshot"] = _to_counter_map(ub.get("values_by_snapshot", {}))
        inj = ub.get("values_by_snapshot_injected")
        ub["values_by_snapshot_injected"] = (
            _to_counter_map(inj) if inj else _deep_counter_clone(ub["values_by_snapshot"])
        )
    snap_freq = Counter()
    for ub in ubs:
        for s, c in ub["values_by_snapshot"].items():
            snap_freq[s] += sum(c.values())
    if sum(snap_freq.values()) == 0 or args.percentage <= 0:
        Path(args.output).write_text("\n".join(json.dumps(ub) for ub in ubs))
        print("Nothing to inject.")
        return
    inv_cdf = (
        _inv_uniform()
        if args.distribution == "uniform"
        else _inv_normal(args.mu, args.sigma)
        if args.distribution == "normal"
        else _inv_zipf(args.alpha)
    )
    err_counts = allocate_errors(args.percentage, snap_freq, inv_cdf)
    plan = split_into_labels(err_counts, max(1, args.num_labels))
    occ_index = build_occ_index(ubs)
    inject_errors(ubs, occ_index, plan)
    verify_and_fix(ubs)
    with Path(args.output).open("w") as out_f:
        out_f.writelines(json.dumps(ub) + "\n" for ub in ubs)
    print(
        f"Injected {sum(err_counts.values())} errors over {len(err_counts)} snapshots "
        f"→ {args.output}"
    )
if __name__ == "__main__":
    main()
