#!/usr/bin/env python3
"""
parallel_correction.py - Perform correction of Uniform Blocks (UBs) in parallel.
This script reads UBs in NDJSON format from an input file (default "ub_injected.txt"),
applies a correction method in parallel using pandarallel, and writes the corrected UBs to an output file (default "result.txt").
Two correction methods are supported:
 1. majority_vote: Unify all entries in each UB to the most frequent label found in that UB.
 2. emd: Use Earth Mover's Distance (EMD) to compare the distribution of each candidate label's occurrences across snapshots to the ideal distribution from the clean data. Select the label with minimum score = EMD + λ * (1 - frequency(label)/total_matches), and unify the UB to that label.
Usage:
    python parallel_correction.py --method [majority_vote|emd] [--lambda <value>] [--input <file>] [--output <file>]
The --lambda parameter is only used for the 'emd' method (default 0.5).
"""
import json
import argparse
from collections import Counter
import pandas as pd
from pandarallel import pandarallel
import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
try:
    from scipy.stats import wasserstein_distance
except ImportError:
    import math
    def wasserstein_distance(u_values, v_values, u_weights=None, v_weights=None):
        if u_weights is None:
            u_weights = [1.0/len(u_values)] * len(u_values)
        if v_weights is None:
            v_weights = [1.0/len(v_values)] * len(v_values)
        combined = sorted(set(u_values) | set(v_values))
        u_w_dict = {val: 0.0 for val in combined}
        v_w_dict = {val: 0.0 for val in combined}
        for val, w in zip(u_values, u_weights):
            u_w_dict[val] += w
        for val, w in zip(v_values, v_weights):
            v_w_dict[val] += w
        cdf_diff_area = 0.0
        cumu = 0.0
        cumv = 0.0
        prev_val = None
        for val in combined:
            if prev_val is not None:
                diff = abs(cumu - cumv)
                cdf_diff_area += diff * (val - prev_val)
            cumu += u_w_dict[val]
            cumv += v_w_dict[val]
            prev_val = val
        return cdf_diff_area
def load_ubs(filename):
    """Load UBs from a NDJSON file. Returns a list of UB dictionaries."""
    ubs = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ubs.append(json.loads(line))
    return ubs
def save_results(ubs, filename):
    """Save a list of UB dictionaries to a NDJSON file."""
    with open(filename, "w") as f:
        for ub in ubs:
            f.write(json.dumps(ub) + "\n")
    print(f"Saved corrected UBs to {filename}")
def majority__vote_correction(ub):
    """Correct a UB by majority vote. Unifies all entries to the most frequent label in the UB."""
    overall_counter = Counter()
    for snap, counter in ub.get("values_by_snapshot_injected", {}).items():
        overall_counter.update(counter)
    if not overall_counter:
        return ub  
    return ub
def majority_vote_correction(ub):
    """
    Correct a UB by majority vote.
    Unifies all entries to the most frequent label in the UB.
    """
    overall_counter = Counter()
    for snap, counter in ub.get("values_by_snapshot_injected", {}).items():
        overall_counter.update(counter)
    if not overall_counter:
        return ub  
    max_count = max(overall_counter.values())
    tied_labels = [label for label, cnt in overall_counter.items() if cnt == max_count]
    if len(tied_labels) >= 2:
        dominant_label = tied_labels[1]
    else:
        dominant_label = tied_labels[0]
    for snap, counter in ub.get("values_by_snapshot_injected", {}).items():
        total_count = sum(counter.values())
        ub["values_by_snapshot_injected"][snap] = {dominant_label: total_count}
    return ub
def emd_correction(ub, lam=0.5):
    """Correct a UB using EMD-based method. Unifies all entries to the label with minimum EMD-based score."""
    values_by_snapshot = ub.get("values_by_snapshot", {})
    if not values_by_snapshot:
        return ub  
    snapshot_indices = sorted(int(s) for s in values_by_snapshot.keys())
    ideal_counts = []
    for snap in snapshot_indices:
        snap_str = str(snap)
        counts = values_by_snapshot.get(snap_str, {})
        ideal_counts.append(sum(counts.values()) if isinstance(counts, dict) else 0)
    ideal_total = sum(ideal_counts)
    if ideal_total == 0:
        return ub  
    ideal_dist = [c / ideal_total for c in ideal_counts]
    positions = [float(s) for s in snapshot_indices]
    total_matches = sum(sum(cnt.values()) for cnt in ub.get("values_by_snapshot_injected", {}).values())
    if total_matches == 0:
        return ub
    candidates = set()
    for snap_counter in ub.get("values_by_snapshot_injected", {}).values():
        candidates.update(snap_counter.keys())
    best_label = None
    best_score = float("inf")
    for label in candidates:
        label_counts = []
        label_total = 0
        for snap in snapshot_indices:
            snap_str = str(snap)
            count = 0
            if snap_str in ub["values_by_snapshot_injected"]:
                count = ub["values_by_snapshot_injected"][snap_str].get(label, 0)
            label_counts.append(count)
            label_total += count
        if label_total == 0:
            continue  
        label_dist = [c / label_total for c in label_counts]
        try:
            dist = wasserstein_distance(positions, positions, u_weights=label_dist, v_weights=ideal_dist)
        except Exception:
            dist = sum(abs(ld - id_) for ld, id_ in zip(label_dist, ideal_dist))
        freq_proportion = label_total / total_matches
        score = dist + lam * (1 - freq_proportion)
        if score < best_score:
            best_score = score
            best_label = label
    if best_label is None:
        return ub  
    for snap, counter in ub.get("values_by_snapshot_injected", {}).items():
        total_count = sum(counter.values())
        ub["values_by_snapshot_injected"][snap] = {best_label: total_count}
    return ub
def impute_linear_correction(ub):
    """Correct a UB using multiple imputation via Linear Regression (time as feature)."""
    total_matches = sum(sum(cnt.values()) for cnt in ub.get("values_by_snapshot_injected", {}).values())
    if total_matches == 0:
        return ub
    candidates = set()
    for snap_counter in ub.get("values_by_snapshot_injected", {}).values():
        candidates.update(snap_counter.keys())
    best_label = None
    best_score = float("inf")
    snapshot_indices = sorted(int(s) for s in ub.get("values_by_snapshot_injected", {}).keys())
    total_counts = [sum(ub["values_by_snapshot_injected"].get(str(snap), {}).values()) for snap in snapshot_indices]
    times = np.array(snapshot_indices, dtype=float).reshape(-1, 1)
    for label in candidates:
        label_counts = []
        label_total = 0
        for snap in snapshot_indices:
            snap_str = str(snap)
            count = ub["values_by_snapshot_injected"].get(snap_str, {}).get(label, 0)
            label_counts.append(count)
            label_total += count
        if label_total == 0:
            continue
        label_counts_arr = np.array(label_counts, dtype=float)
        total_counts_arr = np.array(total_counts, dtype=float)
        observed_mask = np.array([(cnt > 0 or total_counts_arr[i] == 0) for i, cnt in enumerate(label_counts_arr)], dtype=bool)
        if not observed_mask.any():
            continue
        x_obs = times[observed_mask]
        y_obs = label_counts_arr[observed_mask]
        model = LinearRegression()
        try:
            model.fit(x_obs, y_obs)
        except Exception:
            continue
        y_pred = model.predict(times)
        sse = float(np.sum((y_pred - total_counts_arr) ** 2))
        if sse < best_score:
            best_score = sse
            best_label = label
    if best_label is None:
        return ub
    for snap, counter in ub.get("values_by_snapshot_injected", {}).items():
        total_count = sum(counter.values())
        ub["values_by_snapshot_injected"][snap] = {best_label: total_count}
    return ub
def impute_em_correction(ub):
    """Correct a UB using an EM-inspired iterative imputation (BayesianRidge)."""
    total_matches = sum(sum(cnt.values()) for cnt in ub.get("values_by_snapshot_injected", {}).values())
    if total_matches == 0:
        return ub
    candidates = set()
    for snap_counter in ub.get("values_by_snapshot_injected", {}).values():
        candidates.update(snap_counter.keys())
    best_label = None
    best_score = float("inf")
    snapshot_indices = sorted(int(s) for s in ub.get("values_by_snapshot_injected", {}).keys())
    total_counts = [sum(ub["values_by_snapshot_injected"].get(str(snap), {}).values()) for snap in snapshot_indices]
    times = np.array(snapshot_indices, dtype=float)
    for label in candidates:
        label_counts = []
        label_total = 0
        for snap in snapshot_indices:
            snap_str = str(snap)
            count = ub["values_by_snapshot_injected"].get(snap_str, {}).get(label, 0)
            label_counts.append(float(count))
            label_total += count
        if label_total == 0:
            continue
        data = []
        for t, count, total in zip(times, label_counts, total_counts):
            if total > 0 and count == 0.0:
                data.append([t, np.nan])
            else:
                data.append([t, float(count)])
        data = np.array(data, dtype=float)
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, sample_posterior=False, random_state=0)
        try:
            imputed_data = imp.fit_transform(data)
        except Exception:
            continue
        y_imputed = imputed_data[:, 1]
        total_arr = np.array(total_counts, dtype=float)
        sse = float(np.sum((y_imputed - total_arr) ** 2))
        if sse < best_score:
            best_score = sse
            best_label = label
    if best_label is None:
        return ub
    for snap, counter in ub.get("values_by_snapshot_injected", {}).items():
        total_count = sum(counter.values())
        ub["values_by_snapshot_injected"][snap] = {best_label: total_count}
    return ub
def impute_pmm_correction(ub):
    """Correct a UB using Predictive Mean Matching via k-NN regression."""
    total_matches = sum(sum(cnt.values()) for cnt in ub.get("values_by_snapshot_injected", {}).values())
    if total_matches == 0:
        return ub
    candidates = set()
    for snap_counter in ub.get("values_by_snapshot_injected", {}).values():
        candidates.update(snap_counter.keys())
    best_label = None
    best_score = float("inf")
    snapshot_indices = sorted(int(s) for s in ub.get("values_by_snapshot_injected", {}).keys())
    total_counts = [sum(ub["values_by_snapshot_injected"].get(str(snap), {}).values()) for snap in snapshot_indices]
    times = np.array(snapshot_indices, dtype=float).reshape(-1, 1)
    for label in candidates:
        label_counts = []
        label_total = 0
        for snap in snapshot_indices:
            snap_str = str(snap)
            count = ub["values_by_snapshot_injected"].get(snap_str, {}).get(label, 0)
            label_counts.append(float(count))
            label_total += count
        if label_total == 0:
            continue
        label_counts_arr = np.array(label_counts, dtype=float)
        total_counts_arr = np.array(total_counts, dtype=float)
        observed_mask = np.array([(cnt > 0 or total_counts_arr[i] == 0) for i, cnt in enumerate(label_counts_arr)], dtype=bool)
        if not observed_mask.any():
            continue
        x_obs = times[observed_mask]
        y_obs = label_counts_arr[observed_mask]
        n_neighbors = min(3, len(y_obs))
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        try:
            model.fit(x_obs, y_obs)
        except Exception:
            continue
        y_pred = model.predict(times)
        sse = float(np.sum((y_pred - total_counts_arr) ** 2))
        if sse < best_score:
            best_score = sse
            best_label = label
    if best_label is None:
        return ub
    for snap, counter in ub.get("values_by_snapshot_injected", {}).items():
        total_count = sum(counter.values())
        ub["values_by_snapshot_injected"][snap] = {best_label: total_count}
    return ub
def emd_topk_correction(ub, *, lam=0.5, k=0):
    """
    EMD correction but only evaluates the k most frequent candidate labels
    in the injected UB.  If k<=0 or k exceeds the number of candidates,
    behaviour is identical to standard EMD.
    """
    if k and k < 0:
        raise ValueError("k must be non-negative")
    values_by_snapshot = ub.get("values_by_snapshot", {})
    if not values_by_snapshot:
        return ub
    snap_idx = sorted(int(s) for s in values_by_snapshot.keys())
    ideal_counts = [sum(values_by_snapshot.get(str(s), {}).values()) for s in snap_idx]
    ideal_total = sum(ideal_counts)
    if ideal_total == 0:
        return ub
    ideal_dist = [c / ideal_total for c in ideal_counts]
    positions = [float(s) for s in snap_idx]
    total_matches = sum(
        sum(cnt.values()) for cnt in ub.get("values_by_snapshot_injected", {}).values()
    )
    if total_matches == 0:
        return ub
    freq = Counter()
    for snap_counter in ub.get("values_by_snapshot_injected", {}).values():
        freq.update(snap_counter)
    candidates = [lbl for lbl, _ in freq.most_common(k or None)]
    if not candidates:
        return ub
    best_label, best_score = None, float("inf")
    for label in candidates:
        label_counts = []
        label_total = 0
        for s in snap_idx:
            snap_str = str(s)
            cnt = ub["values_by_snapshot_injected"].get(snap_str, {}).get(label, 0)
            label_counts.append(cnt)
            label_total += cnt
        if label_total == 0:
            continue
        label_dist = [c / label_total for c in label_counts]
        try:
            dist = wasserstein_distance(
                positions, positions, u_weights=label_dist, v_weights=ideal_dist
            )
        except Exception:
            dist = sum(abs(ld - id_) for ld, id_ in zip(label_dist, ideal_dist))
        freq_prop = label_total / total_matches
        score = dist + lam * (1 - freq_prop)
        if score < best_score:
            best_label, best_score = label, score
    if best_label is None:
        return ub
    for snap, cnt in ub.get("values_by_snapshot_injected", {}).items():
        total = sum(cnt.values())
        ub["values_by_snapshot_injected"][snap] = {best_label: total}
    return ub
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Correct UBs in parallel using pandarallel."
    )
    parser.add_argument(
        "--method",
        choices=[
            "majority_vote",
            "emd",
            "emd_topk",          
            "impute_linear",
            "impute_em",
            "impute_pmm",
        ],
        default="majority_vote",
    )
    parser.add_argument(
        "--lambda",
        dest="lam",
        type=float,
        default=0.5,
        help="Lambda for EMD-based methods.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Top-k budget for emd_topk (ignored by other methods).",
    )
    parser.add_argument("--input", "-i", default="ub_injected.txt")
    parser.add_argument("--output", "-o", default="result.txt")
    return parser.parse_args()
def main():
    args = _parse_args()
    pandarallel.initialize(progress_bar=True)
    ubs = load_ubs(args.input)
    if args.method == "majority_vote":
        correct_func = majority_vote_correction
    elif args.method == "emd":
        correct_func = lambda ub: emd_correction(ub, lam=args.lam)
    elif args.method == "emd_topk":
        correct_func = lambda ub: emd_topk_correction(ub, lam=args.lam, k=args.k)
    elif args.method == "impute_linear":
        correct_func = impute_linear_correction
    elif args.method == "impute_em":
        correct_func = impute_em_correction
    else:  
        correct_func = impute_pmm_correction
    out_series = pd.Series(ubs).parallel_apply(correct_func)
    save_results(list(out_series), args.output)
import atexit, time, os, sys  
__pc_start_time = time.perf_counter()
def _write_runtime_log() -> None:
    elapsed = time.perf_counter() - __pc_start_time
    out_path = "result.txt"
    for flag in ("--output", "-o"):
        if flag in sys.argv:
            try:
                out_path = sys.argv[sys.argv.index(flag) + 1]
            except IndexError:
                pass
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "runtime_results.txt"), "a") as fh:
        fh.write(f"{os.path.basename(out_path)}\t{elapsed:.4f}\n")
atexit.register(_write_runtime_log)
if __name__ == "__main__":
    main()
