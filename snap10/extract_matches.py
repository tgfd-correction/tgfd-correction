#!/usr/bin/env python3
"""
extract_matches.py
Generate Uniform Blocks (UBs) for the TGFD
    (actor_name , actor_uri)  →  movie_name        with  Δ = (0,1)
i.e. for every actor (identified by BOTH name and URI) there may be
at most one distinct movie across two consecutive snapshots.
The script:
  • Scans snapshot*.nt files.
  • Keeps only triples whose predicate is  <http://xmlns.com/foaf/0.1/actor_of>
  • Derives keys and values:
        LHS  = (actor_name, actor_uri)
        RHS  = movie_name
  • Builds UBs using the Δ‐connected‑components algorithm already.
Usage (unchanged from the original):
    python extract_matches.py -n 4 --delta 0 1
"""
import argparse
import glob
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from multiprocessing import Process
from urllib.parse import unquote
from ub_model import UniformBlock
_TRIPLE_RX = re.compile(r'^<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s+\.$')
ACTOR_OF_URI = 'http://xmlns.com/foaf/0.1/actor_of'
def _actor_name_from_uri(uri: str) -> str:
    """
    Given   http://imdb.org/actor/Rhodes%2C%20Donnelly
    return  'Rhodes, Donnelly'
    """
    return unquote(uri.rsplit('/', 1)[-1])
def _movie_name_from_uri(uri: str) -> str:
    """
    Given   http://imdb.org/movie/Two%20Faces%20West
    return  'Two Faces West'
    """
    return unquote(uri.rsplit('/', 1)[-1])
def get_matches(directory: str = ".", prefix: str = "imdb-"):
    """
    Yields tuples (actor_uri, actor_name, movie_name, snapshot_index)
    for every triple   <actor_uri> <actor_of> <movie_uri> .
    """
    pattern = os.path.join(directory, f"{prefix}*.nt")
    paths = glob.glob(pattern)
    if not paths:
        return []
    paths.sort(key=lambda p: int(re.search(rf"{prefix}(\d+)\.nt$", os.path.basename(p)).group(1)))
    out = []
    for path in paths:
        snap_idx = int(re.search(rf"{prefix}(\d+)\.nt$", os.path.basename(path)).group(1))
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                m = _TRIPLE_RX.match(line)
                if not m:
                    continue
                subj_uri, pred_uri, obj_uri = m.groups()
                if pred_uri != ACTOR_OF_URI:
                    continue
                actor_uri = f"<{subj_uri}>"
                actor_name = _actor_name_from_uri(subj_uri)
                movie_name = _movie_name_from_uri(obj_uri)
                out.append((actor_uri, actor_name, movie_name, snap_idx))
    return out
def process_partition(part_idx: int, part_data: dict, delta_a: int, delta_b: int):
    ubs = []
    for (actor_uri, actor_name), snaps_dict in part_data.items():
        snapshots = sorted(snaps_dict.keys())
        if not snapshots:
            continue
        adj = {s: [] for s in snapshots}
        n = len(snapshots)
        for i in range(n):
            for j in range(i + 1, n):
                diff = snapshots[j] - snapshots[i]
                if diff < delta_a:
                    continue
                if diff > delta_b:
                    break
                si, sj = snapshots[i], snapshots[j]
                adj[si].append(sj)
                adj[sj].append(si)
        visited = set()
        for snap in snapshots:
            if snap in visited:
                continue
            stack = [snap]
            component = []
            visited.add(snap)
            while stack:
                node = stack.pop()
                component.append(node)
                for nei in adj[node]:
                    if nei not in visited:
                        visited.add(nei)
                        stack.append(nei)
            component.sort()
            ub_vals = {s: snaps_dict[s] for s in component}
            ubs.append(UniformBlock(subject=actor_uri,
                                    hours=actor_name,          
                                    values_by_snapshot=ub_vals))
    ubs.sort(key=lambda ub: (ub.subject, min(ub.values_by_snapshot)))
    with open(f"ub_part_{part_idx}.txt", "w") as fout:
        for ub in ubs:
            fout.write(json.dumps(ub.to_dict()) + "\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract TGFD matches (actor_of) and build Uniform Blocks."
    )
    parser.add_argument("-n", "--num-partitions", type=int, default=16,
                        help="Number of worker processes (partitions).")
    parser.add_argument("--delta", type=int, nargs=2, metavar=("A", "B"),
                        default=(0, 1),
                        help="Δ range (inclusive) for temporal connectivity.")
    parser.add_argument("--snapshot-dir", default=".",
                        help="Directory that contains snapshot*.nt files.")
    parser.add_argument("--prefix", default="imdb-",
                        help="Filename prefix for snapshots (default 'imdb-').")
    args = parser.parse_args()
    delta_a, delta_b = args.delta
    matches = get_matches(directory=args.snapshot_dir, prefix=args.prefix)
    group_map = defaultdict(lambda: defaultdict(Counter))
    for actor_uri, actor_name, movie_name, snap in matches:
        group_map[(actor_uri, actor_name)][snap][movie_name] += 1
    partitions = {i: [] for i in range(args.num_partitions)}
    for key in group_map:
        key_str = f"{key[0]}|{key[1]}"
        part_idx = int(hashlib.md5(key_str.encode()).hexdigest(), 16) % args.num_partitions
        partitions[part_idx].append(key)
    procs = []
    for idx in range(args.num_partitions):
        part_data = {k: group_map[k] for k in partitions[idx]}
        p = Process(target=process_partition, args=(idx, part_data, delta_a, delta_b))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
