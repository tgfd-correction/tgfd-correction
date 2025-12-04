#!/usr/bin/env python3
"""
extract_matches.py  ──  TGFD extractor for   Person ▸(role)▸ Movie
TGFD enforced
-------------
Within any 3‑snapshot window (distance 1–3), the same person *p*
(identified by both URI and canonical name) linked to the same movie *m*
must always carry **exactly one role label** drawn from
    { actor_of , actress_of , director_of }.
Formally:
    X  →  Y     with  Δ = (1,3)
    X = { p.name , p.uri , m.name }
    Y = { r.label }
Outputs are Uniform Blocks partitioned by (p.uri , p.name , m.name); each
snapshot’s counter tallies the set of role labels observed.
-----------------------------------------------------------------------
Command‑line example
-----------------------------------------------------------------------
    python extract_matches.py \
        --prefix imdb-         \   
        --snapshot-dir .       \
        --delta 1 3            \   
        -n 8                       
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
from snap25.ub_model import UniformBlock
_TRIPLE_RX = re.compile(
    r'^<([^>]+)>\s+'           
    r'<([^>]+)>\s+'            
    r'<([^>]+)>\s+\.$'         
)
ROLE_URIS = {
    'http://xmlns.com/foaf/0.1/actor_of':    'actor_of',
    'http://xmlns.com/foaf/0.1/actress_of':  'actress_of',
    'http://xmlns.com/foaf/0.1/director_of': 'director_of',
}
def _person_name_from_uri(uri: str) -> str:
    """Decode 'http://imdb.org/actor/Smith%2C%20John' → 'Smith, John'."""
    return unquote(uri.rsplit('/', 1)[-1])
def _movie_name_from_uri(uri: str) -> str:
    """Decode 'http://imdb.org/movie/Foo%20Bar' → 'Foo Bar'."""
    return unquote(uri.rsplit('/', 1)[-1])
def get_matches(directory: str = ".", prefix: str = "imdb-"):
    """
    Yield tuples (person_uri, person_name, movie_name, role_label, snapshot_idx)
    for triples  <person_uri> <role_pred> <movie_uri> .
    """
    pattern = os.path.join(directory, f"{prefix}*.nt")
    files   = glob.glob(pattern)
    if not files:
        return []
    files.sort(key=lambda p: int(re.search(rf"{prefix}(\d+)\.nt$", os.path.basename(p)).group(1)))
    out = []
    for path in files:
        snap_idx = int(re.search(rf"{prefix}(\d+)\.nt$", os.path.basename(path)).group(1))
        with open(path, "r") as fh:
            for line in fh:
                m = _TRIPLE_RX.match(line.strip())
                if not m:
                    continue
                subj_uri, pred_uri, obj_uri = m.groups()
                if pred_uri not in ROLE_URIS:
                    continue
                person_uri  = f"<{subj_uri}>"
                person_name = _person_name_from_uri(subj_uri)
                movie_name  = _movie_name_from_uri(obj_uri)
                role_label  = ROLE_URIS[pred_uri]
                out.append((person_uri, person_name, movie_name, role_label, snap_idx))
    return out
def process_partition(idx: int, part_data: dict, dmin: int, dmax: int):
    ubs = []
    for key, snaps in part_data.items():           
        snapshots = sorted(snaps.keys())
        if not snapshots:
            continue
        adj = {s: [] for s in snapshots}
        for i, si in enumerate(snapshots):
            for sj in snapshots[i+1:]:
                diff = sj - si
                if diff < dmin:  continue
                if diff > dmax:  break
                adj[si].append(sj);  adj[sj].append(si)
        visited = set()
        for s in snapshots:
            if s in visited:
                continue
            comp, stack = [], [s]
            visited.add(s)
            while stack:
                v = stack.pop();  comp.append(v)
                for w in adj[v]:
                    if w not in visited:
                        visited.add(w);  stack.append(w)
            comp.sort()
            ub_vals = {s: snaps[s] for s in comp}
            p_uri, p_name, m_name = key
            subject = f"{p_uri}|{m_name}"
            ubs.append(UniformBlock(subject=subject,
                                    hours=p_name,
                                    values_by_snapshot=ub_vals))
    ubs.sort(key=lambda ub: (ub.subject, min(ub.values_by_snapshot)))
    with open(f"ub_part_roles_{idx}.txt", "w") as f:
        for ub in ubs:
            f.write(json.dumps(ub.to_dict()) + "\n")
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract TGFD (role labels) and build UBs")
    ap.add_argument("-n", "--num-partitions", type=int, default=16)
    ap.add_argument("--delta", type=int, nargs=2, metavar=("A","B"), default=(1,3),
                    help="Δ range (inclusive) for temporal connectivity (default 1 3)")
    ap.add_argument("--snapshot-dir", default=".")
    ap.add_argument("--prefix", default="imdb-")
    args = ap.parse_args()
    dmin, dmax = args.delta
    matches = get_matches(args.snapshot_dir, args.prefix)
    groups = defaultdict(lambda: defaultdict(Counter))
    for p_uri, p_name, m_name, role, snap in matches:
        groups[(p_uri, p_name, m_name)][snap][role] += 1
    buckets = {i: [] for i in range(args.num_partitions)}
    for key in groups:
        h = int(hashlib.md5("|".join(key).encode()).hexdigest(),16) % args.num_partitions
        buckets[h].append(key)
    procs=[]
    for idx in range(args.num_partitions):
        data={k:groups[k] for k in buckets[idx]}
        p=Process(target=process_partition,args=(idx,data,dmin,dmax))
        p.start(); procs.append(p)
    for p in procs:
        p.join()
