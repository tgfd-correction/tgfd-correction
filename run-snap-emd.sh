#!/usr/bin/env bash
set -euo pipefail
SNAP_DIRS=(snap10 snap40 snap60)   
JOBS=8                             
CORRECT="./parallel_correction.py"
[[ -f "$CORRECT" ]] || { echo "ERROR: $CORRECT not found"; exit 1; }
run_one () {
    local INFILE="$1" UBDIR="$2"
    local BASE
    BASE=$(basename "$INFILE")
    BASE=${BASE
    BASE=${BASE%_injected.txt}
    local OUTDIR="${UBDIR}/emd"
    mkdir -p "$OUTDIR"
    python "$CORRECT" \
        --method emd \
        --lambda 0.5 \
        --input  "$INFILE" \
        --output "${OUTDIR}/result_${BASE}.txt"
}
export CORRECT ; export -f run_one
for SNAP in "${SNAP_DIRS[@]}"; do
    UBDIR="${SNAP}/ub_30_uniform"
    echo "▶ Processing ${UBDIR}"
    [[ -d "$UBDIR" ]] || { echo "  (skip; folder missing)"; continue; }
    FILES=("${UBDIR}"/ub_part_*_injected.txt)
    [[ -e "${FILES[0]}" ]] || { echo "  (no injected parts found)"; continue; }
    if command -v parallel >/dev/null 2>&1; then
        parallel --jobs "$JOBS" run_one {1} {2} ::: "${FILES[@]}" :::+ "${UBDIR}"
    else
        for f in "${FILES[@]}"; do
            run_one "$f" "$UBDIR" &
            while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 0.1; done
        done
        wait
    fi
done
echo "✅  EMD correction completed for snap10, snap40, snap60."
