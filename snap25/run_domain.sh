#!/usr/bin/env bash
set -euo pipefail
KS=(2 3 4)         
LAMBDA=0.5
JOBS=8             
CORRECT="./parallel_correction.py"
[[ -f "$CORRECT" ]] || { echo "Cannot find $CORRECT"; exit 1; }
run_one () {
    local INFILE="$1" K="$2" UBDIR="$3"
    local PART=${INFILE
    local SUBDIR="${UBDIR}/emd_k${K}"
    mkdir -p "$SUBDIR"
    python "$CORRECT" \
        --method emd_topk \
        --lambda "$LAMBDA" \
        --k "$K" \
        --input "$INFILE" \
        --output "${SUBDIR}/result_${PART}.txt"
}
export CORRECT ; export LAMBDA
export -f run_one
find . -maxdepth 1 -type d -name 'ub_*_*' | while read -r UBDIR; do
    echo "▶ ${UBDIR
    FILES=("${UBDIR}"/ub_part_*_injected.txt)
    [[ -e "${FILES[0]}" ]] || continue
    if command -v parallel >/dev/null 2>&1; then
        parallel --jobs "$JOBS" run_one {2} {1} {3} \
                 ::: "${KS[@]}" ::: "${FILES[@]}" :::+ "${UBDIR}"
    else
        for k in "${KS[@]}"; do
            for f in "${FILES[@]}"; do
                run_one "$f" "$k" "$UBDIR" &
                while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 0.1; done
            done
        done
        wait
    fi
done
echo "✅  emd_topk completed for k = ${KS[*]}"
