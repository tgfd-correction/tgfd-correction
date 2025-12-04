#!/usr/bin/env bash
set -euo pipefail
LAMBDAS=(0 0.25 0.5 1)     
JOBS=8                     
CORRECT="./parallel_correction.py"
[[ -f "$CORRECT" ]] || { echo "Cannot find $CORRECT"; exit 1; }
run_emd () {
    local INFILE="$1" LAMBDA="$2" UBDIR="$3"
    local PART
    PART=$(basename "$INFILE")
    PART=${PART
    PART=${PART%_injected.txt}
    local SUBDIR="${UBDIR}/emd_lambda_${LAMBDA}"
    mkdir -p "$SUBDIR"
    python "$CORRECT" \
        --method emd \
        --lambda "$LAMBDA" \
        --input  "$INFILE" \
        --output "${SUBDIR}/result_${PART}.txt"
}
export CORRECT
export -f run_emd
find . -maxdepth 1 -type d -name 'ub_*_*' | while read -r UBDIR; do
    echo "▶ Processing ${UBDIR
    FILES=("${UBDIR}"/ub_part_*_injected.txt)
    [[ -e "${FILES[0]}" ]] || { echo "  (no injected parts found)"; continue; }
    if command -v parallel >/dev/null 2>&1; then
        parallel --jobs "$JOBS" run_emd {2} {1} {3} \
                 ::: "${LAMBDAS[@]}" \
                 ::: "${FILES[@]}"  \
                 :::+ "${UBDIR}"
    else
        for lam in "${LAMBDAS[@]}"; do
            for infile in "${FILES[@]}"; do
                run_emd "$infile" "$lam" "$UBDIR" &
                while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 0.1; done
            done
        done
        wait
    fi
done
echo "✅  EMD corrections complete for λ = ${LAMBDAS[*]}"
