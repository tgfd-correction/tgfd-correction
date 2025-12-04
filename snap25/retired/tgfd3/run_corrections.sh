#!/usr/bin/env bash
set -euo pipefail
METHODS=(emd)
JOBS=8                            
CORRECT="./parallel_correction.py"
[[ -f "$CORRECT" ]] || { echo "Cannot find $CORRECT"; exit 1; }
correct_one() {
  local METHOD="$1"   
  local INFILE="$2"   
  local OUTDIR="$3"   
  local PART_IDX
  PART_IDX=$(basename "$INFILE")
  PART_IDX=${PART_IDX
  PART_IDX=${PART_IDX%_injected.txt}
  mkdir -p "$OUTDIR"
  python "$CORRECT" \
         --method "$METHOD" \
         --input  "$INFILE" \
         --output "${OUTDIR}/result_${PART_IDX}.txt"
}
export CORRECT
export -f correct_one
find . -maxdepth 1 -type d -name 'ub_*_*' | while read -r UBDIR; do
  echo "▶ Processing ${UBDIR
  for METHOD in "${METHODS[@]}"; do
    SUBDIR="${UBDIR}/${METHOD}"
    mkdir -p "$SUBDIR"
  done
  FILES=("${UBDIR}"/ub_part_*_injected.txt)
  [[ -e "${FILES[0]}" ]] || { echo "  (no injected parts found)"; continue; }
  if command -v parallel >/dev/null 2>&1; then
    parallel --jobs "$JOBS" \
      correct_one {1} {2} {3}/{1} ::: "${METHODS[@]}" ::: "${FILES[@]}" :::+ "${UBDIR}"
  else
    for FILE in "${FILES[@]}"; do
      for METHOD in "${METHODS[@]}"; do
        correct_one "$METHOD" "$FILE" "${UBDIR}/${METHOD}" &
        while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 0.1; done
      done
    done
    wait
  fi
done
echo "✅  All corrections complete."
