#!/usr/bin/env bash
set -euo pipefail
PERCENTS=(0.30)          
DISTS=(uniform)          
JOBS=4                                   
INJECTOR="./injector.py"                
PART_FILES=(ub_part_tgfd2_*.txt)              
[[ -x "$(command -v python)" ]] || { echo "python not found"; exit 1; }
[[ -f "$INJECTOR" ]] || { echo "Cannot find $INJECTOR"; exit 1; }
[[ ${
export INJECTOR
inject_one() {
  local PCT="$1" DIST="$2" PART="$3"
  local PCTLABEL
  printf -v PCTLABEL "%.0f" "$(echo "$PCT * 100" | bc -l)"   
  local OUTDIR="ub_${PCTLABEL}_${DIST}"
  mkdir -p "$OUTDIR"
  local BASE="${PART%.txt}"
  python "$INJECTOR" "$PCT" "$DIST" \
         --input "$PART" \
         --output "${OUTDIR}/${BASE}_injected.txt"
}
export -f inject_one            
if command -v parallel >/dev/null 2>&1; then
  parallel --jobs "$JOBS" \
    inject_one {1} {2} {3} ::: "${PERCENTS[@]}" ::: "${DISTS[@]}" ::: "${PART_FILES[@]}"
else
  for pct in "${PERCENTS[@]}"; do
    for dist in "${DISTS[@]}"; do
      for part in "${PART_FILES[@]}"; do
        inject_one "$pct" "$dist" "$part" &
        while (( $(jobs -rp | wc -l) >= JOBS )); do sleep 0.1; done
      done
    done
  done
  wait
fi
echo "✅  All injections complete."
