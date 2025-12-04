set -euo pipefail
REPORT="./report.py"
[[ -f "$REPORT" ]] || { echo "Cannot find report.py"; exit 1; }
report_dir () {
    local DIR="$1"
    local OUT="${DIR}/report.txt"
    local FILES=( "${DIR}"/result*.txt )
    [[ -e "${FILES[0]}" ]] || return
    python "$REPORT" -o "$OUT" "${FILES[@]}"
}
report_file () {
    local FILE="$1"
    local OUT="${FILE%.*}_report.txt"
    python "$REPORT" -o "$OUT" "$FILE"
}
find . -maxdepth 1 -type d -name 'ub_*_*' | while read -r DATASET; do
    echo "▶ Dataset ${DATASET
    find "$DATASET" -mindepth 1 -maxdepth 1 -type d -name 'emd*' | \
      while read -r EDIR; do
          echo "  • summarising $(basename "$EDIR")"
          report_dir "$EDIR"
      done
    SRDIR="${DATASET}/single_runs"
    if [[ -d "$SRDIR" ]]; then
        for FILE in "$SRDIR"/result*.txt; do
            [[ -e "$FILE" ]] || continue
            echo "  • single $(basename "$FILE")"
            report_file "$FILE"
        done
    fi
done
echo "✅  Reports generated."
