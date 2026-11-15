# sourced by run_*.sh: resolves the repository root, creates output/log folders, provides failure tracking
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="${OUT:-$ROOT/rerun_output}"   # new records go here; the shipped raw results in results/ are never overwritten
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT/results" "$OUT/logs"
rm -f "$OUT/logs/failures.txt"
cd "$HERE"
# finish NAME: write the completion marker only if no case failed; otherwise report and return a non-zero status
finish() {
  if [ -s "$OUT/logs/failures.txt" ]; then
    echo "$1: FAILED, see $OUT/logs/failures.txt" >&2; cat "$OUT/logs/failures.txt" >&2; exit 1
  fi
  echo ALLDONE > "$OUT/logs/$1_done.txt"
}
