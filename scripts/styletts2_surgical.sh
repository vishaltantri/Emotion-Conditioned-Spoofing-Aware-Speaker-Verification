#!/usr/bin/env bash
# One-off StyleTTS2.zip surgical repair: delete all parts referenced by the
# current diag, refetch each deleted part at 8 parallel sub-slices (two parts
# in flight), rebuild the zip, CRC-diagnose, and extract when valid.
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] repair: $*" >> "$LOG"; }
SEGDL="/home/vct/Desktop/AI project/scripts/segdl.sh"
PY3="/home/vct/Desktop/AI project/.venv/bin/python"
DIAG="/home/vct/Desktop/AI project/scripts/zip_diag.py"
URL="https://zenodo.org/api/records/15557842/files/StyleTTS2.zip/content"
Z="$D/emospoof/StyleTTS2.zip"
TOTAL=2935038030
NPARTS=12
export SEGDL_KEEP_PARTS=1

refetch_part() {  # refetch_part IDX — 8 parallel sub-slices into a fresh part
  local idx="$1"
  local start=$(( idx * (TOTAL / NPARTS + 1) ))
  local end=$(( start + TOTAL / NPARTS - 1 )); [[ $end -ge $TOTAL ]] && end=$(( TOTAL - 1 ))
  local subtotal=$(( end - start + 1 ))
  local T="$D/emospoof/repair_p${idx}.bin"
  rm -f "$T" "$T".part*
  if bash "$SEGDL" "$URL" "$T" "$subtotal" 8 "$D/emospoof/segdl.log"; then
    mv "$T" "$Z.part$(printf '%02d' "$idx")"
    rm -f "$T" "$T".part*
    say "part$idx refetched OK"
    return 0
  fi
  say "part$idx REFETCH FAILED"; return 1
}

for cycle in 1 2 3 4 5 6; do
  rm -f "$Z"
  # collect bad parts from the latest diag (cycle 1 uses the known list)
  local_json="$D/emospoof/bad_parts.json"
  if [[ $cycle -eq 1 ]]; then
    :   # bad_parts.json already written by the driver analysis
  fi
  IDX=$(python3 -c "import json;print(' '.join(map(str,json.load(open('$local_json')))))" 2>/dev/null)
  if [[ -z "$IDX" ]]; then
    say "no bad parts listed; verifying zip instead"
  fi
  # delete every bad part
  for i in $IDX; do
    rm -f "$Z.part$(printf '%02d' "$i")"
  done
  say "cycle $cycle: refetching parts: $IDX"
  # refetch two parts at a time (16 connections total)
  arr=($IDX)
  n=${#arr[@]}
  for ((j=0; j<n; j+=2)); do
    refetch_part "${arr[$j]}" &
    [[ $((j+1)) -lt $n ]] && refetch_part "${arr[$((j+1))]}" &
    wait
  done
  # rebuild + diag
  bash "$SEGDL" "$URL" "$Z" "$TOTAL" "$NPARTS" "$D/emospoof/segdl.log" || { say "SEGDL_FAIL cycle $cycle"; exit 1; }
  if "$PY3" "$DIAG" "$Z" > "$D/emospoof/StyleTTS2_diag.txt" 2>&1; then
    nbad=$(wc -l < "$D/emospoof/StyleTTS2_diag.txt")
    say "cycle $cycle: $nbad bad members"
    if [[ "$nbad" -eq 0 ]]; then
      say "StyleTTS2.zip FULLY VALID (cycle $cycle)"
      # extract + marker
      rm -rf "$D/emospoof/StyleTTS2" "$D/emospoof/__MACOSX"
      if timeout 500 unzip -q -n "$Z" -d "$D/emospoof/"; then
        rm -f "$Z" "$Z".part*
        say "StyleTTS2.zip extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
        exit 0
      else
        say "EXTRACT_FAILED despite valid CRC"; exit 2
      fi
    fi
    # recompute the bad-parts map for the next cycle
    python3 - "$D/emospoof/StyleTTS2_diag.txt" "$D/emospoof/bad_parts.json" <<'PYEOF'
import sys, json, collections
CHUNK = 2935038030 // 12 + 1
idx = collections.Counter()
for line in open(sys.argv[1]):
    if line.strip():
        idx[int(line.split()[0]) // CHUNK] += 1
json.dump(sorted(idx), open(sys.argv[2], "w"))
print(sorted(idx))
PYEOF
  else
    say "diag failure in cycle $cycle"; sleep 20
  fi
done
say "gave up after 6 surgical cycles"
exit 3
