#!/usr/bin/env bash
# segdl.sh URL DEST TOTAL_SIZE [NPARTS] [LOGFILE]
# Segmented, resumable HTTP download: splits TOTAL_SIZE into NPARTS ranges,
# downloads parts concurrently with curl -r, then concatenates and verifies.
# A complete destination short-circuits. Parts live next to DEST with suffix
# ".partNN".
set -u
URL="$1"; DEST="$2"; TOTAL="$3"; NPARTS="${4:-12}"; LOG="${5:-$DEST.segdl.log}"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

if [[ -f "$DEST" && "$(stat -c%s "$DEST")" -ge "$TOTAL" ]]; then
  say "$DEST already complete ($(stat -c%s "$DEST")); skip"; exit 0
fi

CHUNK=$(( (TOTAL + NPARTS - 1) / NPARTS ))
pids=()
for i in $(seq 0 $((NPARTS-1))); do
  START=$(( i * CHUNK ))
  END=$(( START + CHUNK - 1 ))
  [[ $END -ge $TOTAL ]] && END=$(( TOTAL - 1 ))
  PART="$DEST.part$(printf '%02d' "$i")"
  EXPECT=$(( END - START + 1 ))
  (
    while :; do
      have=$(stat -c%s "$PART" 2>/dev/null || echo 0)
      if [[ $have -gt $EXPECT ]]; then
        # oversized part = residue from an overlapping writer with possibly a
        # different chunk split; delete rather than truncate (bytes untrustworthy)
        rm -f "$PART"
        have=0
      fi
      if [[ $have -eq $EXPECT ]]; then break; fi
      from=$(( START + have ))
      to="$END"
      # NOTE: no --retry here: a curl retry restarts the whole range and
      # appends duplicate bytes after partial output (corruption source).
      # The outer while-loop is the retry mechanism.
      curl -sf --speed-limit 1024 --speed-time 30 \
           -r "$from-$to" -o - "$URL" >> "$PART" || sleep 5
      have_now=$(stat -c%s "$PART" 2>/dev/null || echo 0)
      if [[ $have_now -gt $EXPECT ]]; then
        rm -f "$PART"   # range alignment broken by an oversize append: restart part
      fi
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done

# verify all parts
ok=1
for i in $(seq 0 $((NPARTS-1))); do
  START=$(( i * CHUNK ))
  END=$(( START + CHUNK - 1 )); [[ $END -ge $TOTAL ]] && END=$(( TOTAL - 1 ))
  PART="$DEST.part$(printf '%02d' "$i")"
  EXPECT=$(( END - START + 1 ))
  have=$(stat -c%s "$PART" 2>/dev/null || echo 0)
  if [[ "$have" -ne "$EXPECT" ]]; then say "$PART incomplete ($have/$EXPECT)"; ok=0; fi
done
[[ $ok -eq 1 ]] || { say "segdl FAILED parts for $DEST"; exit 1; }

parts=()
for i in $(seq 0 $((NPARTS-1))); do
  parts+=("$DEST.part$(printf '%02d' "$i")")
done
cat "${parts[@]}" > "$DEST"
size=$(stat -c%s "$DEST")
if [[ "$size" -eq "$TOTAL" ]]; then
  if [[ -z "${SEGDL_KEEP_PARTS:-}" ]]; then
    for i in $(seq 0 $((NPARTS-1))); do rm -f "$DEST.part$(printf '%02d' "$i")"; done
  fi
  say "$DEST complete ($size bytes)"
else
  say "size mismatch $DEST ($size != $TOTAL)"; exit 1
fi
