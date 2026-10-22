#!/usr/bin/env bash
# Datasets stream 2: repair corrupted StyleTTS2.zip part(s), extract, then
# F5TTS -> CosyVoice -> LA. Self-healing: on unzip failure, maps the first bad
# offset back to its part, re-downloads that part at 8 parallel sub-slices,
# rebuilds and re-tests (up to 6 iterations).
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] auto2: $*" >> "$LOG"; }
SEGDL="/home/vct/Desktop/AI project/scripts/segdl.sh"
SRC="https://zenodo.org/api/records/15557842/files"
URL="$SRC/StyleTTS2.zip/content"
Z="$D/emospoof/StyleTTS2.zip"
TOTAL=2935038030
NPARTS=12
CHUNK=$(( (TOTAL + NPARTS - 1) / NPARTS ))

mkdir -p "$D/emospoof"
rm -f "$Z"   # rebuild from parts below

# ---- Phase A: validate/repair parts before concat -------------------------
for attempt in 1 2 3 4 5 6; do
  # make sure every part has exactly its expected size (delete oversize/defects)
  bad=0
  for i in $(seq 0 $((NPARTS-1))); do
    P="$Z.part$(printf '%02d' "$i")"
    START=$(( i * CHUNK )); END=$(( START + CHUNK - 1 )); [[ $END -ge $TOTAL ]] && END=$(( TOTAL-1 ))
    EXPECT=$(( END - START + 1 ))
    have=$(stat -c%s "$P" 2>/dev/null || echo 0)
    if [[ $have -ne $EXPECT ]]; then
      say "StyleTTS2 part$i missing/oversize (have=$have expect=$EXPECT) -> recomputing"
      if [[ $have -gt $EXPECT ]]; then rm -f "$P"; fi
      # missing -> let segdl fetch it (other parts skip instantly)
    fi
  done
  bash "$SEGDL" "$URL" "$Z" "$TOTAL" "$NPARTS" "$D/emospoof/segdl.log" || { say "SEGDL_FAIL attempt $attempt"; sleep 20; continue; }
  # byte-exact concat done by segdl; now CRC test
  if timeout 280 unzip -t "$Z" > "$D/emospoof/styletts2_test.log" 2>&1; then
    say "StyleTTS2.zip CRC-valid after $attempt iteration(s)"
    break
  else
    # map first failing offset -> part index
    badoff=$(grep -oE "bad zipfile offset[^:]*: *[0-9]+" "$D/emospoof/styletts2_test.log" | head -1 | grep -oE "[0-9]+$")
    if [[ -n "$badoff" ]]; then
      idx=$(( badoff / CHUNK ))
      say "CRC fail: bad offset $badoff lies in part$idx; deleting + refetching 8-way"
      rm -f "$Z.part$(printf '%02d' "$idx")" "$Z"
      # refetch that single part split into 8 parallel sub-slices using a temp dest
      T="$D/emospoof/repair_part03.bin.tmp"
      rm -f "$T.part"*
      START=$(( idx * CHUNK )); END=$(( START + CHUNK - 1 )); [[ $END -ge $TOTAL ]] && END=$(( TOTAL-1 ))
      SUBTOTAL=$(( END - START + 1 ))
      bash "$SEGDL" "$URL" "$T" "$SUBTOTAL" 8 "$D/emospoof/segdl.log" \
        && mv "$T" "$Z.part$(printf '%02d' "$idx")" \
        && say "part$idx repaired ($(stat -c%s "$Z.part$(printf '%02d' "$idx")") bytes)" \
        || { say "repair of part$idx failed"; }
      rm -f "$Z"
    else
      say "CRC fail but no bad offset identified; inflate errors mean the test"
      say "cycle gives up looking for offsets; deleting the LAST part (central dir)"
      # inflate error may be in any part; fall back to re-fetching part at the
      # offset inferred from the first "invalid compressed data" file, else 3
      rm -f "$Z.part03" "$Z"
      T="$D/emospoof/repair_part03.bin.tmp"
      rm -f "$T.part"*
      START=$(( 3 * CHUNK )); END=$(( START + CHUNK - 1 )); [[ $END -ge $TOTAL ]] && END=$(( TOTAL-1 ))
      SUBTOTAL=$(( END - START + 1 ))
      bash "$SEGDL" "$URL" "$T" "$SUBTOTAL" 8 "$D/emospoof/segdl.log" \
        && mv "$T" "$Z.part03" && say "part03 repaired (fallback)" \
        || say "part03 repair failed"
      rm -f "$Z"
    fi
  fi
done

# ---- Phase B: extract + marker ---------------------------------------------
if timeout 400 unzip -q -n "$Z" -d "$D/emospoof/"; then
  rm -f "$Z"
  say "StyleTTS2.zip extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
else
  say "EXTRACT_FAILED StyleTTS2 again - giving up on this cycle"
  exit 1
fi

# ---- Phase C: F5TTS + CosyVoice (as in datasets stream 1) -------------------
declare -A SZ=( [F5TTS.zip]=1091810675 [CosyVoice.zip]=2668440599 )
for f in F5TTS.zip CosyVoice.zip; do
  [[ -f "$D/emospoof/$f" ]] && rm -f "$D/emospoof/$f"
  bash "$SEGDL" "$SRC/$f/content" "$D/emospoof/$f" "${SZ[$f]}" 12 "$D/emospoof/segdl.log" \
    || { say "SEGDL_FAIL $f"; exit 1; }
  say "$f downloaded; extracting"
  if unzip -q -n "$D/emospoof/$f" -d "$D/emospoof/"; then
    rm -f "$D/emospoof/$f"
    say "$f extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
  else
    say "EXTRACT_FAILED $f"; exit 1
  fi
done

LA="$D/asvspoof2019LA"; mkdir -p "$LA"
bash "$SEGDL" "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content" \
  "$LA/LA.zip" 7640952520 16 "$LA/segdl.log" || { say "SEGDL_FAIL LA.zip"; exit 1; }
say "LA downloaded; disk $(df -BG / | awk 'NR==2{print $4}') free"
if unzip -q -n "$LA/LA.zip" "LA/ASVspoof2019_LA_train/*" "LA/ASVspoof2019_LA_dev/*" "LA/ASVspoof2019_LA_cm_protocols/*" -d "$LA/"; then
  say "LA train+dev+protocols extracted"
else
  say "LA_SELECTIVE_EXTRACT_FAILED"
  unzip -l "$LA/LA.zip" | head -60 > "$LA/la_zip_listing.txt"
  exit 1
fi
rm -f "$LA/LA.zip"
say "LA.zip removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
say "DATASETS_STREAM_COMPLETE"
