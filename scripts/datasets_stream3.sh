#!/usr/bin/env bash
# Datasets stream v3: rebuild StyleTTS2.zip from retained parts with a
# targeted diagnostic repair loop (only bad parts are deleted), extract,
# then F5TTS -> CosyVoice -> LA.
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] auto3: $*" >> "$LOG"; }
SEGDL="/home/vct/Desktop/AI project/scripts/segdl.sh"
DIAG="/home/vct/Desktop/AI project/scripts/zip_diag.py"
PY3="/home/vct/Desktop/AI project/.venv/bin/python"
SRC="https://zenodo.org/api/records/15557842/files"
URL="$SRC/StyleTTS2.zip/content"
Z="$D/emospoof/StyleTTS2.zip"
TOTAL=2935038030
NPARTS=12
CHUNK=$(( (TOTAL + NPARTS - 1) / NPARTS ))
export SEGDL_KEEP_PARTS=1   # never blow parts away after a concat

mkdir -p "$D/emospoof"
rm -f "$Z.part05"   # known bad from earlier analysis (if still present)

# ---- Phase A: converge on a CRC-valid zip ----------------------------------
valid=0
for attempt in 1 2 3 4 5 6 7 8; do
  rm -f "$Z"                       # force segdl to rebuild (parts retained)
  bash "$SEGDL" "$URL" "$Z" "$TOTAL" "$NPARTS" "$D/emospoof/segdl.log" \
    || { say "SEGDL_FAIL attempt $attempt"; sleep 20; continue; }
  if "$PY3" "$DIAG" "$Z" > "$D/emospoof/StyleTTS2_diag.txt" 2>&1; then
    nbad=$(wc -l < "$D/emospoof/StyleTTS2_diag.txt")
    if [[ "$nbad" -eq 0 ]]; then
      say "StyleTTS2.zip FULLY VALID after $attempt diagnosed attempts"
      valid=1
      break
    fi
    first=$(head -1 "$D/emospoof/StyleTTS2_diag.txt")
    off=$(echo "$first" | cut -d' ' -f1)
    idx=$(( off / CHUNK ))
    say "attempts=$attempt: $nbad bad members; first bad offset $off -> part$idx; deleting that part"
    rm -f "$Z.part$(printf '%02d' "$idx")"
  else
    say "diag tool itself failed (attempt $attempt); waiting"; sleep 20
  fi
done
if [[ $valid -ne 1 ]]; then
  say "StyleTTS2 repair gave up after 8 attempts"; exit 1
fi

# ---- Phase B: extract + marker ---------------------------------------------
rm -rf "$D/emospoof/StyleTTS2" "$D/emospoof/__MACOSX"   # stale partial extraction
if timeout 500 unzip -q -n "$Z" -d "$D/emospoof/"; then
  rm -f "$Z" "$Z".part*
  say "StyleTTS2.zip extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
else
  say "EXTRACT_FAILED StyleTTS2 after valid CRC"; exit 1
fi

# ---- Phase C: F5TTS + CosyVoice + LA --------------------------------------
declare -A SZ=( [F5TTS.zip]=1091810675 [CosyVoice.zip]=2668440599 )
for f in F5TTS.zip CosyVoice.zip; do
  [[ -f "$D/emospoof/$f" ]] && rm -f "$D/emospoof/$f"
  ( unset SEGDL_KEEP_PARTS
    bash "$SEGDL" "$SRC/$f/content" "$D/emospoof/$f" "${SZ[$f]}" 12 "$D/emospoof/segdl.log" ) \
    || { say "SEGDL_FAIL $f"; exit 1; }
  # verify each downloaded generator zip with the CRC diagnostic too
  if "$PY3" "$DIAG" "$D/emospoof/$f" > "$D/emospoof/${f}.diag" 2>&1 \
     && [[ -s "$D/emospoof/${f}.diag" ]]; then
    say "CRC FAIL in $f: $(wc -l < "$D/emospoof/${f}.diag") bad members; giving up on it"
    exit 1
  fi
  say "$f downloaded+CRC-valid; extracting"
  if unzip -q -n "$D/emospoof/$f" -d "$D/emospoof/"; then
    rm -f "$D/emospoof/$f"
    say "$f extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
  else
    say "EXTRACT_FAILED $f"; exit 1
  fi
done

LA="$D/asvspoof2019LA"; mkdir -p "$LA"
( unset SEGDL_KEEP_PARTS
  bash "$SEGDL" "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content" \
    "$LA/LA.zip" 7640952520 16 "$LA/segdl.log" ) || { say "SEGDL_FAIL LA.zip"; exit 1; }
say "LA downloaded; disk $(df -BG / | awk 'NR==2{print $4}') free"
if unzip -q -n "$LA/LA.zip" "LA/ASVspoof2019_LA_train/*" "LA/ASVspoof2019_LA_dev/*" "LA/ASVspoof2019_LA_cm_protocols/*" -d "$LA/"; then
  say "LA train+dev+protocols extracted"
else
  say "LA_SELECTIVE_EXTRACT_FAILED"; exit 1
fi
rm -f "$LA/LA.zip" "$LA/LA.zip".part*
say "LA.zip removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
say "DATASETS_STREAM_COMPLETE"
