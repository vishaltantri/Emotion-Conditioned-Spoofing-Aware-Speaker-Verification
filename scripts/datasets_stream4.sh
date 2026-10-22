#!/usr/bin/env bash
# Datasets stream v4: wait for the StyleTTS2 marker (written by the surgical
# repair), then F5TTS -> CosyVoice -> ASVspoof 2019 LA (selective extract).
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] auto4: $*" >> "$LOG"; }
SEGDL="/home/vct/Desktop/AI project/scripts/segdl.sh"
PY3="/home/vct/Desktop/AI project/.venv/bin/python"
DIAG="/home/vct/Desktop/AI project/scripts/zip_diag.py"
SRC="https://zenodo.org/api/records/15557842/files"

# wait for StyleTTS2 to be extracted
while ! grep -q "rescue: StyleTTS2.zip extracted+removed" "$LOG" 2>/dev/null; do
  sleep 60
done
say "StyleTTS2 marker seen; starting F5TTS"

declare -A SZ=( [F5TTS.zip]=1091810675 [CosyVoice.zip]=2668440599 )
for f in F5TTS.zip CosyVoice.zip; do
  unset SEGDL_KEEP_PARTS
  bash "$SEGDL" "$SRC/$f/content" "$D/emospoof/$f" "${SZ[$f]}" 12 "$D/emospoof/segdl.log" \
    || { say "SEGDL_FAIL $f"; exit 1; }
  if "$PY3" "$DIAG" "$D/emospoof/$f" > "$D/emospoof/${f}.diag" 2>&1 \
     && [[ -s "$D/emospoof/${f}.diag" ]]; then
    say "CRC FAIL in $f: $(wc -l < "$D/emospoof/${f}.diag") bad members; aborting chain"
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
bash "$SEGDL" "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content" \
  "$LA/LA.zip" 7640952520 16 "$LA/segdl.log" || { say "SEGDL_FAIL LA.zip"; exit 1; }
say "LA downloaded; disk $(df -BG / | awk 'NR==2{print $4}') free"
if unzip -q -n "$LA/LA.zip" "LA/ASVspoof2019_LA_train/*" "LA/ASVspoof2019_LA_dev/*" "LA/ASVspoof2019_LA_cm_protocols/*" -d "$LA/"; then
  say "LA train+dev+protocols extracted"
else
  say "LA_SELECTIVE_EXTRACT_FAILED"; exit 1
fi
rm -f "$LA/LA.zip" "$LA/LA.zip".part*
say "LA.zip removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
say "DATASETS_STREAM_COMPLETE"
