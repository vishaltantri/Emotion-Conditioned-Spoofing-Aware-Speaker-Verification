#!/usr/bin/env bash
# Remaining pipeline, built on segdl (parallel ranged curl):
#  1) remaining wheels  -> then offline pip install
#  2) StyleTTS2, F5TTS, CosyVoice  -> extract, delete each zip
#  3) ASVspoof 2019 LA -> selective extract (train/dev/protocols), delete
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
SEGDL="/home/vct/Desktop/AI project/scripts/segdl.sh"

declare -A SZ=( [F5TTS.zip]=1091810675 [CosyVoice.zip]=2668440599 [StyleTTS2.zip]=2935038030 )
SRC="https://zenodo.org/api/records/15557842/files"

# ---- Stage 1: wheels (8 parallel ranges each; sequential across wheels) ----
mkdir -p "$D/wheels"
say "segmented wheel download started"
while read -r url; do
  fname="$D/wheels/$(basename "$url")"
  len=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
  [[ -z "$len" ]] && { say "no-length $url"; continue; }
  bash "$SEGDL" "$url" "$fname" "$len" 8 "$D/wheels/segdl.log" || say "SEGDL_FAIL $url"
done < "$D/wheels/urls.txt"
say "wheels done; installing offline"
cd "/home/vct/Desktop/AI project"
.venv-rm/bin/python -m pip install --no-index --find-links "$D/wheels" \
  torch torchaudio transformers soundfile speechbrain >> "$D/wheels/install.log" 2>&1 \
  && say "VENV_RM_READY" || say "VENV_RM_INSTALL_FAILED (see wheels/install.log)"

# ---- Stage 2: EmoSpoof remaining -------------------------------------------
# StyleTTS2/F5TTS partials were single-connection prefixes; restart segmented.
for f in StyleTTS2.zip F5TTS.zip CosyVoice.zip; do
  [[ -f "$D/emospoof/$f" ]] && rm -f "$D/emospoof/$f"
  bash "$SEGDL" "$SRC/$f/content" "$D/emospoof/$f" "${SZ[$f]}" 12 "$D/emospoof/segdl.log" \
    || { say "SEGDL_FAIL $f"; exit 1; }
  say "$f downloaded; extracting"
  if unzip -q -n "$D/emospoof/$f" -d "$D/emospoof/"; then
    rm -f "$D/emospoof/$f"; say "$f extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
  else
    say "EXTRACT_FAILED $f"; exit 1
  fi
done

# ---- Stage 3: ASVspoof 2019 LA ----------------------------------------------
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
say "ALL_REMAINING_COMPLETE"
