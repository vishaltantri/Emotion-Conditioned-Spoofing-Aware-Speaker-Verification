#!/usr/bin/env bash
# Tier-A dataset orchestrator for the EASV project.
# Disk rule: never hold more than one large archive + its extraction.
# Status messages append to ~/data/easv/status.log
set -u
SRC="https://zenodo.org/api/records/15557842/files"
LA_URL="https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
ROOT="$HOME/data/easv"
EMU="$ROOT/emospoof"
LA="$ROOT/asvspoof2019LA"
LOG="$ROOT/status.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

mkdir -p "$EMU" "$LA"

# ---- Stage 1: wait for the EmoSpoof files by verified size ---------------
# (robust to the downloader dying: if it died, this stage downloads the
#  remainder itself with wget -c resume)
declare -A SZ=( [F5TTS.zip]=1091810675 [CosyVoice.zip]=2668440599 [StyleTTS2.zip]=2935038030 [wav_list.zip]=116559 )
for f in "${!SZ[@]}"; do
  while true; do
    s=$(stat -c%s "$EMU/$f" 2>/dev/null || echo 0)
    if [[ $s -ge ${SZ[$f]} ]]; then say "$f size OK ($s)"; break; fi
    say "$f incomplete ($s < ${SZ[$f]}): resuming download"
    wget -c --tries=3 "$SRC/$f/content" -O "$EMU/$f" 2>>"$EMU/download.log" && true
  done
done
say "EmoSpoof files verified by size"

expect=( "F5TTS.zip" "CosyVoice.zip" "StyleTTS2.zip" "wav_list.zip" )
for f in "${expect[@]}"; do
  if [[ -s "$EMU/$f" ]]; then say "OK $EMU/$f"; else say "MISSING $EMU/$f"; fi
done

# ---- Stage 2: extract each EmoSpoof zip, delete the zip ------------------
for f in "${expect[@]}"; do
  if [[ -s "$EMU/$f" ]]; then
    say "extracting $f"
    if unzip -q -n "$EMU/$f" -d "$EMU/"; then rm -f "$EMU/$f"; say "extracted+removed $f"
    else say "EXTRACT_FAILED $f"; fi
  fi
done
say "EmoSpoof extraction done; disk now: $(df -BG / | awk 'NR==2{print $4}') free"

# ---- Stage 3: ASVspoof 2019 LA download ----------------------------------
ZIP="$LA/LA.zip"
if [[ ! -s "$ZIP" ]]; then
  say "downloading ASVspoof 2019 LA.zip (7.12 GB)"
  wget -c --tries=3 "$LA_URL" -O "$ZIP" 2>>"$LA/download.log" || { say "LA_DOWNLOAD_FAILED"; exit 1; }
fi
say "LA download done; disk now: $(df -BG / | awk 'NR==2{print $4}') free"

# ---- Stage 4: selective extraction (train + dev + metadata only) ---------
say "listing LA.zip top-level dirs"
unzip -l "$ZIP" | grep -E "ASVspoof2019_LA_(train|dev|cm_meta|asv_match)/" | head -3 >> "$LOG" 2>&1
if unzip -q -n "$ZIP" "LA/ASVspoof2019_LA_train/*" "LA/ASVspoof2019_LA_dev/*" -d "$LA/"; then
  say "train+dev extracted"
else
  say "SELECTIVE_EXTRACT_FAILED full-listing instead"
  unzip -l "$ZIP" | head -40 >> "$LA/la_zip_listing.txt"
  exit 1
fi
rm -f "$ZIP"; say "LA.zip removed; disk now: $(df -BG / | awk 'NR==2{print $4}') free"

# ---- Stage 5: report -------------------------------------------------------
{
  echo "== EmoSpoof-TTS =="; du -sh "$EMU"
  echo "== ASVspoof 2019 LA =="; du -sh "$LA"
  echo "UNUSED_SPACE: $(df -BG / | awk 'NR==2{print $4}')"
} >> "$LOG"
say "TIER_A_COMPLETE"
