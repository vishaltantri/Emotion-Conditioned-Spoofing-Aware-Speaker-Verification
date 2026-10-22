#!/usr/bin/env bash
# Sequencer: finish EmoSpoof downloads (StyleTTS2 orphan -> F5TTS -> CosyVoice),
# extract, then fetch + selectively extract ASVspoof 2019 LA. Disk-safe: one big
# archive + its extraction at a time. Fully resumable.
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
SRC="https://zenodo.org/api/records/15557842/files"
LA_URL="https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"

declare -A SZ=( [F5TTS.zip]=1091810675 [CosyVoice.zip]=2668440599 [StyleTTS2.zip]=2935038030 )

# ---- Stage 1: let the running StyleTTS2 wget finish its file --------------
while [[ ! -f "$D/emospoof/StyleTTS2.zip" || $(stat -c%s "$D/emospoof/StyleTTS2.zip") -lt ${SZ[StyleTTS2.zip]} ]]; do
  # if no wget owns it, resume it here
  if ! ps -o pid= --no-headers -C wget 2>/dev/null | xargs -r -I{} sh -c 'grep -q StyleTTS2 "/proc/{}/cmdline" 2>/dev/null' | grep -q 1; then
    say "StyleTTS2 owner gone; resuming"
    ( cd "$D/emospoof" && wget -c --tries=10 --timeout=30 "$SRC/StyleTTS2.zip/content" -O StyleTTS2.zip 2>>download.log ) &
    wait $!
  else
    sleep 30
  fi
done
say "StyleTTS2 complete"

# ---- Stage 2: F5TTS then CosyVoice ----------------------------------------
for f in F5TTS.zip CosyVoice.zip; do
  while [[ $(stat -c%s "$D/emospoof/$f" 2>/dev/null || echo 0) -lt ${SZ[$f]} ]]; do
    ( cd "$D/emospoof" && wget -c --tries=10 --timeout=30 "$SRC/$f/content" -O "$f" 2>>download.log ) \
      || sleep 10
  done
  say "$f complete; extracting"
  if unzip -q -n "$D/emospoof/$f" -d "$D/emospoof/"; then
    rm -f "$D/emospoof/$f"
    say "$f extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
  else
    say "EXTRACT_FAILED $f"; exit 1
  fi
done

# ---- Stage 3: LA download + selective extraction ---------------------------
LA="$D/asvspoof2019LA"; mkdir -p "$LA"
if [[ $(stat -c%s "$LA/LA.zip" 2>/dev/null || echo 0) -lt 7640952520 ]]; then
  say "downloading ASVspoof 2019 LA.zip"
  ( cd "$LA" && wget -c --tries=10 --timeout=30 "$LA_URL" -O LA.zip 2>>download.log ) || { say LA_DOWNLOAD_FAILED; exit 1; }
fi
say "LA downloaded; disk $(df -BG / | awk 'NR==2{print $4}') free"
if unzip -q -n "$LA/LA.zip" "LA/ASVspoof2019_LA_train/*" "LA/ASVspoof2019_LA_dev/*" "LA/ASVspoof2019_LA_cm_protocols/*" -d "$LA/"; then
  say "LA train+dev+protocols extracted"
else
  say "LA_SELECTIVE_EXTRACT_FAILED (listing kept in la_zip_listing.txt)"
  unzip -l "$LA/LA.zip" | head -60 > "$LA/la_zip_listing.txt"
  exit 1
fi
rm -f "$LA/LA.zip"
say "LA.zip removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
say "DOWNLOADS_COMPLETE"
