#!/usr/bin/env bash
# Extract the English ESD subset (speakers 0011-0020) once disk allows it.
# Waits for the marker "LA.zip removed" written by download_tierA.sh, which
# guarantees ~7.12 GB has been freed. Resumable and idempotent.
set -u
ROOT="$HOME/data/easv"
LOG="$ROOT/status.log"
ZIP="$ROOT/esd/ESD.zip"
OUT="$ROOT/esd"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

# wait for the download to complete and the disk-heavy LA stage to finish
while ! grep -q "LA.zip removed" "$LOG" 2>/dev/null; do sleep 60; done

# ESD.zip may still be downloading (slower mirror) — wait for ESD_DONE marker
while ! grep -q "ESD_DONE" "$ROOT/esd/download.log" 2>/dev/null; do
  # robustness: also accept a complete file by size (2,446,419,486 bytes)
  s=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  if [[ $s -ge 2446419486 ]]; then break; fi
  sleep 60
done
say "ESD download complete; extracting English subset"

if unzip -q -n "$ZIP" "*English*" -d "$OUT/" 2>>"$LOG"; then
  say "English subset extracted"
else
  say "filter extract failed — falling back to full listing"
  unzip -l "$ZIP" | head -50 > "$OUT/esd_zip_listing.txt"
  exit 1
fi
rm -f "$ZIP"; say "ESD.zip removed; disk now: $(df -BG / | awk 'NR==2{print $4}') free"

ls -d "$OUT/English"/* 2>/dev/null | head -15 >> "$LOG"
say "ESD_STAGE_COMPLETE"
