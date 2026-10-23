#!/usr/bin/env bash
# hf_fetch.sh URL DEST [EXPECTED_BYTES LOGFILE]
# Resumable download of one public Hugging Face dataset file.
#
# The Edinburgh and Zenodo hosts cap this machine at ~200 KB/s, while the HF CDN
# serves the same corpora at >1 MB/s, so the mirrors are the practical route. HF
# redirects to a signed CDN URL; `curl -L -C -` re-resolves on each retry, which
# is why a plain resume loop is safe here.
set -u
URL="$1"; DEST="$2"; EXPECT="${3:-0}"; LOG="${4:-$DEST.fetch.log}"

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

for attempt in $(seq 1 60); do
  have=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
  if [[ "$EXPECT" -gt 0 && "$have" -ge "$EXPECT" ]]; then
    say "$DEST complete ($have bytes)"; exit 0
  fi
  say "attempt $attempt from byte $have"
  curl -sfL -C - --max-time 5400 --retry 3 --retry-delay 5 "$URL" -o "$DEST" || true
  after=$(stat -c%s "$DEST" 2>/dev/null || echo 0)
  if [[ "$EXPECT" -gt 0 && "$after" -ge "$EXPECT" ]]; then
    say "$DEST complete ($after bytes)"; exit 0
  fi
  if [[ "$after" -le "$have" ]]; then
    say "no progress ($after bytes); backing off"
    sleep 20
  fi
done

say "$DEST INCOMPLETE after 60 attempts"
exit 1
