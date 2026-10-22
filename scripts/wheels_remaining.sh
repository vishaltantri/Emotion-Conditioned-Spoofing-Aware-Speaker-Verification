#!/usr/bin/env bash
# Finish wheel downloads (segdl short-circuits completed files) and install.
# Two passes over the url list: pass 1 downloads, pass 2 re-fetches whatever
# failed or is still missing before the offline install fires.
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
SEGDL="/home/vct/Desktop/AI project/scripts/segdl.sh"

say "wheels stream: starting/resuming"
cd "$D/wheels"
for pass in 1 2; do
  i=0
  missed=0
  while read -r url; do
    i=$((i+1))
    fname="$D/wheels/$(basename "$url")"
    {
      len=$(curl -sIL --max-time 30 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
      if [[ "$pass" -eq 1 || -z "$len" || ! -s "$fname" ]]; then
        [[ -z "$len" ]] && { say "no-length wheel $i ($fname)"; missed=$((missed+1)); continue; }
        bash "$SEGDL" "$url" "$fname" "$len" 8 "$D/wheels/segdl.log" || { say "SEGDL_FAIL $fname (pass $pass)"; missed=$((missed+1)); }
      fi
    } || true
  done < "$D/wheels/urls.txt"
  say "wheels pass $pass done; missed=$missed"
done
say "wheels done; offline install"
cd "/home/vct/Desktop/AI project"
.venv-rm/bin/python -m pip install --no-index --find-links "$D/wheels" \
  torch torchaudio transformers soundfile speechbrain >> "$D/wheels/install.log" 2>&1 \
  && say "VENV_RM_READY" || say "VENV_RM_INSTALL_FAILED (see wheels/install.log)"
