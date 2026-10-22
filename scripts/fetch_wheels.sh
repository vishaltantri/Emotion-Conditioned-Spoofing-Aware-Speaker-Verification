#!/usr/bin/env bash
# Download every wheel in /tmp/easv_report.json (resumable, wget -c) into
# ~/data/easv/wheels, then install all of them into .venv-rm offline.
set -u
WHEELDIR="$HOME/data/easv/wheels"
REPORT=/tmp/easv_report.json
LOG="$HOME/data/easv/status.log"
mkdir -p "$WHEELDIR"
python3 - "$REPORT" "$WHEELDIR/urls.txt" <<'EOF'
import json, sys
rep = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w") as out:
    for item in rep["install"]:
        out.write(item["download_info"]["url"] + "\n")
EOF

echo "[$(date '+%F %T')] wheel download started" >> "$LOG"
while read -r url; do
  fname="$WHEELDIR/$(basename "$url")"
  if [[ -f "$fname" ]]; then
    expected_len=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
    actual_len=$(stat -c%s "$fname" 2>/dev/null || echo 0)
    if [[ -n "$expected_len" && "$actual_len" -eq "$expected_len" ]]; then
      continue
    fi
  fi
  wget -c --tries=20 --timeout=30 --waitretry=5 "$url" -O "$fname" 2>>"$WHEELDIR/fetch.log" \
    || echo "[$(date '+%F %T')] RETRY_LATER $(basename "$url")" >> "$LOG"
done < "$WHEELDIR/urls.txt"
echo "[$(date '+%F %T')] wheel download finished" >> "$LOG"

cd "/home/vct/Desktop/AI project"
.venv-rm/bin/python -m pip install --no-index --find-links "$WHEELDIR" \
  torch torchaudio transformers soundfile speechbrain >> "$WHEELDIR/install.log" 2>&1 \
  && echo "[$(date '+%F %T')] VENV_RM_READY" >> "$LOG" \
  || echo "[$(date '+%F %T')] VENV_RM_INSTALL_FAILED (see wheels/install.log)" >> "$LOG"
