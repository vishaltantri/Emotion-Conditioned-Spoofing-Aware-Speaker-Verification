#!/usr/bin/env bash
# Final dataset rescue: sequential single-stream downloads from zenodo with
# md5 gates (checkums straight from the record metadata), then USVspoof LA.
set -u
D=/home/vct/data/easv
LOG="$D/status.log"
say() { echo "[$(date '+%F %T')] rescue: $*" >> "$LOG"; }
PY3="/home/vct/Desktop/AI project/.venv/bin/python"
DIAG="/home/vct/Desktop/AI project/scripts/zip_diag.py"
declare -A MD5S=( )
MD5S[StyleTTS2.zip]=cc70b4d6da9f2ecde9637b57b5e56726
MD5S[F5TTS.zip]=2f05f42cc810a90f5870f22b9636bf73
MD5S[CosyVoice.zip]=b4a8257cb6acbba8c99903929b8eaab4
declare -A SZ=( [StyleTTS2.zip]=2935038030 [F5TTS.zip]=1091810675 [CosyVoice.zip]=2668440599 )
mkdir -p "$D/emospoof"

fetch_md5() {  # KEY -> downloads + md5-verifies one archive, resuming across cuts
  local key="$1" md5="${MD5S[$key]}" size="${SZ[$key]}"
  local dest="$D/emospoof/$key" tmp attempts max=8 stalls count
  local url="https://zenodo.org/api/records/15557842/files/$key/content"
  # A stable partial name lets `curl -C -` continue where the server cut the
  # stream: Zenodo has been truncating this 2.9 GB archive mid-transfer, and
  # restarting from zero each time burned hours without gaining a byte.
  tmp="$dest.download"
  for attempts in $(seq 1 $max); do
    stalls=0
    while :; do
      count=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
      if [[ $count -ge $size ]]; then break; fi
      say "fetch $key attempt $attempts from byte $count (resume)"
      curl -sf -C - --max-time 3600 -H "Accept-Encoding: identity" \
           "$url" -o "$tmp" || true
      local now
      now=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
      if [[ $now -le $count ]]; then
        stalls=$((stalls + 1))
        say "no progress ($now of $size); stall $stalls"
        [[ $stalls -ge 3 ]] && break
        sleep 20
      else
        stalls=0
      fi
    done
    local got_size got_md5
    got_size=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
    got_md5=$(md5sum "$tmp" 2>/dev/null | cut -d' ' -f1)
    if [[ "$got_md5" == "$md5" ]]; then
      mv "$tmp" "$dest"
      say "$key DOWNLOADED md5-VERIFIED ($got_size bytes, attempt $attempts)"
      return 0
    fi
    say "md5 mismatch $key attempt $attempts (got $got_size bytes md5=$got_md5); deleting partial and restarting clean"
    rm -f "$tmp"   # a resumed file from mixed replicas can never verify
    sleep 30
  done
  return 1
}

# Optional argv = only those archives (e.g. `dataset_rescue.sh StyleTTS2.zip`).
if [[ $# -gt 0 ]]; then
  keys=("$@")
else
  keys=(StyleTTS2.zip F5TTS.zip CosyVoice.zip)
fi

for key in "${keys[@]}"; do
  tree="$D/emospoof/${key%.zip}"
  # An already extracted tree must not be re-downloaded nor deleted: earlier
  # versions removed StyleTTS2 unconditionally after every archive, which wiped
  # the finished tree while extracting F5TTS.
  if [[ -n "$(find "$tree" -name '*.wav' -print -quit 2>/dev/null)" ]]; then
    say "$key already extracted ($(find "$tree" -name '*.wav' | wc -l) wavs); skip"
    continue
  fi
  if fetch_md5 "$key"; then
    rm -rf "$tree" "$D/emospoof/__MACOSX"
    say "$key CRC-verifying"
    if timeout 600 "$PY3" "$DIAG" "$D/emospoof/$key" > "$D/emospoof/${key}.diag" 2>&1 \
       && [[ ! -s "$D/emospoof/${key}.diag" ]]; then
      say "$key CRC-valid; extracting"
    else
      say "$key md5 matched but CRC failed?!" ; exit 1
    fi
    if unzip -q -n "$D/emospoof/$key" -d "$D/emospoof/"; then
      rm -f "$D/emospoof/$key"
      say "$key extracted ($(find "$tree" -name '*.wav' | wc -l) wavs)+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
    else
      say "EXTRACT_FAILED $key"; exit 1
    fi
  else
    say "GAVE UP on $key after 8 attempts"
    exit 1
  fi
done

say "zenodo set complete"
