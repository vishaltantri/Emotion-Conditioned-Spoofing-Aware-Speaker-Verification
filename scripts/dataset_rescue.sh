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

fetch_md5() {  # KEY -> downloads + md5-verifies one archive
  local key="$1" md5="${MD5S[$key]}" size="${SZ[$key]}"
  local dest="$D/emospoof/$key" tmp attempts max=6
  local url="https://zenodo.org/api/records/15557842/files/$key/content"
  for attempts in 1 2 3 4 5 6 7 8; do
    [[ $attempts -gt $max ]] && return 1
    tmp="$dest.partial$RANDOM"
    say "fetch $key attempt $attempts (cache-buster query)"
    curl -sf --max-time 7200 -H "Accept-Encoding: identity" \
         "$url?rchar=$RANDOM" -o "$tmp" || true
    local got_size got_md5
    got_size=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
    got_md5=$(md5sum "$tmp" 2>/dev/null | cut -d' ' -f1)
    if [[ "$got_md5" == "$md5" ]]; then
      mv "$tmp" "$dest"
      say "$key DOWNLOADED md5-VERIFIED ($got_size bytes)"
      return 0
    fi
    say "md5 mismatch $key attempt $attempts (got $got_size bytes md5=$got_md5; server sometimes serves corrupt replicas); deleting tmp"
    rm -f "$tmp"
    sleep 30
  done
  return 1
}

for key in StyleTTS2.zip F5TTS.zip CosyVoice.zip; do
  if fetch_md5 "$key"; then
    rm -rf "$D/emospoof/StyleTTS2" "$D/emospoof/__MACOSX"
    say "$key CRC-verifying"
    if timeout 600 "$PY3" "$DIAG" "$D/emospoof/$key" > "$D/emospoof/${key}.diag" 2>&1 \
       && [[ ! -s "$D/emospoof/${key}.diag" ]]; then
      say "$key CRC-valid; extracting"
    else
      say "$key md5 matched but CRC failed?!" ; exit 1
    fi
    if unzip -q -n "$D/emospoof/$key" -d "$D/emospoof/"; then
      rm -f "$D/emospoof/$key"
      say "$key extracted+removed; disk $(df -BG / | awk 'NR==2{print $4}') free"
    else
      say "EXTRACT_FAILED $key"; exit 1
    fi
  else
    say "GAVE UP on $key after 8 attempts"
    exit 1
  fi
done

say "zenodo set complete"
