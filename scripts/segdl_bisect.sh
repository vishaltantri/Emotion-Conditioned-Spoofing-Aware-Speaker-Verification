#!/usr/bin/env bash
set -u
U="https://zenodo.org/api/records/15557842/files/StyleTTS2.zip/content"
rm -f /tmp/t2.bin /tmp/t2.bin.part*
bash "/home/vct/Desktop/AI project/scripts/segdl.sh" "$U" /tmp/t2.bin 3000 3 /tmp/t2.log >/dev/null 2>&1

echo '--- server truth per 1000-byte slice ---'
for i in 0 1 2; do
  off=$(( i * 1000 ))
  curl -sf --max-time 40 -r "$off-$(( off + 999 ))" "$U" \
    | sha256sum | cut -c1-16
done
echo '--- segdl /tmp/t2.bin per 1000-byte slice ---'
for i in 0 1 2; do
  off=$(( i * 1000 ))
  dd if=/tmp/t2.bin bs=1 skip="$off" count=1000 2>/dev/null \
    | sha256sum | cut -c1-16
done
echo '--- individual part files ---'
for p in /tmp/t2.bin.part00 /tmp/t2.bin.part01 /tmp/t2.bin.part02; do
  stat -c%s "$p" 2>/dev/null || echo "missing $p"
done
echo '--- t2 log ---'
cat /tmp/t2.log
