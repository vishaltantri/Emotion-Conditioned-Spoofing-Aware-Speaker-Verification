#!/usr/bin/env bash
set -u
U="https://zenodo.org/api/records/15557842/files/StyleTTS2.zip/content"
BASE=244586503
SIZE=2000000
rm -f /tmp/t3.bin /tmp/t3.bin.part* /tmp/t3.direct
bash "/home/vct/Desktop/AI project/scripts/segdl.sh" "$U" /tmp/t3.bin "$SIZE" 4 /tmp/t3.log >/dev/null 2>&1
curl -sf --max-time 300 -r "$BASE-$(( BASE + SIZE - 1 ))" "$U" -o /tmp/t3.direct
stat -c%s /tmp/t3.bin /tmp/t3.direct
python3 - <<'EOF'
import hashlib
def chunks(path, size=4096):
    with open(path, "rb") as f:
        while True:
            b = f.read(size)
            if not b:
                return
            yield b
a = list(chunks("/tmp/t3.bin"))
b = list(chunks("/tmp/t3.direct"))
n = min(len(a), len(b))
diff = None
for i in range(n):
    if a[i] != b[i]:
        diff = i * 4096
        print("first divergent chunk offset:", diff)
        break
if a == b:
    print("IDENTICAL - assembly correct at this size")
else:
    print("sizes:", len(a), len(b))
    print("segdl head 64:", a[0][:64].hex())
    print("srv  head 64:", b[0][:64].hex())
    if diff is not None:
        pad = (diff // 4096) * 4096
        print("segdl@diff:", a[pad//4096][:64].hex())
        print("srv@diff :", b[pad//4096][:64].hex())
EOF
cat /tmp/t3.log | tail -5
