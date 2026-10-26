#!/usr/bin/env bash
# Fetch the CREMA-D audio without cloning the repository and without git-lfs.
#
# Why not `git lfs pull`: the release stores AudioWAV/ behind Git LFS and, on the
# machine this pipeline was built on, `git lfs pull` sat for minutes without
# fetching a single object (the batch endpoint never answered). GitHub serves the
# same bytes from media.githubusercontent.com/media/<owner>/<repo>/<ref>/<path>,
# which needs no LFS client, no authentication and no full clone (the full clone is
# 7.55 GB because it carries video too; AudioWAV/ alone is ~600 MB).
#
# Usage:
#   scripts/fetch_crema.sh [DEST]            # default: /home/vct/data/crema_audio
#   LIMIT=20 scripts/fetch_crema.sh /tmp/x   # smoke test: first 20 files only
#
# Resumable: a file already on disk with a readable 16 kHz header is skipped, so an
# interrupted run is restarted by re-running the same command. Every download is
# validated before it is moved into place; nothing half-written survives.

set -euo pipefail

export DEST=${1:-/home/vct/data/crema_audio}
export LIMIT=${LIMIT:-0}
export PARALLEL=${PARALLEL:-24}
export REPO=${REPO:-CheyneyComputerScience/CREMA-D}
export REF=${REF:-master}

mkdir -p "$DEST"

echo "== listing AudioWAV/ from the git tree (no LFS objects fetched)"
export LIST
LIST=$(mktemp)
curl -sfL "https://api.github.com/repos/${REPO}/git/trees/${REF}?recursive=1" \
  | python3 -c '
import json, sys
tree = json.load(sys.stdin)
paths = [e["path"] for e in tree.get("tree", [])
         if e["path"].startswith("AudioWAV/") and e["path"].endswith(".wav")]
if not paths:
    sys.exit("no AudioWAV/*.wav entries in the tree response")
print("\n".join(sorted(paths)))
' > "$LIST"
trap 'rm -f "$LIST"' EXIT

if [ "$(wc -l < "$LIST")" -eq 0 ]; then
  echo "no files listed; the GitHub tree API answered without AudioWAV/"
  exit 1
fi
echo "   $(wc -l < "$LIST") file(s) in the release"
if [ "$LIMIT" -gt 0 ]; then
  head -n "$LIMIT" "$LIST" > "$LIST.head" && mv "$LIST.head" "$LIST"
  echo "   LIMIT=$LIMIT -> $(wc -l < "$LIST") file(s) this run"
fi

echo "== downloading what is missing or unusable"
python3 - <<'PY'
import concurrent.futures as cf
import os
import sys
import urllib.request
import wave

dest = os.environ["DEST"]
repo = os.environ["REPO"]
ref = os.environ["REF"]
parallel = int(os.environ["PARALLEL"])
paths = [p for p in open(os.environ["LIST"], encoding="utf-8").read().split() if p]

def usable(target: str) -> bool:
    try:
        with wave.open(target, "rb") as fh:
            return fh.getframerate() == 16000 and fh.getnframes() > 0
    except Exception:
        return False

todo = [p for p in paths if not usable(os.path.join(dest, os.path.basename(p)))]
print(f"   {len(paths) - len(todo)} already present and valid, "
      f"{len(todo)} to fetch", flush=True)
if not todo:
    sys.exit(0)

def fetch(path: str) -> str | None:
    """Download one file; one retry, 45 s per attempt.

    A stalled connection on a 7442-file fetch must cost seconds, not minutes: the
    timeout is per attempt and a failed file is reported rather than retried
    forever, because re-running the script resumes from what is on disk.
    """
    name = os.path.basename(path)
    target = os.path.join(dest, name)
    tmp = os.path.join(dest, f".{name}.part")
    url = f"https://media.githubusercontent.com/media/{repo}/{ref}/{path}"
    last: Exception | None = None
    for _attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=45) as response, open(tmp, "wb") as out:
                out.write(response.read())
            if not usable(tmp):
                os.unlink(tmp)
                return f"BAD {name}: not a readable 16 kHz wav"
            os.replace(tmp, target)
            return None
        except Exception as exc:
            last = exc
            if os.path.exists(tmp):
                os.unlink(tmp)
    return f"FAIL {name}: {last}"

failures = []
with cf.ThreadPoolExecutor(max_workers=parallel) as pool:
    for i, problem in enumerate(pool.map(fetch, todo), 1):
        if problem:
            failures.append(problem)
            print(f"  {problem}", flush=True)
        elif i % 500 == 0:
            print(f"   {i}/{len(todo)} fetched", flush=True)
if failures:
    print(f"{len(failures)} file(s) failed; re-run the same command to retry them")
    sys.exit(1)
PY

echo "== verifying the whole directory"
python3 - "$DEST" <<'PY'
import glob
import os
import sys
import wave

files = sorted(glob.glob(os.path.join(sys.argv[1], "*.wav")))
bad, seconds = [], 0.0
for path in files:
    try:
        with wave.open(path, "rb") as fh:
            assert fh.getframerate() == 16000
            seconds += fh.getnframes() / fh.getframerate()
    except Exception as exc:
        bad.append((os.path.basename(path), str(exc)[:60]))
print(f"{len(files)} wav(s), {seconds / 3600:.2f} h of audio, {len(bad)} unusable")
for name, why in bad[:10]:
    print(f"  BAD {name}: {why}")
sys.exit(1 if bad else 0)
PY

echo "== done: $DEST"
