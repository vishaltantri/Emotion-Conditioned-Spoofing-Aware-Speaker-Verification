#!/usr/bin/env bash
# Auto-pipeline state machine. Polls ~/.collect markers in /home/vct/data/easv/status.log
# and fires each stage exactly once when its inputs land. Every stage logs to
# status.log; transcripts go to $STAGE/*.log so a failed stage is re-runnable.
set -u
D=/home/vct/data/easv
PROJ="/home/vct/Desktop/AI project"
LOG="$D/status.log"
PYRM="$PROJ/.venv-rm/bin/python"
PY="$PROJ/.venv/bin/python"
EURV="$PROJ/.venv/bin/easv"
say() { echo "[$(date '+%F %T')] auto: $*" >> "$LOG"; }
marker(){ grep -q "$1" "$LOG" 2>/dev/null; }
done_flag="$(mktemp -d)"   # stage flags as files in a private dir

mkdir -p "$D/real"

# ---- Stage 1: EmoSpoof manifest once all three generators are extracted ----
(
  while :; do
    if marker "StyleTTS2.zip extracted+removed" && marker "F5TTS.zip extracted+removed" \
       && marker "CosyVoice.zip extracted+removed"; then break; fi
    sleep 60
  done
  say "auto stage1: emospoof extracted -> building manifest"
  "$PY" "$PROJ/scripts/build_manifests.py" emospoof \
    --emo-root "$D/emospoof" --out "$D/real/m_emospoof.csv" >> "$D/real/stage1.log" 2>&1 \
    && say "auto stage1: emospoof manifest OK"
  # merge ESD + EmoSpoof first (LA manifest may still be missing)
  "$PY" "$PROJ/scripts/merge_manifests.py" --esd "$PROJ/results/e2e/m_esd_real.csv" \
    --emospoof "$D/real/m_emospoof.csv" --out "$D/real/merged_force_v1.csv" >> "$D/real/stage1.log" 2>&1 \
    && say "auto stage1: partial merge OK (esd+emospoof)"
) & S1=$!

# ---- Stage 2: producers once wheels are installed AND merged manifest exists
(
  while :; do
    marker VENV_RM_READY && [[ -f "$D/real/merged_force_v1.csv" ]] && break
    sleep 60
  done
  say "auto stage2: venv ready + manifest -> producers"
  cd "$PROJ"
  # guard: ensure torch import works (install failure marker otherwise)
  if ! "$PYRM" -c "import torch, torchaudio, transformers, soundfile, speechbrain" \
      >> "$D/real/stage2_guard.log" 2>&1; then
    say "auto stage2: venv broken; skipped producers (see stage2_guard.log)"
    exit 1
  fi
  "$PYRM" "$PROJ/scripts/producer_asv.py" --manifest "$D/real/merged_force_v1.csv" \
    --out "$D/real/scores_asv.csv" --device cuda >> "$D/real/stage2_asv.log" 2>&1 \
    && say "auto stage2: ASV producer done"
  "$PYRM" "$PROJ/scripts/producer_ser.py" --manifest "$D/real/merged_force_v1.csv" \
    --out "$D/real/scores_ser.csv" --device cuda >> "$D/real/stage2_ser.log" 2>&1 \
    && say "auto stage2: SER producer done"
) & S2=$!

# ---- Stage 3: AASIST training once LA is extracted and venv works ------------
(
  while :; do
    marker "LA train+dev+protocols extracted" && marker VENV_RM_READY && break
    sleep 60
  done
  say "auto stage3: LA extracted -> building ASVspoof manifest + launching AASIST"
  "$PY" "$PROJ/scripts/build_manifests.py" asvspoof --la-root "$D/asvspoof2019LA/LA" \
    --out "$D/real/m_asvspoof.csv" >> "$D/real/stage3.log" 2>&1
  "$PY" "$PROJ/scripts/merge_manifests.py" --asvspoof "$D/real/m_asvspoof.csv" \
    --esd "$PROJ/results/e2e/m_esd_real.csv" --emospoof "$D/real/m_emospoof.csv" \
    --out "$D/real/merged_final.csv" >> "$D/real/stage3.log" 2>&1 \
    && say "auto stage3: final merge OK"
  "$EURV" audit --manifest "$D/real/merged_final.csv" --out "$D/real/audit_final" \
    >> "$D/real/stage3_audit.log" 2>&1 && say "auto stage3: final audit written"
  cp "$PROJ/scripts/EASIST_LA.conf" "$D/aasist/config/EASIST_LA.conf"
  cd "$D/aasist"
  # torchcontrib is sdist-only: install it first, then launch training headless
  "$PROJ/.venv-rm/bin/python" -m pip install --no-index \
    --find-links /home/vct/data/easv/wheels torchcontrib \
    >> "$D/real/stage3_tc.log" 2>&1 || \
    "$PROJ/.venv-rm/bin/python" -m pip install torchcontrib \
    >> "$D/real/stage3_tc.log" 2>&1
  nohup "$PROJ/.venv-rm/bin/python" main.py --config config/EASIST_LA.conf \
    --output_dir "$D/aasist/exp_easist" --seed 1234 \
    >> "$D/real/aasist_train.log" 2>&1 &
  say "auto stage3: AASIST training launched (pid $!)"
) & S3=$!

wait $S1 $S2 $S3
say "auto pipeline: all stages settled"
