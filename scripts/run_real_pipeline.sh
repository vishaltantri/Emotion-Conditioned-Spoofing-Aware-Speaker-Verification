#!/usr/bin/env bash
# End-to-end real-data pipeline: manifests -> scores -> fusion evaluation.
#
# Assumes the corpora are already on disk (see docs/PROGRESS_LOG.md for the
# download campaign): ESD English under $D/esd and EmoSpoof-TTS generators under
# $D/emospoof. Every stage is resumable; re-running skips completed work.
#
# Env overrides: DATA_ROOT (default /home/vct/data/easv), DEVICE (cuda|cpu),
#                OUT (evaluation output dir), CM (indomain|aasist)
set -euo pipefail
D="${DATA_ROOT:-/home/vct/data/easv}"
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$P/.venv/bin/python"
PYRM="$P/.venv-rm/bin/python"
DEVICE="${DEVICE:-cuda}"
CM="${CM:-indomain}"
EVAL_OUT="${OUT:-$P/results/real/run_$CM}"
export HF_HOME="$D/hf_cache"

mkdir -p "$P/results/e2e" "$D/real"

say() { echo "[$(date '+%F %T')] $*"; }

say "1/7 manifests"
"$PY" "$P/scripts/build_manifests.py" esd --esd-root "$D/esd" \
      --out "$P/results/e2e/m_esd_real.csv"
"$PY" "$P/scripts/build_manifests.py" emospoof --emo-root "$D/emospoof" \
      --out "$P/results/e2e/m_emospoof.csv"
"$PY" "$P/scripts/merge_manifests.py" --esd "$P/results/e2e/m_esd_real.csv" \
      --emospoof "$P/results/e2e/m_emospoof.csv" \
      --out "$P/results/e2e/merged_manifest.csv"
MERGED="$P/results/e2e/merged_manifest.csv"

say "2/7 audit (hard stop on errors)"
"$PY" -m easv.cli audit --manifest "$MERGED" --out "$P/results/e2e/audit_real"

say "3/7 ASV scores"
"$PYRM" "$P/scripts/producer_asv.py" --manifest "$MERGED" \
        --out "$D/real/scores_asv_merged.csv" --device "$DEVICE"

say "4/7 SER scores"
"$PYRM" "$P/scripts/producer_ser.py" --manifest "$MERGED" \
        --out "$D/real/scores_ser_merged.csv" --device "$DEVICE"

say "5/7 countermeasure scores ($CM)"
if [[ "$CM" == "aasist" ]]; then
  "$PY" "$P/scripts/producer_cm_aasist.py" --manifest "$MERGED" \
        --out "$D/real/scores_cm_merged.csv" --aasist-root "$D/aasist" --device "$DEVICE"
  CM_FILE="$D/real/scores_cm_merged.csv"
else
  "$PY" "$P/scripts/train_cm_indomain.py" --manifest "$MERGED" \
        --out-dir "$D/real/cm_indomain" --device "$DEVICE"
  CM_FILE="$D/real/cm_indomain/scores_cm_indomain.csv"
fi

say "6/7 combine scores"
"$PY" "$P/scripts/combine_scores.py" --speaker "$D/real/scores_asv_merged.csv" \
      --cm "$CM_FILE" --emotion "$D/real/scores_ser_merged.csv" \
      --out "$D/real/scores_combined_$CM.csv"

say "7/7 fusion evaluation + ASV summary"
"$PY" -m easv.cli evaluate --manifest "$MERGED" \
      --scores "$D/real/scores_combined_$CM.csv" \
      --config "$P/configs/real_run.yaml" --out "$EVAL_OUT"
"$PY" "$P/scripts/summarize_real_scores.py" --manifest "$MERGED" \
      --scores "$D/real/scores_asv_merged.csv" --out "$P/results/real/asv_summary"

say "done: $EVAL_OUT/results.md"
