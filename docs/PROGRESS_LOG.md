# Execution progress log (2026-10-21 → 2026-10-22)

This log complements [REMAINING_WORK_PLAN.md](REMAINING_WORK_PLAN.md): it records
what has actually been executed since the initial evaluation-core commit, what is
verified, what is running, and what is still blocked. Working data root
(`/home/vct/data/easv`, ~250 GB volume) is outside this repository; manifests that
exist so far live in `results/e2e/` and are deliberately uncommitted.

## Status snapshot

| stage | status |
|---|---|
| Evaluation core (manifest/audit/fusion/metrics/CLI, 66 tests) | **done, committed** |
| Prior-art novelty gate | **closed (conditional GO)** — [NOVELTY_MATRIX.md](NOVELTY_MATRIX.md) |
| Course-report deliverable (DOCX, 19 pages, synthetic-only claims) | **done** — [Course_Project_Report_EASV_Audited.docx](../Course_Project_Report_EASV_Audited.docx), audit record [WORK_COMPLETION_AUDIT.md](../WORK_COMPLETION_AUDIT.md) |
| Offline wheel stack for `.venv-rm` (torch 2.10+cu13, torchaudio, transformers, speechbrain… **69 wheels**) | **done** — recovery pass finished, offline install ran; import verify mechanism in `scripts/wheels_remaining.sh` |
| ESD English subset (HF mirror `sonchuate/ESD_dataset`, speakers 0011–0020) | **done** — 17,660 real trials extracted + manifested |
| ESD manifest built + audited | **done** — `results/e2e/m_esd_real.csv` (17,660 rows), `easv audit` clean |
| ASV producer (SpeechBrain ECAPA-TDNN embeddings, cosine) | **done on real ESD audio** → `real/scores_asv_esd.csv` |
| SER producer (wav2vec2-base-superb-ER softmax → emo_* cols) | **done on real ESD audio** → `real/scores_ser_esd.csv` |
| EmoSpoof-TTS download (StyleTTS2 2.9 GB → F5-TTS 1.1 GB → CosyVoice 2.7 GB) | **running** — single-stream md5-gated rescue (see below) |
| ASVspoof 2019 LA download (7.6 GB, selective train/dev/protocols extract) | **queued** after EmoSpoof (stream 4) |
| AASIST CM training | **blocked** on LA audio |
| EmoSpoof manifest + merged manifest + audit | **blocked** on EmoSpoof extraction |
| Fusion on real scores (RSVs + ASV + SER integration) | blocked on EmoSpoof spoof rows |

## What was executed in this pass

1. **Wheels / offline environment.** 69 wheels fetched through the segmented
   downloader (`scripts/fetch_wheels.sh`, `scripts/segdl.sh`); the offline
   install ran and the `VENV_RM_READY` marker was placed. One install failure was
   logged and resolved in the next pass; `scripts/wheels_remaining.sh` now
   verifies `import torch, torchaudio, transformers, soundfile, speechbrain`
   before declaring the venv ready.
2. **Real ESD data.** English subset downloaded via the public HF mirror,
   extracted, and moved through `scripts/build_manifests.py esd`. Result: a
   17,660-trial manifest (4 emotions × 10 speakers) that passes `easv audit`.
3. **Real score producers.** `scripts/producer_asv.py` (ECAPA-TDNN speaker
   embeddings, cosine scoring, CUDA) and `scripts/producer_ser.py`
   (wav2vec2-base-superb-ER → SOFTMAX → `emo_*` columns) both completed on the
   full 17,660-trial manifest, producing real per-trial scores:
   `real/scores_asv_esd.csv`, `real/scores_ser_esd.csv` (both 17,660 rows,
   resumable `_part` slice design in the producers).
4. **Dataset download campaign (long-running).** Zenodo-style segmented
   downloading repeatedly failed:
   * `curl --retry` restarts a whole range and appends duplicated bytes (parts
     grew past their expected size) — fixed by removing `--retry` from
     `scripts/segdl.sh` (the outer loop retries instead);
   * the same byte range began returning *different* content over time (a
     5,743-member CRC diagnostic traced to mixing downloads from server-side
     replica flips / re-uploaded file versions — byte-flapping was verified by
     direct fetched-vs-local comparisons at fixed offsets);
   * the record exposes an authoritative MD5, so byte-level truth is now
     checksums, not range probes.
   Outcome: **segmented fetching on this record is unreliable and abandoned**;
   `scripts/dataset_rescue.sh` re-downloads EmoSpoof-TTS archives as a single
   stream with MD5 verification + retries with a cache-buster, and
   `scripts/datasets_stream4.sh` waits for the StyleTTS2 marker and continues
   with F5-TTS, CosyVoice and ASVspoof 2019 LA afterwards. Both runs detached and
   union-kill-safe.
5. **Infra hygiene fixed in this pass.** `scripts/auto_pipeline.sh` and the venv
   guard were reconciled; overlapping producers that had double-written score CSVs
   were killed; clean rebuilds of SER/ASV completed without stale headers.

## Root-cause lessons recorded for reference

* Never trust a resumed range download without a byte-level checksum; Zenodo's
  MD5 for each file's content is in the API record (`/api/records/…/files`).
* A producer that writes scores to CSV without a final-state marker can be
  relaunched while the previous iteration is still writing — always audit for
  extra header rows before consuming a score CSV.
* On the 250 GB data volume the disk is 12 GiB free today — EmoSpoof
  extract-then-delete remains gated at one corpus at a time.

## What still blocks the "one real result"

1. The LA splits (LA train for CM training; eval protocol as needed) — the 7.6 GB
   Edinburgh zip is queued post-EmoSpoof.
2. An ASV/CM/SER integration pass (fusion evaluation with real SASV-EER style
   reporting against the ESD-manifest + spoof rows).
3. The AASIST training recipe (`scripts/EASIST_LA.conf`) cannot start until LA
   train audio is extracted.

Recorded by the agent working on the owner's behalf; numbers quoted are read off
`wc -l` of the manifest/score files above (17661 lines including header) and the
running-process list at the time of writing.
