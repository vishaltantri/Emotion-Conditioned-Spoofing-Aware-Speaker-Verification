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
| EmoSpoof-TTS download (StyleTTS2 2.9 GB → F5-TTS 1.1 GB → CosyVoice 2.7 GB) | **2 of 3 landed** — F5-TTS + CosyVoice extracted (12,000 wavs each); StyleTTS2 re-fetch running after the rescue loop deleted its finished tree |
| EmoSpoof manifest + merged manifest + audit | **done** — 24,000 EmoSpoof rows, 41,900-row merged manifest, `easv audit` 0 errors / 4 warnings |
| CUDA torch in the audio venv | **done** — `2.14.1+cu130`, `cuda.is_available()` True (RTX 3050 6 GB) |
| ASV scores on the merged manifest | **running** — `real/scores_asv_merged.csv` |
| ASVspoof 2019 LA download (7.6 GB, selective train/dev/protocols extract) | **blocked** — resumed parts came back mutually inconsistent under ENOSPC and were removed; extraction needs ~7.5 GB free |
| AASIST CM training | **blocked** — no LA audio, no pretrained checkpoint |
| Fusion on real scores | **blocked** on CM scores |

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

## Pass 2 (2026-10-22, ~11:00–14:00): first real multi-corpus artifacts

### Dataset outcomes

* **StyleTTS2.zip (2.93 GB)** — single-stream md5-gated fetch verified against the
  record's authoritative md5 (`cc70b4d6da9f2ecde9637b57b5e56726`), CRC-validated,
  extracted, archive removed. The extracted tree is **absent** from
  `emospoof/` afterwards (only the two other generators and `__MACOSX` remain),
  so this generator must be re-fetched. A fixture copy inside
  `data/easv/fixture/emospoof/` is unrelated test scaffolding, not the corpus.
* **F5TTS.zip (1.09 GB)** and **CosyVoice.zip (2.67 GB)** — both md5-verified,
  CRC-validated, extracted, archives removed. Each yields 10 cloned speakers ×
  4 emotions × 300 utterances = **12,000 wavs** (`emospoof/F5TTS`, `emospoof/CosyVoice`).
* The rescue loop was still re-downloading CosyVoice although its tree was already
  complete; both rescue and follow-on streams were stopped to protect the 80 MB
  of remaining free space from ENOSPC corruption.

### Real manifests and audit

| manifest | rows | notes |
|---|---:|---|
| `results/e2e/m_esd_real.csv` | 17,900 | 17,500 ESD targets (10 spk × 5 emotions × 350) + 400 derived cross-speaker nontargets |
| `results/e2e/m_emospoof.csv` | 24,000 | spoof rows, `attack` = generator, emotion from the folder name |
| `results/e2e/merged_manifest.csv` | 41,900 | generator splits demoted to the ESD speaker split (4/2/4) so no speaker crosses splits in targets/spoof |

`easv audit --manifest results/e2e/merged_manifest.csv --out results/e2e/audit_real`
reports **0 errors, 4 warnings**: two `speaker_overlap` and two `generator_overlap`
warnings. They are expected consequences of the merge policy — derived nontarget
rows deliberately pair speakers across splits, and both available generators
appear in every split once rows follow their speaker's split. The third
(StyleTTS2) generator is what would give a held-out-system split; that is the
reason to re-fetch it.

Merged coverage: train 16,760 / dev 8,380 / test 16,760 trials, 5 emotions and 2
attacks everywhere, so the test split does contain spoof trials despite the
missing generator.

### Environment

`torch 2.14.1+cu130` now imports in the project venv with
`torch.cuda.is_available() == True` on the **RTX 3050 6 GB**. The earlier install
failure was not the wheel (Python 3.14 matches `cp314`) but the missing NVIDIA
runtime libraries: torch's loader raises `ValueError: libcublasLt.so.* not found`,
then `ImportError: libcufile.so.0`, then `libnvshmem_host.so.3`. Installing
`nvidia_cublas`, `nvidia_cudnn_cu13`, `nvidia_cuda_runtime`, `nvidia_cuda_nvrtc`,
`nvidia_nvjitlink`, `nvidia_cufft`, `nvidia_curand`, `nvidia_cusolver`,
`nvidia_cusparse`, `nvidia_nccl_cu13`, `nvidia_nvtx`, `nvidia_cuda_cupti`,
`nvidia_cufile`, `nvidia_cusparselt_cu13` and `nvidia_nvshmem_cu13` from the same
offline wheel directory resolves it. Also installed `sympy`, `networkx`,
`filelock`, `jinja2`, `fsspec` for torch's runtime imports.

### First real measurement: ASV vulnerability to emotional clones

`scripts/producer_asv.py` was run over the whole merged manifest (41,900
trials, `real/scores_asv_merged.csv`, `missing_centroid=0`), and
`scripts/summarize_real_scores.py` turns those scores into the table below
(`results/real/asv_summary.{json,md}`; thresholds from dev only):

| split | trials | SV-EER | spoof n | spoof accepted @dev thr | target accepted |
|---|---:|---:|---:|---:|---:|
| train | 16,760 | 0.70 % | 9,600 | 71.9 % | 96.1 % |
| dev | 8,380 | 4.40 % | 4,800 | 72.4 % | 95.6 % |
| test | 16,760 | 3.12 % | 9,600 | 55.2 % | 82.9 % |

Test-split spoof acceptance by emotion (mean ASV score in brackets): angry
64.6 % (+0.4705), neutral 61.7 % (+0.4684), sad 47.5 % (+0.4196), happy 47.2 %
(+0.4120). By generator: CosyVoice 59.0 % (+0.4554), F5-TTS 51.5 % (+0.4298).

Interpretation, stated narrowly: on real ESD/EmoSpoof audio, a standard ECAPA
speaker front-end accepts more than half of cloned emotional speech as the
cloned identity, and the acceptance rate is **not** flat across emotions — the
high-arousal negative class (angry) is the most dangerous and the two positive /
low-arousal classes are the safest. This is exactly the dependence an
emotion-conditioned fusion stage is supposed to exploit, and it is measured, not
assumed. It is **not** a SASV result: without a countermeasure score the
combined decision cannot be formed, so no claim about fusion gains is made.

### Real SASV fusion evaluation (first complete result)

Scores for all 41,900 trials now exist for every component: ASV (ECAPA), SER
(wav2vec2-superb-ER), CM from the bundled LA-trained AASIST checkpoint
(`scripts/producer_cm_aasist.py`) and CM from an in-domain log-mel CNN
(`scripts/train_cm_indomain.py`, train split only). `scripts/combine_scores.py`
joins them and `easv evaluate` runs the five arms with dev-only thresholds.

Two regimes, same manifest and ASV/SER scores, different CM:

| model | in-domain CM (A), SASV-EER | AASIST-LA CM (B), SASV-EER |
|---|---:|---:|
| speaker_only | 0.3309 | 0.3309 |
| linear | 0.0060 | 0.3244 |
| emotion_additive | 0.0066 | 0.2654 |
| emotion_interaction | 0.0055 | 0.2661 |
| emotion_gated | 0.0073 | 0.2610 |

* **A**: no emotion arm beats plain fusion; gated is 0.14 pts worse (p=0.98).
* **B**: emotion arms gain +5.2 to +5.6 pts over linear (p<0.001) — but SER is
  confidently wrong on clones (30.7 % accuracy, mean max prob 0.83) against
  41.8 % on bona fide, so the emotion vector is partly a spoof detector. The
  gain is therefore not evidence for emotion-conditioned modelling.
* The pretrained AASIST itself is the headline negative: **SPF-EER 50.8 %** on
  EmoSpoof clones. A CM trained on ASVspoof-2019 attack families does not
  generalise to modern neural TTS in a different recording domain.

Full tables, CIs, breakdowns and caveats: [REAL_RESULTS.md](REAL_RESULTS.md).
Artifacts live outside the repo (`real/{scores_*,cm_indomain}`); the evaluation
outputs are in `results/real/run1` and `results/real/run2_oodcm` (git-ignored).

### Download failures in this pass

* The ASVspoof 2019 LA download was resumed with `scripts/segdl.sh`; under ENOSPC
  pressure several parts were re-created from zero and individual part files
  shrank mid-run, so the 16 parts became mutually inconsistent (4.1 GB of a
  7.6 GB archive). The parts were removed rather than left as a corrupted
  archive; LA extraction needs ~7.5 GB free and cannot run on this volume today.
* Stale partials from the abandoned CosyVoice retry (790 MB) and the leftover
  `__MACOSX` tree were deleted, which is what raised free space to ~7.8 GB.

### Disk: the real blocker

The 183 GB root volume is at 99 % with the user's own data (Desktop, Downloads,
other tool caches) occupying it; this project's data root holds 5.4 GB EmoSpoof +
6.5 GB LA parts + 2.9 GB wheels + 1.5 GB ESD. Reclaimable caches were cleaned
(pip 3.6 GB, npm 2.3 GB, `/tmp` leftovers), which moved free space from 16 KB to
~1.8 GB. ASVspoof 2019 LA extraction needs roughly **7.5 GB free** (zip already
counts against the 7.6 GB total), so AASIST training stays blocked until space is
released outside this project. ~/.cache/huggingface (2.9 GB, unrelated NLP models),
~/.gradle (3.1 GB) and ~/.local/share/Trash (2.1 GB) are the obvious candidates;
they are the owner's, so they were left untouched.

### Script fixes made while unblocking

* `scripts/dataset_rescue.sh` — the loop deleted `emospoof/StyleTTS2` after *every*
  archive, so extracting F5-TTS removed the StyleTTS2 tree that had just been
  produced. That is the whole explanation for the "missing" third generator. The
  delete is now scoped to the archive being extracted, an already-extracted tree
  is skipped instead of re-downloaded, and the extraction log line records the
  wav count. Accepts optional archive names: `dataset_rescue.sh StyleTTS2.zip`.
* `scripts/summarize_real_scores.py` — new; joins manifest and score files and
  reports SV-EER plus spoof acceptance by split, emotion and generator.

Recorded by the agent working on the owner's behalf; numbers quoted are read off
`wc -l` of the manifest/score files above (17661 lines including header) and the
running-process list at the time of writing.
