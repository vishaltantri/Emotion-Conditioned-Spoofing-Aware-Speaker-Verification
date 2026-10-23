# Execution progress log (2026-10-21 → 2026-10-23)

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
| EmoSpoof-TTS download (StyleTTS2 2.9 GB → F5-TTS 1.1 GB → CosyVoice 2.7 GB) | **done** — all three generators on disk, 12,000 wavs each (StyleTTS2 via the Hugging Face parquet mirror) |
| EmoSpoof manifest + merged manifest + audit | **done** — 36,000 EmoSpoof rows, 53,900-trial merged manifest, `easv audit` 0 errors / 4 warnings |
| CUDA torch in the audio venv | **done** — `2.14.1+cu130`, `cuda.is_available()` True (RTX 3050 6 GB) |
| ASV scores on the merged manifest | **done** — 53,900 trials, `real/scores_asv_merged.csv` |
| ASVspoof 2019 LA corpus | **done** — 25,380 train utterances from the HF parquet mirror (`la_train.parquet`, 1.59 GB) + official CM protocols |
| AASIST CM training | **done** — LA-pretrained checkpoint scored (50.8 %), and AASIST retrained on LA train audio (see Pass 3) |
| Fusion on real scores | **done** — four countermeasure regimes, `results/real/{run1,run2_oodcm,run3_crossgen,run4_3gen_crossgen}` |

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

A third regime closes the generalisation question the two-generator corpus can
answer: `scripts/train_cm_indomain.py --holdout-attack f5_tts` excludes one
generator from training *and* from dev-based early stopping, so every F5-TTS
clone is scored as an unseen attack. That CM reaches 3.19 % EER on the generators
it trained on and **6.27 % on the held-out one** (overall 4.28 %), against 50.8 %
for AASIST-LA: generator-adjacent training data transfers far better than the
ASVspoof lineage. Fusion under this CM lands at 4.11 % (linear), 3.81 %
(interaction, +0.38 pts, p<0.001) and 3.93 % (gated, +0.27 pts, CI −0.23–0.86,
p=0.23), so the proposed arm still does not clear the significance bar.

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
* `scripts/dataset_rescue.sh` — downloads now resume across attempts
  (`curl -C -` into a stable `.download` file with stall detection) instead of
  restarting from zero: Zenodo truncated this 2.9 GB archive twice, at 1.24 GB and
  at 1.45 GB, so the previous logic could burn an hour and keep nothing. A partial
  is still deleted whenever the cumulative file fails the md5, because bytes from
  mixed server replicas can never verify.
* `scripts/train_cm_indomain.py` — `--holdout-attack` excludes a generator from
  training and from dev-based early stopping, turning the run into a
  leave-one-generator-out generalisation test; the held-out EER is computed
  against the bona fide trials of the same split (an EER inside a spoof-only
  subset is undefined and raised).

Recorded by the agent working on the owner's behalf; numbers quoted are read off
`wc -l` of the manifest/score files above (17661 lines including header) and the
running-process list at the time of writing.

## Pass 3 (2026-10-23): three-generator corpus, ASVspoof LA, retrained countermeasure

### How the two blocked corpora actually arrived

Both downloads were capped at ~200 KB/s by their origin servers (Zenodo for
EmoSpoof, Edinburgh DataShare for the 7.6 GB LA zip), which is roughly twelve
hours per corpus. Public Hugging Face mirrors serve the same content at
1.2–2 MB/s per connection and ~2 MB/s aggregate with four parallel streams, so
both corpora were fetched from the mirrors instead
([scripts/hf_fetch.sh](../scripts/hf_fetch.sh)):

* EmoSpoof-TTS — 36,000 utterances across StyleTTS2, F5-TTS and CosyVoice in the
  mirror's parquet shards. The `path` column preserves the original
  `Generator/speaker/Emotion/utterance.wav` tree, so the StyleTTS2 subset
  (shards 0–2) writes back to the layout `build_manifests.py emospoof` already
  reads.
* ASVspoof 2019 LA — only the train partition is mirrored
  (`la_train.parquet`, 1.59 GB): 2,580 bona fide + 22,800 spoof utterances, which
  is all an in-domain AASIST retrain needs, plus the official CM protocol files
  from the DataShare archive. The eval partition was not needed and not fetched,
  because the EASV corpus is the target domain and its own dev split selects
  everything.

[scripts/extract_hf_corpora.py](../scripts/extract_hf_corpora.py) converts the
parquet back to the on-disk layouts, skipping files that already exist so an
interrupted conversion resumes instead of restarting.

### A real bug the LA data exposed

`parse_cm_protocol_line` assumed the protocol column order
`speaker - utterance label attack`, which is the order used in the *ASVspoof 2019
evaluation plan* description but **not** in the distributed CM files. The real
line is `LA_0079 LA_T_1138215 - - bonafide`, i.e. `speaker utterance - system
label`. The old parser therefore returned `-` as every utterance id, so the
builder pointed at `-.flac` for the whole corpus. The parser now locates the
label field instead of trusting a position, keeps the attack id, and is pinned by
[tests/test_build_manifests.py](../tests/test_build_manifests.py); the older
synthetic fixture was corrected to the official column order as well.

### Three-generator corpus and the cross-generator regime

The merged manifest is now **53,900 trials** (17,900 ESD bona fide + 36,000
clones, 12,000 per generator) with a 4/2/4 speaker split; `easv audit` reports
0 errors and the same 4 expected warnings (cross-split nontarget derivations and
all three generators present in every split). ASV and SER were rescored on the
full manifest (53,900 rows each, 0 missing).

The countermeasure for the headline regime is trained with StyleTTS2 excluded
from training *and* from dev-based early stopping
(`train_cm_indomain.py --holdout-attack styletts2`, 21,560 training trials). It
reaches **1.60 % SPF-EER overall and 2.50 % on the unseen StyleTTS2 clones** —
the third generator is detected at the same order of magnitude as the seen ones,
which is the strongest countermeasure result in the project.

Fusion on that corpus (`results/real/run4_3gen_crossgen`, thresholds from dev
only): 27.12 % (speaker only), 2.27 % (linear), 2.51 % (additive), 2.09 %
(interaction), **1.54 % (gated)** SASV-EER. The gated arm beats linear by
+0.68 pts (0.15–1.12, p=0.002); a paired contrast that the CLI does not run by
default ([scripts/compare_arms.py](../scripts/compare_arms.py)) shows it also
beats the interaction control by **+0.48 pts (0.06–0.99)** and the capacity
control by +0.85 pts. So the proposed arm *is* the best model on the completed
design — with one caveat that undercuts the mechanism claim: its learned gate
weights are flat across emotions (speaker 4.02–4.05, CM 9.69–9.73; in the
in-domain regime the five per-emotion weights agree to three decimals). The arm
wins because of its decision function and calibration, not because it conditions
on emotion.

The ASV vulnerability also turned out to be generator-specific: at the dev
threshold the ECAPA front-end accepts 59.0 % of CosyVoice clones and 51.5 % of
F5-TTS clones but only **5.9 %** of StyleTTS2 clones (mean speaker score +0.256
against +0.43–0.46), so the 38.8 % aggregate is not a property of "neural
clones" in general. Audio for the StyleTTS2 subset was checked (16 kHz, ~2.8 s,
same speaker distribution), so this is a property of that generator's output.

### AASIST on ASVspoof 2019 LA

Two attempts, and the first one is worth recording because it produced a
plausible-looking but meaningless number:

1. **Selecting the checkpoint on EASV dev.** The first run trained on LA train
   and early-stopped on the EASV dev split (the target domain). Dev EER was best
   at epoch 1 (39.8 %) and got worse, so "the best model" was an essentially
   untrained network and its 40.8 % test EER measured the domain gap, not the
   countermeasure. The artifacts are kept in
   `real/cm_aasist_la_ongenv_selected` for reference.
2. **Selecting on held-out LA speakers** (the reported run): 2 of 20 LA speakers
   (2,576 utterances) are held out and only they choose the checkpoint; EASV dev
   and test are then scored once as *transfer* numbers. Batch 24 OOMs on the
   6 GB card, so the run uses batch 12 with bf16 autocast.

Result: the retrained AASIST reaches **2.11 % EER on the held-out LA speakers**
(epoch 3 of 3, 16,000 training utterances; epoch 1 was 9.47 %, epoch 2 2.97 %,
so the model was still improving when the budget ran out). It is therefore a
working LA countermeasure by any reasonable reading. On the EASV corpus the same
model scores **49.4 % EER on dev and 43.8 % on test** — chance level, like the
pretrained checkpoint's 50.8 %. The conclusion is unchanged and now has two
supports: ASVspoof-2019 supervision, pretrained or retrained, does not describe
these neural clones, whereas a log-mel countermeasure trained on two adjacent
EmoSpoof generators detects the third at 2.50 % EER.

No fusion regime was added for this countermeasure. With a chance-level CM the
arm ladder reduces to Regime B (already reported with the pretrained
checkpoint), and the EASV train-split scores needed for fitting the fusion arms
were deliberately not produced: the component result answers the question the
run was for.

### What is left

Nothing in the planned scope. The two follow-ups that would change conclusions
are both ablations, not missing data: replacing the emotion features with a
single SER-confidence scalar (to test whether the Regime B gain survives), and a
capacity-matched gate (to test whether the Regime D gain is architecture or
capacity). Both need a new frozen config and a fresh test report; neither is
required to state the results above.

## Audit pass (2026-10-23): what is actually open

Code, corpora and the reported evaluation are complete for the current design.
The audit found stale documentation more often than missing work, and both are
listed here so the next reader does not have to reconstruct the state.

Documentation corrected in this pass: the report abstract still claimed that no
real corpus, backend or real result existed (rewritten with the measured numbers);
section 4.1 still quoted 65 tests; the appendix listed only synthetic commands;
README's "not implemented" list contradicted the rest of the file; the novelty
matrix header still said every cell was an unverified TODO; the GCCE 2023 record
had lost its heading and was merged into the Zhao entry; PROTOCOL section 6 still
promised a pre-registration that never happened.

Open items, ordered by how much they would change a conclusion:

1. **Pre-registration was never done.** PROTOCOL section 6 now records the rules
   that were applied and marks the numeric margins and the generalisation track as
   post hoc. A future write-up needs a frozen decision table before the next test
   report, otherwise the same criticism applies again.
2. **Real-data ablations.** `ablation_no_emotion_uncertainty.yaml` and
   `ablation_target_far_threshold.yaml` exist but have only been run on the
   synthetic demo. Two `easv evaluate` invocations would close both
   plan-required ablations on the 53,900-trial corpus.
3. **No held-out-emotion track.** Every emotion condition appears in the training
   split, so the per-emotion tables are descriptive. A cross-emotion claim needs
   an emotion-held-out manifest; none exists.
4. **Single seed, single split.** Fusion fits use seed 7 and one 4/2/4 speaker
   split. Run-to-run and split-to-split variation are unreported.
5. **LA validation is thin.** The retrained AASIST held out two LA speakers and
   ran three epochs (still improving: 9.47 % -> 2.97 % -> 2.11 % dev EER). The
   LA evaluation partition was never fetched, so there is no standard LA eval
   number for it.
6. **No fusion regime under the LA-retrained CM**, because its EASV train-split
   scores were not produced. With a chance-level countermeasure the ladder reduces
   to Regime B, which is reported, but the regime itself is absent from the tables.
7. **The StyleTTS2 ASV anomaly is unexplained.** Only 5.9 % of its clones are
   accepted by the speaker front-end against 51.5-59.0 % for the other two
   generators. Audio properties were checked; per-generator speaker-identity
   recovery was not measured.
8. **No published-baseline reproduction**, so the metric code is validated by unit
   tests only, not against reference numbers from SASV 2022 or ASVspoof.
9. **Reproducibility loose ends.** `run_real_pipeline.sh` does not encode the
   headline holdout regime or the LA training; `results/` is git-ignored; there is
   no corpus checksum/licence table; the repository has no LICENSE file and no CI.

### Documentation truth pass (same day, second pass)

Every document was re-read against the executed state and corrected where it was
false, not just stale-in-tone:

* **Report DOCX.** Abstract replaced (the old one stated that no real corpus,
  backend or result existed) with a 183-word summary of the measured findings, still
  inside the 150-200 word requirement. Section 4.1 now reports 79 tests and points to
  the real execution; the appendix lists the commands that produced the reported
  numbers instead of only the synthetic demo; the conclusion and future-work
  paragraphs describe what is actually open (held-out emotion, pre-registration,
  ablations, seeds, published baseline, LA eval partition, licence). A duplicated
  cross-generator paragraph left by a guard bug was removed. The contents table was
  a static table, not a Word field, so its page numbers had gone stale when the
  real-data text was added; chapters 5, references and appendix were re-pointed from
  the added word count (~5 pages for chapter 4). **These three numbers are estimates
  until the file is opened in Word and the pagination confirmed.**
* **README.md.** "Not implemented" listed things that are now implemented (real data,
  ECAPA/AASIST training) and omitted the things that actually remain; rewritten,
  including a licence caveat and the correct two-environment requirement.
* **docs/IMPLEMENTATION_PLAN.md** and **docs/REMAINING_WORK_PLAN.md** were rewritten
  from "not started" plans into status documents: what each stage produced, its
  measured outcome, and what is still open. The old plan's wrong assumption that a
  Hugging Face mirror resolves the ESD licence was corrected.
* **docs/NOVELTY_MATRIX.md** header no longer claims every cell is an unverified TODO;
  the checklist now records which gate items are closed and which are not, and the
  GCCE 2023 record (which had lost its heading and merged into the Zhao entry) was
  repaired.
* **directory READMEs.** `results/` documents every real run and its artifacts (not
  only the demo); `data_manifests/` lists the manifests actually built, the 0-error
  audit and the licence/checksum gaps; `paper/` and `figures/` state that the course
  report is the current write-up and that no manuscript or figure set exists.
* **WORK_COMPLETION_AUDIT.md** is now explicitly labelled as the 2026-10-21 snapshot,
  with the superseded claims marked closed.
* **script docstrings and generated text.** `summarize_real_scores.py` no longer emits
  "no countermeasure scores exist yet" into its own report; the countermeasure and
  in-domain trainer docstrings no longer say the LA corpus cannot be extracted or that
  cross-generator evaluation is untestable; `run_real_pipeline.sh` states that it does
  not build the headline regime; `configs/real_run.yaml` records that only its
  provenance text was edited after the run, and the SASV template is marked unused.

Nothing in `src/easv/` changed: its docstrings already described the score-level
contract accurately, and all 79 tests still pass.

## Pass 5 (2026-10-23): SER-confidence ablation — the mechanism question closed

Publication deferred to a workshop-first schedule ([PUBLICATION_ACTION_PLAN.md](PUBLICATION_ACTION_PLAN.md);
ESD licence demoted to a courtesy email after the licence research: EmoSpoof-TTS
is CC BY 4.0 with no ESD audio inside, ESD's own terms state research use is free).
First experiment: the spoof-cue ablation.

**Implementation.** `training.emotion_features: confidence_only` in the evaluation
config (`src/easv/evaluation.py`): the emo_* vector collapses to a single scalar,
the SER's maximum predicted probability, and uncertainty features are disabled for
that mode (entropy is undefined for a 1-of-1 taxonomy). Zero code change for the
parent runs — the mode is off by default, and the synthetic demo with the mode on
smoke-tests in seconds (additive falls from 12 to 4 features, gated from 60 to 23).
`scripts/compare_arms.py` gained `--results-b` so two runs can be paired against
each other; that exposed and fixed a real bug — same-named arms collided in the
score dict, silently comparing arm b against itself (+0.000 everywhere).

**Runs.** `results/real/run2b_serconf` (Regime B, AASIST-LA failing CM) and
`results/real/run4b_serconf` (Regime D, three-generator cross-gen). Sanity check
passed: emotion-free arms reproduce the parent runs to the fourth decimal, so only
the feature construction differs.

**Outcome, both directions satisfied:**

* **Regime D (the headline run): the mechanism claim is dead.** The gated arm's
  edge over linear fusion reproduces *entirely* with the scalar alone: 1.49 % vs
  1.54 % SASV-EER, cross-run contrast +0.02 pts (CI −0.04…+0.09, p=0.76) —
  indistinguishable. A one-dimensional input cannot do emotion conditioning, so
  the +0.7-point win is capacity and calibration on one spoof-correlated feature.
  Two independent measurements now refute the mechanism (flat per-emotion gate
  weights; scalar-only reproduction). The interaction arm is different: it does
  need the full vector (scalar-only −0.19 pts, p=0.046), which separates the two
  emotion arms' behaviour.
* **Regime B: the +5.5-point emotion gain is one-third spoof cue, but not only.**
  The scalar recovers +1.6 points (CI 0.70–2.74, p<0.001); the full vector is
  significantly better than the scalar (additive −3.68 pts, CI −7.68…−0.78).
  So a third of the gain is the "confidently wrong SER" spoof cue, and the rest is
  the *per-class* structure of the out-of-domain SER output — still not
  demonstrated emotion semantics. Regime B is unusable as evidence for
  emotion-aware fusion in any form.

**Paper framing now fixed:** measurement/negative-result paper for a workshop; the
gated arm's Regime D win is reported as a calibrated decision function, never as
emotion conditioning. Full tables: [REAL_RESULTS.md](REAL_RESULTS.md), section
"SER-confidence ablation".
