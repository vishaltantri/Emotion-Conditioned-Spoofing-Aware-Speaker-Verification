# Remaining work plan — what is left and in what order

Derived from the repository audit ([../WORK_COMPLETION_AUDIT.md](../WORK_COMPLETION_AUDIT.md)),
the implementation plan ([IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)) and the
protocol ([PROTOCOL.md](PROTOCOL.md)). This file is the actionable to-do list; it
restates what is done in one paragraph and then specifies every unfinished stage
with concrete steps, costs and decision points.

## Where the project actually stands

Done and verified (65 tests, synthetic `easv demo` end-to-end):

- manifest schema + leakage/coverage audits (`easv audit`);
- score alignment by trial_id, fusion ladder (`speaker_only`, `linear`,
  `emotion_additive`, `emotion_interaction`, `emotion_gated`) with JSON checkpoints;
- metric suite: SASV-EER, SV-EER, SPF-EER, min a-DCF, min t-DCF, FAR/FRR at a
  dev-selected threshold, Brier/ECE, grouped + paired bootstrap;
- CLI (`audit`, `evaluate`, `demo`, `predict`, `manifest-template`), ablation configs;
- prior-art matrix with four papers read/verified, method-distinction paragraph.

Not done — and this is the entire remaining project:

1. **No audio enters this codebase.** There is no ASV model, no countermeasure
   model, no speech-emotion model, no checkpoints. The pipeline starts from
   `speaker_score`, `cm_score`, `emo_*` columns in a scores CSV.
2. **No real datasets.** ASVspoof 2019/2021 LA, SASV 2022 protocol, EmoSpoof-TTS
   (and an emotion corpus) are not downloaded, licensed, or manifested.
3. **The only trained model in EASV — the fusion head — already trains on CPU in
   seconds.** "Models to be trained" really means *upstream score producers must
   either be pretrained checkpoints or be trained by us on ASVspoof data*, plus
   optional SER fine-tuning.
4. Pre-registered success-criteria table (PROTOCOL §6) is still TODO, and the
   novelty gate closes only after a real-data run.
5. Paper writing (Phase 6/7) not started.

Hardware reality check (measured on this machine):

| constraint | value | consequence |
|---|---|---|
| GPU | **GeForce RTX 3050 6GB Laptop GPU**, present but **broken**: userspace NVML 595.99.02 vs loaded kernel module 595.91.07 (`Driver/library version mismatch`) | a reboot should load the upgraded kernel module; if still mismatched, `sudo apt install --reinstall nvidia-driver-595` + reboot |
| RAM | 14 GiB | fine for score library; audio dataloaders must stream |
| free disk | **21 GiB** on `/` | sufficient with the Tier-A bundle and one-at-a-time extract-then-delete (peak stays under ~11 GB kept; see Stage 2 table) |
| CPU | 20 cores | score extraction possible on CPU with wav2vec2-family models, but slow (~1–2 trials/s/thread); GPU strongly preferred |

## Key conceptual correction

"Train the models" is mostly **not** needed. The minimal publishable path uses:

- **ASV**: pretrained SpeechBrain `spkrec-ecapa-voxceleb` (or WavLM-large + ECAPA) — no training;
- **CM**: either (a) released checkpoint of SSL-AntiSpoofing (Tak et al. 2022,
  wav2vec-2.0 XLSR + AASIST head, `TRAIN SET: ASVspoof 2019 LA`) or (b) train
  RawNet2/AASIST ourselves on ASVspoof 2019 LA (~3–6 h on one decent GPU);
- **SER**: pretrained e.g. `superb/wav2vec2-base-superb-er` — no training; optional
  fine-tune to a 4-class taxonomy (angry/neutral/happy/sad) matching EmoSpoof-TTS.

Only if we want headline numbers beyond "published checkpoints + our fusion" do
we fine-tune any of the three. That is a later, optional phase (RQ: does
co-training help?).

## Stage 0 — Environment quick fixes (0.5 day, no dependencies)

1. Reboot → re-run `nvidia-smi`. If still mismatched, reinstall the NVIDIA driver
   (`sudo apt install --reinstall nvidia-driver-5xx` matching NVML 595) and reboot.
   Do not let this block Stage 1 — everything infra-free can proceed.
2. Disk strategy (updated after size verification): the four corpora download
       to ≈ 21.6 GB compressed total; keep audio as FLAC (never convert to wav)
       and that is roughly also the extracted size. A staged
       **download → extract → delete archive** sequence peaks near ~28–32 GB —
       more than the 21 GiB currently free, so free ≈ 10–15 GB of space, or
       extract on an external drive. All dataset paths go into manifests via the
       `path` column, never into the repo.
   `~/data/asvspoof/...`. All dataset paths go into manifests via column `path`,
   never into the repo.
3. Create `easv-py` handling for torch/transformers/speechbrain **in a separate
   venv** (.venv-rm) so the tiny core `.venv` (numpy/scipy/yaml) stays testable:
   `python -m venv .venv-rm && .venv-rm/bin/pip install torch torchaudio transformers speechbrain`.

## Stage 1 — Close the novelty matrix completely (2–3 days, CPU only)

Status: matrix largely filled; the open items are verification-level, not reading-level.

1. [ ] Re-verify every ✅ cell against the actual PDF/HTML of the four papers
       (Sensors 2025 §5 fusion equations; GEM §3.2 gating; GCCE 2023 §3 two-stage).
2. [ ] Fill the four `TODO` rows (multi-task spoofing-robust ASV 2022; WavLM;
       ECAPA-TDNN; AASIST) — encoder-level, so "n/a" is acceptable where true.
3. [ ] Record the go/no-go decision with date + justification in the decision table.
4. [ ] Freeze the method-distinction paragraph; make sure none of "first/novel/
       state of the art" appears unless backed by a checked cell.
5. [ ] Optional: run the 12 documented search strings across Scholar/IEEE/ISCA and
       log any NEW 2023–2026 emotionally-aware SOSV work as an extra row.

Deliverable: NOVELTY_MATRIX.md with no TODO left except real-data evaluation.

## Stage 2 — Datasets: obtain, license, manifest (4–7 days, network + disk)

Tiered target sets, **verified sizes** — the full original list was overestimated;
staged selective extraction makes everything fit in the current 21 GiB free disk.

### Tier A — recommended “GEM-replica” bundle (~16.3 GB downloads, ~11 GB kept on disk)

| corpus | role | download | kept on disk after cleanup | notes |
|---|---|---|---|---|
| ASVspoof 2019 LA `LA.zip` | CM training (train split) + baseline reproduction (dev) | 7.12 GB, one zip, **no public per-partition archives** | ~2 GB (train ≈ 25,380 utts ≈ 10 h, dev ≈ 24,844; eval ≈ 71,237 utts dominates the zip and is **extracted-selectively, then discarded**) | Edinburgh DataShare; register first |
| EmoSpoof-TTS | spoof trials: emotion labels + 3 generators | 6.7 GB (CosyVoice 2.7 / F5-TTS 1.1 / StyleTTS2 2.9) | ~7 GB extracted | spoof-only — **contains no bona fide audio**, per its Hugging Face README |
| ESD (English, speakers 0011–0020) | bona fide trials + SER fine-tune support | ~2.45 GB single zip via the public Hugging Face mirror `sonchuate/ESD_dataset` — **unblocks immediately, no form needed**; usage stays research-only per ESD terms, and the signed-form e-mail can optionally follow as formal re-confirmation | ~1–1.5 GB (English subset only) | bona fide counterpart of EmoSpoof-TTS (same speakers, 300 utts × 4 emotions + 50 reference utts) |
| SASV 2022 protocols | metric-code validation + baseline reproduction | ~4.5 MB | tiny | public trial-list files (dev/eval) on GitHub |

### Tier 0 fallback — if the ESD licence is delayed (~9.2 GB downloads)

EmoSpoof-TTS + ESD only; train the CM from scratch directly on EmoSpoof's train
split (CosyVoice spoof + ESD bona fide, exactly as the GEM paper fine-tunes
RawNet2). Losses: weaker headline comparability to SASV-2022-style protocols,
no ASVspoof baseline reproduction. ASVspoof 2019 LA train/dev can be added later.

### Optional Tier B — post-A extensions (not downloaded initially)

**EmoFake (8.8 GB, `emofake.tar.gz`, Zenodo record 12806228, CC BY-NC-ND 4.0)** —
recommended add-on: contains the **real ESD recordings of the same ten English
speakers 0011–0020** (17,500 bona fide files, all five emotions) — an openly
licenced, citable alternative confirmation — plus two additional spoof generators
(TTS and speech-to-speech conversion per speaker pair) as new attack families.
Also: ASVspoof 2021 LA eval (7.8 GB, second cross-generator benchmark) and the
ASVspoof 2019 LA eval partition. Neither blocks any planned claim.

Steps (order matters for the 21 GiB disk — one corpus at a time, extract
selectively, delete the archive before the next):

1. [ ] ~~Day-1 action — ESD licence~~ **RESOLVED**: ESD is being pulled from the
       public Hugging Face mirror now (`~/data/easv/esd/`), so the human licence
       step no longer blocks Stage 2. If later desired, fill the official form
       ([ESD GitHub](https://github.com/HLTSingapore/Emotional-Speech-Data)) and
       e-mail it — the draft at [../docs/ESD_LICENCE_EMAIL.md](ESD_LICENCE_EMAIL.md)
       is ready; sending is optional, not a blocker.
2. [ ] Download each corpus in Tier-A order (ASVspoof 2019 LA → extract only
       `ASVspoof2019_LA_train` + `ASVspoof2019_LA_dev` → delete `LA.zip`; then
       EmoSpoof-TTS → extract → delete zips; then ESD → extract only speakers
       `0011`–`0020` → delete zip). Record licence + version + checksum for each in
       `data_manifests/README.md`.
2. [ ] Write `scripts/build_manifest_asvspoof.py` — turn CM file + protocol in
       official `*.txt` metadata of the train/dev partitions into our CSV schema
       (`trial_id, speaker_id, claimed_speaker, trial_type, emotion, attack, split, source, path`).
       ASVspoof files have **no emotion ground truth** — leave `emotion` empty
       there; the SER producer's *predicted* probabilities are what the fusion sees.
3. [ ] Write `scripts/build_manifest_emospoof.py` — EmoSpoof-TTS contributes the
       spoof rows (emotion ground truth × 3 generators: train=CosyVoice /
       val=F5-TTS / test=StyleTTS2); the **target and nontarget trials come from
       ESD recordings of the same 10 English speakers** (ESD IDs `0011`–`0020`,
       300 utts × 4 emotions + 50 reference utts each per the paper §3.1),
       because the EmoSpoof-TTS release contains no bona fide audio.
4. [ ] For SASV 2022: the public GitHub repo provides the dev/eval **trial
       protocols** and the baseline *code*, but **no official precomputed
       baseline score files**. Two routes: (a) run the SASVC2022_Baseline model
       (AASIST-based SV + CM) on the 2019 LA audio with those protocols — costs
       a few GPU-hours but is fully self-contained; (b) email the organizers
       (​sasv-challenge) for baseline scores. Either route yields a baseline
       reproduction on official protocols: run `easv evaluate` with those
       scores + trial lists; it must match published numbers — this is the first
       real-data result and it validates the metric code.
5. [ ] Run `easv audit` on each manifest; fix errors, record intentional warnings.

## Stage 3 — Score producers (the real engineering, 5–10 days GPU)

Each producer is a standalone script; output is always `scores.csv` matching the
manifest by `trial_id`, columns exactly as `src/easv/scores.py` consumes them.

3.1 ASV producer
- [ ] `scripts/score_asv_speechbrain.py`: load `speechbrain/spkrec-ecapa-voxceleb`,
      compute utterance embeddings per trial, cosine-similarity between enrollment
      and test, write `speaker_score`.
- [ ] Batch by `_part` slices; resumable (skips completed outputs).
- [ ] Sanity: SV-EER on bona fide trials within 1–2% of published SASV 2022 SV baselines.

3.2 CM producer
- [ ] Option A (no training): released SSL-AntiSpoofing checkpoint → `cm_score =
      logit of bona-fide probability`. Verify SPF-EER aligns with the paper.
- [ ] **Option B — selected by owner: train from scratch.** Written recipe for
      this laptop's RTX 3050 6GB Laptop GPU:
      - repository: `clovaai/aasist` (official, ~585k params, trains on ≤8 GB cards)
      - dataset: ASVspoof 2019 LA train split (flac); dev monitored for the best model
      - env: `.venv-rm` with torch + torchaudio; flac → 16 kHz wav decoded on the fly
      - hyperparameters: batch 32 (6 GB), Adam lr 1e-3 with cosine decay, 30 epochs,
        random 4-s crops; trained on the 7-class spoof-tag objective, evaluated on
        the pooled bona-fide-vs-spoof logit
      - speed estimate: ~15–20 min/epoch on the 3050 → **~8–10 h total**; run in
        `tmux` overnight with a resumable checkpoint each epoch
      - acceptance: SPF-EER on ASVspoof 2019 LA dev within ~1.5% of published AASIST
        performance; a monotonically decreasing dev loss over early epochs
      - `cm_score` = logit of the bona-fide probability (higher = more bona fide)

3.3 SER producer
- [ ] `scripts/score_ser_wav2vec2.py`: `superb/wav2vec2-base-superb-er` → softmax
      → relabel taxonomy to {angry, happy, neutral, sad} (+ collapse other classes
      into "neutral" with a written rule), add `emo_*` probability columns.
- [ ] Compute uncertainty features in `easv` at fit time — entropy, max-prob,
      confidence are derived inside `fusion.py` already, no new code needed.
- [ ] Optional fine-tune arm: fine-tune the SER head on EmoSpoof-TTS train
      generators so the taxonomy exactly matches; log this as a separate arm
      (SER-src-pretrained vs SER-src-finetuned), never mix in one comparison.

3.4 Calibration
- [ ] Fit a scalar temperature on each producer on the dev split only; document
      score direction (`speaker_score` ↑ = target, `cm_score` ↑ = bona fide).

## Stage 4 — Pre-register success criteria before any test run (0.5 day, pure writing)

Fill the TODO rows of PROTOCOL.md §6 with concrete numbers chosen from Stage 2
baseline reproduction:

| criterion | proposed value |
|---|---|
| primary metric | SASV-EER on test (fixed) |
| comparison baseline | `linear` fusion (fixed) |
| required gain over baseline | ≥ 0.02 absolute SASV-EER (propose; owner confirms) |
| must-hold generalization track | cross-generator (unseen generator EmoSpoof test) |
| acceptable bona-fide FRR increase | ≤ +0.01 absolute at the matched dev threshold |
| statistical requirement | paired bootstrap 95% CI excluding 0 (fixed) |

Sign and date this commit *before* the Stage 5 evaluation call — that is the whole
point of pre-registration.

## Stage 5 — Real-data evaluation (2–3 days compute, after Stages 2–4)

```
easv audit --manifest <real>.csv --out results/<run>/audit
easv evaluate --manifest <real>.csv --scores <real_scores>.csv \
              --config configs/sasv2022_template.yaml \
              --config configs/ablation_no_emotion_uncertainty.yaml \
              --out results/<run>
```

1. [ ] Fit fusion on train split only, thresholds on dev, **touch test once**.
2. [ ] Report SASV-EER / min a-DCF / min t-DCF per fusion arm, per emotion, per
      attack family; grouped bootstrap CIs; paired p_b_not_better vs `linear`.
3. [ ] Inspect `gate_weights` per emotion — is the mechanism interpretable?
4. [ ] Ablations: `use_uncertainty: false`, `target_far` threshold; also the
      capacity-control arm (`emotion_interaction`) must be reported so a small
      gain is not silently attributed to conditioning.
5. [ ] Failure modes to report honestly: FRR increase on bona fide, certain
      emotion classes where the gate hurts, SER prediction shift across generators.

## Stage 6 — Ablations & depth-explorations (3–5 days, optional but strengthens paper)

- [ ] Frozen vs fine-tuned SER comparison.
- [ ] Different emotion taxonomies (2-class arousal vs 4-class) — feeds RQ on taxonomy.
- [ ] Cross-emotion track: hold out one emotion from training AND testing (EmoSpoof
      supports it natively).
- [ ] Extra arms if compute allows: XLSR-AASIST as second CM to show fusion is
      producer-agnostic.

## Stage 7 — Paper writing (1 week, CPU only)

- [ ] `paper/manuscript.md` using the outline already in `paper/`.
- [ ] Figures: SASV-EER by model (bar), per-emotion gain (heat/dot), gate-weight
      visualization by emotion, reliability diagrams.
- [ ] Duplicate every number in a `paper/results_tables.md` generated from
      `results.json` (write a small table-generator script to avoid manual copy).
- [ ] Final: the course `Course_Project_Report_*.docx` gets its Chapter 4
      regenerated from real results (currently explicitly labelled synthetic).

## Order of execution and parallelism

```
Stage 0 (fix GPU)          ──┐
Stage 1 (novelty)          ──┤  can run in parallel, CPU only
Stage 2 (datasets+manifests)─┘
        │
Stage 3 (score producers) ── needs GPU + datasets
        │
Stage 4 (pre-register) ──── pure writing, do right after Stage 2 baselines
        │
Stage 5 (evaluate) ──────── first real claim
        │
Stage 6 (ablations) → Stage 7 (paper)
```

Realistic elapsed time: **3–4 weeks** part-time with this laptop, assuming the
GPU driver is fixed. Fully local, no Colab, no external drive needed with the
Tier-A dataset bundle.

## Decisions — resolved by the project owner (2026-10-21)

| decision | choice | consequence |
|---|---|---|
| CM producer | **train from scratch** (AASIST recipe on ASVspoof 2019 LA) | Stage 3.2 Option B is the path; a released checkpoint is kept only as a fallback comparison arm |
| GPU work location | **this laptop** (RTX 3050 6GB) | Stage 0 must fix the driver via reboot; training runs in `tmux` overnight |
| Primary generalization track | **cross-generator** (CosyVoice → F5-TTS → StyleTTS2) | headline claim, headline figure, and the pre-registered must-hold track |
| SER taxonomy | **4-class** (angry / happy / neutral / sad) | SER producer relabels superb-ER output into exactly these four classes with a written collapse rule |
| Remaining disk decision | **resolved — Tier A bundle** | 16.3 GB downloads, ~11 GB kept on disk after selective extraction; fits the current 21 GiB with no external drive. ESD arrives via the HF mirror (2.45 GB). ASVspoof 2021 LA eval and the 2019 eval partition are explicitly deferred (Optional Tier B) |

## Explicit non-goals

- Do not fine-tune WavLM/HuBERT themselves (SSL encoders stay frozen; only heads train).
- Do not add new audiences, UI, or deployment work before Phase 7.
- Do not claim any positive result until the pre-registered criteria pass on real data.
