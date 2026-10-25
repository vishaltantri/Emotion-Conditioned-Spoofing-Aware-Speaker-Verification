# Corpus strategy: re-anchoring the journal submission on CREMA-D

**Status: decided 2026-10-25.** The workshop submission stands on the existing
ESD-derived results. The *journal* submission will not: it is re-anchored on
CREMA-D, whose licence requires no request, no signature and no approval. The
ESD work becomes the pilot that generated the hypotheses, the code and the
protocol; CREMA-D supplies the evidence.

## 1. Is CREMA-D actually licence-free? Yes — with two precise caveats

The official repository states:

> "This Crowd-sourced Emotional Mutimodal Actors Dataset (CREMA-D) is made available
> under the **Open Database License**: http://opendatacommons.org/licenses/odbl/1.0/.
> Any rights in individual contents of the database are licensed under the
> **Database Contents License**: http://opendatacommons.org/licenses/dbcl/1.0/"
>
> — [github.com/CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)

and the distribution catalogue lists it as ODbL v1.0 with `usage: commercial`
([audEERING corpus card](https://audeering.github.io/datasets/datasets/crema-d.html)).

What that means in practice:

* **No signature, no email, no approval.** The repository asks users to "fill out
  this form … so that we can keep a record of the community of CREMA-D users". That
  is a courtesy tally, not an access gate — the files are a public `git lfs` clone
  (or GitLab mirror). Contrast ESD, whose terms require a signed licence form and
  an authorisation reply, which is exactly the dependency this change removes.
* **Commercial use is allowed**, unlike RAVDESS (CC BY-NC-SA), TESS (CC BY-NC) and
  EmoV-DB (non-commercial only). A journal does not have to reason about
  "research-only, permission pending".
* **A real, citable instrument exists** — ODbL 1.0 + DbCL 1.0 — so the data
  statement names a licence instead of describing a hope.

The two caveats, stated plainly because a reviewer will raise them:

1. **ODbL/DbCL are copyright and database rights, not personality rights.**
   Neither grants consent to train a system that imitates an identifiable actor,
   and the ODbL text itself excludes "privacy rights / data protection rights /
   personality rights" from its scope. If we clone CREMA-D actors' voices, the
   licence does not answer the ethics question — it only puts the *audio* on a
   firm footing. The same is true of ESD, so this is not a regression, but it must
   be handled explicitly (§4).
2. **Share-alike applies to distributed derivative databases.** If we publish a
   derived database (a bundle of audio, or per-clip scores keyed to clips), ODbL
   requires that database to be offered under ODbL. We distribute trial manifests
   and aggregate metrics only, and will label any future score release ODbL.

Verdict: CREMA-D clears the licence dependency. The residual risk is the cloning
act, not the corpus paperwork, and it is manageable (§4).

## 2. Why CREMA-D is also a scientific upgrade, not just a workaround

| property | current corpus (ESD English 0011–0020) | CREMA-D |
|---|---|---|
| licence | research-only, form required, **never granted** | ODbL 1.0 + DbCL 1.0, open download |
| speakers | **10** | **91** (48 male, 43 female, ages 20–74) |
| evaluation speakers (test split) | **4** — the resampling unit of every CI in the paper | 15 at a 61/15/15 split |
| emotions | 5 intended (angry, happy, neutral, sad, surprise) | 6 intended (anger, disgust, fear, happiness, neutral, sadness) + **perceived** labels from vote tables |
| audio | 48 kHz source, arbitrary length | 16 kHz mono wav, 1.3–5.0 s per clip |
| literature status | non-standard in SASV | one of the standard emotional-speech corpora |

The speaker count is the point that matters most for a journal. Every confidence
interval in the current paper is a grouped bootstrap over **four** claimed
speakers. That is the weakest part of the statistics, it is independent of the
licence, and no reviewer will let it pass in a journal. Moving to 91 actors fixes
it — the swap therefore buys the two things the submission actually needs.

## 3. What survives, what is re-measured

Survives untouched (the reason this is a re-anchor and not a restart):

* `src/easv/` in full — manifest schema and audit, score joining, the five-arm
  fusion ladder, metrics, thresholds, grouped and paired bootstrap, breakdowns;
* the protocol in [PROTOCOL.md](PROTOCOL.md): trial construction, split
  discipline, dev-only thresholds, reported-once test;
* the countermeasure *design* and two of the four regimes **weights included**:
  the ASVspoof-2019 AASIST checkpoint and the LA-retrained AASIST are fixed
  models, so Regimes B and the LA transfer control only need re-scoring on new
  audio — no retraining;
* both ablations (`emotion_features: confidence_only`, `folds.holdout_emotion`) —
  they are config, not code;
* [scripts/make_figures.py](../scripts/make_figures.py), which already takes
  `--results-root` and will point at the new runs;
* the findings themselves as **hypotheses with a pilot measurement**: the
  generator-specific vulnerability, the generator-adjacency transfer rule, and the
  mechanism negative result. Their status changes from "measured on our corpus" to
  "to be confirmed on the clean corpus".

Re-measured on CREMA-D: ASV and SER scores, the in-domain and cross-generator CNNs
(minutes to an hour of training), the fusion ladder, the breakdowns, the figures.

Not carried over: the 36,000 EmoSpoof-TTS clones. They were built from ESD
speakers, so their bona fide counterpart is ESD. They stay usable only if the ESD
licence arrives; otherwise they are reported in the pilot appendix, cited as
CC BY 4.0 data, never as the basis of the claim.

## 4. Compliance position (to appear in the paper's data statement)

* CREMA-D used under ODbL 1.0 / DbCL 1.0; the corpus is cited with its licence.
* Clones generated locally with open systems: StyleTTS2 (MIT code; its README
  requires disclosing that samples are synthetic and using voices whose speakers
  permit cloning — we disclose, in the paper and in every artefact description),
  F5-TTS (MIT code, **weights CC BY-NC 4.0**, so this arm is non-commercial
  research only and is labelled as such), CosyVoice (Apache-2.0).
* **No cloned audio leaves the repository or the institution.** Reported artifacts
  are trial manifests, aggregate metrics and per-trial *scores*; no wav of a real
  or imitated actor is released. This follows the EmoSpoof-TTS precedent, whose
  clones of ESD speakers were released for research.
* Voice imitation is used for attack simulation only, with no intent to
  impersonate, and the paper says so in an ethics paragraph. If a target venue
  requires more, the fallback is to keep the clone corpus internal and publish
  aggregate rates only — which is already the plan.

## 5. Corpus design

Parsing: files are `AudioWAV/<actor>_<sentence>_<EMO>_<LEVEL>.wav`, e.g.
`1001_TAI_HAP_XX.wav`, with `EMO ∈ {ANG, DIS, FEA, HAP, NEU, SAD}` and
`LEVEL ∈ {LO, MD, HI, XX}`.

* **Filter**: keep clips ≥ 2.0 s (CREMA-D bottoms out at 1.3 s, which is too short
  for a stable ASV score or a CM window). References for cloning must be ≥ 3.0 s.
* **Emotion axis**: the SER front-end (wav2vec2-base-superb-er) is a 4-way
  IEMOCAP model (angry / happy / neutral / sad). Those four emotions carry the
  emotion-conditioned arms; `disgust` and `fear` are kept as labelled conditions
  where the SER output is treated as unknown rather than guessed — the evaluation
  code already does exactly that.
* **Labels**: intended labels come from the filename; *perceived* labels come from
  CREMA-D's vote tables (`processedResults/tabulatedVotes.csv`). The journal
  version can report the perceived labels as a robustness check — something the
  ESD setup could not do, and directly relevant to the "SER is confidently wrong"
  finding.* **Split**: actors, not trials — 61 / 15 / 15 train / dev / test, speaker-disjoint by construction, which the audit will now pass cleanly instead of emitting the four `speaker_overlap` warnings the ESD manifests produce.
* **Nontarget trials**: the impostor utterance is drawn from the **same partition**
  as the claimed speaker, as the SASV protocol requires. The ESD builder drew
  impostors from any speaker, which is why all ten speakers appeared in every
  split and why the audit warned four times; the CREMA-D builder fixes the rule
  rather than the warning. (The ESD manifests are left as they were, because
  changing them would silently invalidate every reported number.)

## 6. Clone generation plan

Per (actor, emotion) cell: hold out **2 clips as cloning references**, synthesise
the remaining sentence texts. CREMA-D gives ~10–12 usable clips per cell.

| quantity | estimate |
|---|---|
| cells (91 actors × 6 emotions) | 546 |
| clones per cell per system | ~10 |
| clones per system | ~5,460 |
| clones, three systems | **~16,400** |
| plus bona fide retained | ~6,700 |
| generations at ~2.5 s each | **~12 GPU-hours** (allow 18 with model loading and retries) |

Deliberate choice: we do **not** inflate the count by re-synthesising the same
transcript with multiple reference clips. Duplicate content adds files, not
information, and the ESD corpus's 300-per-cell scale was never what made its
results strong — the speaker count is. If more test trials are needed, the cheap
lever is more emotion levels per cell, not repeated text.

Expected test split at 61/15/15: 15 speakers × 6 emotions × ~11 bona fide ≈ 990
target trials, plus 15 × 6 × 10 × 3 ≈ 2,700 spoof trials, plus generated
cross-speaker nontargets — roughly 4,000 test trials over **15** resampling
clusters, versus 21,560 trials over 4.

Compute reality check (from the vendor docs, not benchmarks we ran): StyleTTS2 and
CosyVoice-300M are plausible on the 6 GB RTX 3050 at batch 1; F5-TTS is borderline
(reported RTF 0.15 on an L20, and its ASR path costs extra VRAM). Plan: run
StyleTTS2 and CosyVoice locally, and either fit F5-TTS at short durations or rent
one GPU-day for it. If even that fails, a two-generator corpus still supports the
transfer claim — but the three-generator design is what produced the headline
finding, so treat it as a last resort and say so.

Generation must be **resumable and deterministic**: one row per (source clip,
reference clip, system) in a CSV written as each file lands, so an interrupted run
loses nothing and the manifest builder can be re-run at any time.

## 7. What the journal still needs beyond the corpus

1. **A reproduced published countermeasure under a matched protocol.** This is the
   last open novelty-gate item ([NOVELTY_MATRIX.md](NOVELTY_MATRIX.md)) and the
   most common reviewer demand. AASIST is already in the tree; the missing piece is
   a matched-protocol run on the new corpus, not a new model.
2. **Seed and split repeats.** Cheap now, and they turn "the arm ordering holds"
   into a claim with variance attached.
3. **A released artefact.** Manifests (ODbL-labelled), score files, configs and the
   figure script — no audio. This is what makes a measurement paper citable.

## 8. Implementation order

1. Fetch CREMA-D audio (`git lfs` sparse checkout of `AudioWAV/`; the full clone is
   7.55 GB including video, which we do not need).
2. `scripts/build_manifests.py crema` — **done** (2026-10-25): new builder reusing
   the existing manifest schema, with the ≥2 s header-only duration filter, the
   actor-disjoint split and same-partition impostors; covered by three tests in
   `tests/test_manifest_builders.py`. It runs on a fixture today and will run on
   the corpus the moment the audio is on disk.
3. `scripts/generate_clones.py` — per-system driver, resumable, writing the wav
   tree plus the generation CSV.
4. `scripts/verify_clone_quality.py` — **the gate**: before any fusion time is
   spent, check that the phenomenon reproduces (per-generator ASV acceptance,
   SER accuracy/confidence on clones, CM-EER). If the clones are unusable, stop
   here and fix the generator rather than the analysis.
5. Re-run producers → CM training → fusion → ablations → folds → figures. No core
   code changes expected.
6. Update the protocol, novelty matrix and results documents with the new numbers;
   keep the ESD results as the pilot appendix.

## 9. Risks

| risk | mitigation |
|---|---|
| clones are too clean / too degraded for the effect to reproduce | step 4 gate exists precisely to catch this; tune duration and reference length before scaling |
| F5-TTS does not fit 6 GB | run it on rented GPU time, or drop to a two-generator corpus and say so |
| a reviewer objects to cloning identifiable actors | aggregate-only reporting, ethics paragraph, no audio release; the fallback is a smaller corpus with no released scores |
| CREMA-D clips are short (1.3–5.0 s) | ≥2 s filter for trials, ≥3 s for references |
| ODbL share-alike | never distribute a derived database; if scores are released, label them ODbL |
