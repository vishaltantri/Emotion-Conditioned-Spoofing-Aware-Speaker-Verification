# Publication action plan

**Verdict (2026-10-23):** the current work is a strong, honest course project with a
journal-grade reproducibility package, but it is **not yet publishable** in a
journal: the corpus licence is unresolved (fatal), the central mechanism claim is
unsupported by its own evidence (the learned gate is flat across emotions), the
closest prior work (GEM, Interspeech 2025) already uses the same corpus and an
almost identical held-out-generator split, there is no held-out-emotion
evaluation, no reproduced published baseline, and only one split and one seed
over 10 speakers.

**Fastest credible path:** reposition as a measurement / negative-result paper for
an ASVspoof, Interspeech or ICASSP workshop, with the SER-confidence ablation as
the key new experiment. The journal path additionally needs the licence and the
full experiment set below.

Owner split: **[USER]** = only the project owner can do it; **[AGENT]** = can be
executed in this repository on request.

## P0 — Actions only the owner can take (start now, nothing blocks them)

| # | change | why | how | done when |
|---|---|---|---|---|
| 1 | **[USER] Send the ESD licence request** | Every result is built on ESD bona fide audio obtained from a mirror, not under licence. No journal accepts unlicensed speech data. | [docs/ESD_LICENCE_EMAIL.md](ESD_LICENCE_EMAIL.md) is a ready draft: download and sign the official form from the ESD GitHub page, fill the placeholders, send to the maintainer address. | Written acknowledgement or grant received; log it in [docs/PROGRESS_LOG.md](PROGRESS_LOG.md). |
| 2 | **[USER] Free ~10 GB of owner-owned disk** | Blocks any further corpus work and backups: the root volume is at 99 %. Candidates measured in [PROGRESS_LOG.md](PROGRESS_LOG.md): `~/.cache/huggingface` (2.9 GB, unrelated NLP models), `~/.gradle` (3.1 GB), `~/.local/share/Trash` (2.1 GB). | Delete or move them yourself; they are outside this project, so the agent will not touch them. | `df -h /` shows ≥ 10 GB free. |
| 3 | **[USER] Check the report's contents page in Word** | The TOC is a static table; page numbers for Ch. 5 / References / Appendix (18 / 19 / 21) are estimates from word growth — no Word or LibreOffice exists on this machine to paginate. | Open `Course_Project_Report_EASV_Audited.docx` in Word, compare the three page numbers, report corrections. | TOC numbers confirmed or corrected. |

## P1 — The one experiment that decides the paper's viability [AGENT]

**SER-confidence ablation.** Replace the 4-way emotion probability vector with a
single scalar — SER max probability (or 1 − entropy) — refit all five fusion arms
on the same scores, rerun `easv evaluate` for Regime B (AASIST-LA CM) and Regime D
(three-generator cross-gen CM).

* **Why:** with a failing countermeasure, emotion features gained +5.5 SASV-EER
  points, but the most likely mechanism is that a *confidently wrong* SER
  (32.8 % accuracy at 0.825 mean confidence on clones) is acting as a spoof cue.
  If a one-dimensional confidence scalar reproduces the gain, the "emotion-aware"
  story is dead and the paper must be a negative result — which is still
  publishable, but only if stated. If the gain disappears, emotion carries signal
  beyond confidence, which strengthens the framing.
* **How:** extend the feature builder in `src/easv/scores.py` /
  `configs/real_run.yaml` with a `features: confidence_only` option; refit; paired
  bootstrap vs both `linear` and the full-emotion arms.
* **Acceptance:** table in [docs/REAL_RESULTS.md](REAL_RESULTS.md) with CIs and
  p-values for {full emotion features} vs {confidence scalar} vs {linear} in
  Regimes B and D; gate weights reported.

## P2 — Held-out-emotion evaluation [AGENT]

Retrain the cross-generator countermeasure and refit the fusion arms with **one
emotion condition excluded** from CM training and from dev-based threshold
selection (the same leave-one-out machinery as `--holdout-attack` in
`scripts/train_cm_indomain.py`), then score the held-out emotion on test.

* **Why:** every emotion currently appears in train, so all per-emotion tables are
  descriptive. This closes the second open item of the novelty gate
  ([docs/NOVELTY_MATRIX.md](NOVELTY_MATRIX.md)) and is the experiment the title's
  word "emotion-aware" actually requires. Run leave-one-emotion-out over all five
  conditions if compute allows; at minimum held-out `angry` (the most dangerous)
  and `neutral`.
* **Acceptance:** per-held-out-emotion SASV-EER with CIs, recorded alongside the
  cross-generator results and reported side by side, never merged.

## P3 — Variance: seeds and split repeats [AGENT]

Three fusion seeds × five arms on the headline run, plus at least two alternative
speaker splits (still 4/2/4, StyleTTS2 held out).

* **Why:** the grouped bootstrap resamples only 10 speakers, so intervals are
  coarse; reviewers will ask whether the Regime D ordering (gated > interaction >
  linear) survives a different split.
* **Acceptance:** a table of mean ± std SASV-EER across seeds and splits; the
  conclusion states explicitly whether the ordering is stable.

## P4 — Reproduce one published baseline under a matched protocol [AGENT]

Train RawNet2 (or the SASV 2022 baseline recipe) on EmoSpoof-TTS with GEM's exact
speaker/generator split and report its EER next to the project's numbers; second
option: reproduce an AASIST LA number on the official LA eval partition.

* **Why:** last open novelty-gate item; also validates the metric code against a
  reference number. Reviewers will otherwise read the headline as "GEM's protocol
  plus an ASV branch" with nothing external to anchor it.
* **Acceptance:** reproduced EER within a stated tolerance of the published value,
  recorded in [docs/NOVELTY_MATRIX.md](NOVELTY_MATRIX.md).

## P5 — Figures and the reframed write-up [AGENT]

No figures exist yet (`figures/` is empty). Generate from the per-trial score CSVs:

1. per-generator ECAPA score distributions (targets vs clones) — the 5.9 % vs
   59.0 % story in one figure;
2. DET curves per regime;
3. gate-weight spread across emotions (the flat-gate evidence, which *is* the
   paper's key figure);
4. spoof-acceptance by emotion × generator heatmap.

Then draft the manuscript in `paper/` with the repositioned claim:
*"emotion-conditioned score fusion for SASV does not improve on conventional
fusion when the countermeasure works; its apparent gains trace to capacity,
calibration and a spoof cue in the SER front-end; cross-generator CM transfer is
governed by generator adjacency, not ASVspoof lineage."* No "novel", "first" or
"state of the art" wording anywhere (novelty-gate rule). **Blocked on P1** — the
ablation decides whether the framing above or a partially positive one is used.

## P6 — Licence fallback [USER decides, AGENT executes]

If ESD refuses or does not answer: the licence chain is unavoidable — EmoSpoof-TTS
clones are generated *from* ESD emotional speech, so any substitute corpus must be
re-synthesised from licensed bona fide audio. Options, in order: (a) supervisors'
institutional ESD access; (b) a different licensed emotional corpus re-cloned with
the same three generators (weeks of compute); (c) keep the workshop path but with
the corpus described as "research use, licence pending" — acceptable at some
workshops, not at journals.

## What NOT to do

* Do not add WavLM/HuBERT encoders, VC/replay attacks or new corpora before P1–P3:
  none of them fixes a rejection reason; they add surface, not evidence.
* Do not merge the two-generator and three-generator results into one table.
* Do not describe the gated arm's Regime D win as an emotion-conditioned
  improvement — the gate weights refute it (flat: 4.02–4.05 / 9.69–9.73).

## Order and dependencies

```
P0 (user, now)          ── licence + disk + TOC       (no dependency)
P1 (agent)              ── SER-confidence ablation     (no dependency; decides framing)
P2 (agent)              ── held-out emotion            (after P1, shares CM training)
P3 (agent)              ── seeds/splits                (after P1)
P4 (agent)              ── baseline reproduction       (independent; disk from P0-2 helps)
P5 (agent)              ── figures + manuscript        (after P1; P2–P4 strengthen it)
P6 (user decision)      ── only if P0-1 fails
```

Estimated effort: P1 ≈ half a day of compute; P2 ≈ 1 day; P3 ≈ 1–2 days; P4 ≈ 3–5
days; P5 ≈ 2–3 days of writing. Workshop submission realistic ~2 weeks after P1–P3;
journal submission realistic only after P0-1 and P2–P4, ~4–6 weeks.
