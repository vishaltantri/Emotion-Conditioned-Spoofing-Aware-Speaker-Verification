# Manuscript

**Status: workshop draft written (2026-10-25).** [manuscript.md](manuscript.md) is a
complete draft — abstract through conclusion, three result tables, the three
mechanism probes, limitations, data/ethics statement and a reference list that still
needs verifying. It reports the ESD/EmoSpoof-TTS pilot numbers and says so in every
caption; the journal version re-runs the same pipeline on CREMA-D
([docs/CORPUS_STRATEGY.md](../docs/CORPUS_STRATEGY.md)) and swaps the tables, not
the argument. Before submission: verify every citation and DOI, decide whether the
five figures in [figures/](../figures/README.md) go in as-is, and re-check that no
sentence credits the gated arm's win to emotion.

The course report
[Course_Project_Report_EASV_Audited.docx](../Course_Project_Report_EASV_Audited.docx)
is the current write-up. It reports the executed real-data results — four
countermeasure regimes over 53,900 trials, the five-arm fusion ladder, the
three-generator cross-generator run, and the LA-retrain transfer control — with the
caveats the data requires.

Manuscript drafting is blocked on two things, not on missing results:

1. **The novelty gate is closed on the mechanism and open only for a baseline
   reproduction.** The closest work has been read and recorded in
   [docs/NOVELTY_MATRIX.md](../docs/NOVELTY_MATRIX.md). The held-out-emotion
   evaluation now exists (two folds; excluding an entire emotion from CM training
   and from the fusion fit costs ≤0.03 points on that emotion) so the
   "never tested on an unseen emotion" objection is answered, but no published
   baseline has been reproduced under a matched protocol. A paper can describe a
   corpus, a protocol and a measured finding, but not a demonstrated advantage of
   the proposed mechanism — the measured gate weights are flat across emotions, the
   SER-confidence ablation reproduces the gated arm's Regime D edge with a single
   scalar, and the held-out-emotion fold is free because the arms were hardly using
   emotion identity (see [docs/REAL_RESULTS.md](../docs/REAL_RESULTS.md)). The
   mechanism claim is therefore closed as a negative result; the paper's
   contribution is the evaluation, the transfer findings and the mechanism analysis.
2. **The corpus licence.** ESD's research licence was never granted (the audio came
   from a public mirror), but a workshop submission does not require it: ESD's own
   terms say "free to use only for research purpose", the EmoSpoof-TTS clones are
   CC BY 4.0 with no ESD audio inside, and the licence email
   ([docs/ESD_LICENCE_EMAIL.md](../docs/ESD_LICENCE_EMAIL.md)) is a courtesy that
   converts "probably fine" into "documented". For the **journal** submission this
   ambiguity is removed rather than footnoted: the corpus is re-anchored on CREMA-D
   (ODbL 1.0, no request needed, 91 actors instead of 10), with clones regenerated
   from the same three systems. Design, compliance position and costs:
   [docs/CORPUS_STRATEGY.md](../docs/CORPUS_STRATEGY.md). See also
   [docs/PUBLICATION_ACTION_PLAN.md](../docs/PUBLICATION_ACTION_PLAN.md).

Target venue per the action plan: an ASVspoof/ICASSP/Interspeech workshop, framed
as a measurement and negative-result paper; a journal submission waits for the
licence plus the baseline-reproduction item (the held-out-emotion item is done).

When drafting starts, the constraints are recorded in
[docs/REAL_RESULTS.md](../docs/REAL_RESULTS.md) ("What these results do and do not
establish") and [docs/PROTOCOL.md](../docs/PROTOCOL.md) §6 (success criteria were not
pre-registered). No abstract or conclusion may describe the gated arm as an
emotion-conditioned improvement: it wins on the three-generator corpus, but the gate
does not use emotion, and it loses to plain fusion on the two-generator corpus.

Section outline taken from the literature-review design document
(`Emotion_Aware_Anti_Spoofing_Speaker_Verification-1.pdf`):

1. Abstract
2. Keywords
3. Introduction — voice biometrics, spoofing threat, emotional variation, why
   speaker/spoof/emotion evidence are usually studied separately
4. Literature review
   * A. Traditional speaker recognition
   * B. Deep-learning speaker verification
   * C. Self-supervised speech representations
   * D. Speech anti-spoofing
   * E. Emotion-aware speech analysis
   * F. Research gap
5. Proposed methodology
   * A. System architecture
   * B. Speech pre-processing
   * C. Speaker representation
   * D. Anti-spoofing module
   * E. Emotion analysis module
   * F. Feature/score fusion
   * G. Authentication decision
6. Experimental setup (protocol: [docs/PROTOCOL.md](../docs/PROTOCOL.md))
7. Results and discussion
8. Conclusion and future work
9. References

Before submission: verify every citation and DOI, dataset licences and official
splits, include limitations and failure cases, check the target venue's scope,
formatting, preprints and generative-AI policies, and confirm no dual submission.
