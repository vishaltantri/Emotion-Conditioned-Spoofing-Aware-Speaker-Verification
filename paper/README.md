# Manuscript

**Status: not started.** The course report
[Course_Project_Report_EASV_Audited.docx](../Course_Project_Report_EASV_Audited.docx)
is the current write-up. It reports the executed real-data results — four
countermeasure regimes over 53,900 trials, the five-arm fusion ladder, the
three-generator cross-generator run, and the LA-retrain transfer control — with the
caveats the data requires.

Manuscript drafting is blocked on two things, not on missing results:

1. **The novelty gate is still conditionally closed.** The closest work has been read
   and recorded in [docs/NOVELTY_MATRIX.md](../docs/NOVELTY_MATRIX.md), but no
   published baseline has been reproduced under a matched protocol, and no
   held-out-emotion evaluation exists. Without those, a paper can describe a corpus,
   a protocol and a measured finding, but not a demonstrated advantage of the
   proposed mechanism — the measured gate weights are flat across emotions.
2. **The corpus licence is unresolved.** ESD's research licence was never granted
   (the audio came from a public mirror). See
   [docs/REMAINING_WORK_PLAN.md](../docs/REMAINING_WORK_PLAN.md), Stage 2.

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
