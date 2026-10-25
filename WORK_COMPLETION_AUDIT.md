# Work Completion and Repository Audit (initial snapshot, 2026-10-21)

> **This audit is a historical snapshot and is superseded for current status.** It
> was written when the repository contained only synthetic data: no corpus, no
> score producers, no countermeasure and 65 tests. Since then the project has
> executed the full pipeline on real speech (53,900 trials, three EmoSpoof-TTS
> generators, four countermeasure regimes, a SER-confidence ablation, two
> held-out-emotion folds, 85 tests) and the report has been updated. Read every
> "does not include real audio", "synthetic only", "65 tests" and "not
> demonstrated" statement below as the state on 2026-10-21. Current
> status: [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md) and
> [docs/REAL_RESULTS.md](docs/REAL_RESULTS.md); remaining open items are listed at
> the end of the progress log.

## Purpose

This file records the work completed to audit the supplied Emotion-Conditioned Fusion for Spoofing-Aware Speaker Verification (EASV) project and to prepare a replacement course-project report. It distinguishes carefully between implemented code, executed checks, synthetic demonstration outputs, and claims that would require a real speech experiment.

The completed Word deliverable is [Course_Project_Report_EASV_Audited.docx](Course_Project_Report_EASV_Audited.docx). The earlier filled report was not used as a technical source; the new report was written from the beginning using the original course-report template and the audited project files.

## Inputs reviewed

### Course-report template

The report template at /home/vct/Downloads/Course Project Work Report Template.docx was inspected before authoring. Its core formal structure was retained: A4 portrait layout, cover page, certificate, abstract, contents, chapter structure, references, appendix, and footer. Report text, headings, captions, and normal content were kept black. The coloured university logo remains the original template graphic rather than report text.

### Instructor requirements

The screenshot instructions were applied as follows:

| Requirement | Completion |
|---|---|
| Abstract of roughly 150–200 words | A 195-word abstract was written for the synthetic snapshot; the real-data pass replaced it with a 183-word abstract that still fits the limit. |
| Story-like introduction of at least two pages | The introduction starts from a voice-authentication scenario and spans pages 5–6. |
| 30–35 papers; at least 7–8 relevant citations | The report has 35 numbered references and substantial in-text citations. |
| Themed literature review | Six technique/problem-oriented themes are used instead of a paper-by-paper list. |
| Methodology of 3–4 pages | Chapter 3 covers pages 10–12. |
| Results and discussion of 3–4 pages, with a table and graph | Chapter 4 covered pages 13–15 with two tables and a labelled chart; the real-data pass added the regime tables and the component table in section 4.4. |
| Single-column Word file | The final deliverable is a single-column DOCX file. |

The course-report requirements do not make a synthetic smoke test equivalent to a real biometric experiment. The final report explicitly labels all project demonstration results as synthetic.

### Repository and source material

The review covered:

- [README.md](README.md)
- [Emotion_Aware_Speaker_Verification_Research_Plan.md](Emotion_Aware_Speaker_Verification_Research_Plan.md)
- [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- [docs/PROTOCOL.md](docs/PROTOCOL.md)
- [docs/NOVELTY_MATRIX.md](docs/NOVELTY_MATRIX.md)
- the implementation under [src/easv](src/easv)
- configuration files under [configs](configs)
- the automated tests under [tests](tests)
- the supplied supporting PDF, Emotion_Aware_Anti_Spoofing_Speaker_Verification-1.pdf

## Reconstructed project objective

EASV is not an end-to-end speech-model training project. It is a score-level research and evaluation framework for this question:

Can predicted emotion probabilities and their uncertainty improve the combination of speaker-verification and anti-spoofing scores while preserving a leakage-aware train, development, and test protocol?

The intended upstream workflow is:

1. A speaker-verification system produces a speaker score.
2. An anti-spoofing countermeasure produces a bona-fide or spoof score.
3. An emotion recogniser produces predicted emotion probabilities and optionally uncertainty features.
4. EASV reads those scores and trial metadata, audits them, trains a controlled set of fusion models, selects a threshold on development data, and evaluates a held-out test split.

The source repository does not include real audio, an ASV audio model, a countermeasure audio model, an emotion-recognition audio model, pretrained weights, or a completed public-corpus benchmark run.

## Implementation audit

### Manifest and data contract

The manifest layer in [src/easv/manifest.py](src/easv/manifest.py) describes the trial metadata used by the framework. It tracks trial identity, split, actual speaker, claimed speaker, trial type, emotion, attack system, source, and path.

The implemented design keeps ground-truth conditions in the manifest and keeps model inputs in score files. This distinction is central to the project:

| Item | Intended use |
|---|---|
| Ground-truth emotion label | Split definition, coverage audit, and post-evaluation error analysis. |
| Predicted emotion probabilities | Optional inputs to the fusion model. |
| Actual and claimed speaker | Trial validity and target/nontarget definitions. |
| Attack system | Cross-generator audit and reporting. |

Ground-truth emotion is not meant to be supplied directly to the fusion model. This reduces a major leakage risk.

The trial classes are:

| Trial class | Meaning |
|---|---|
| Target | Bona fide speech from the claimed speaker. |
| Nontarget | Bona fide speech from a different speaker. |
| Spoof | Synthetic, converted, replayed, or otherwise attacked speech that must be rejected. |

### Audit and score-alignment functions

[src/easv/audit.py](src/easv/audit.py) implements checks for manifest integrity and experimental discipline. The review found checks covering required fields, duplicate trial IDs, claimed-speaker consistency, score/manifest alignment, split coverage, and overlap conditions involving speakers, attack systems, and emotions.

[src/easv/scores.py](src/easv/scores.py) aligns score rows to manifest rows through trial identifiers. This is important because an unreported row-order mismatch can create invalid but apparently plausible metrics. The explicit join behaviour is safer than assuming that score and manifest rows are already aligned.

### Metrics and reporting

[src/easv/metrics.py](src/easv/metrics.py) and [src/easv/evaluation.py](src/easv/evaluation.py) support the evaluation portion of the project. The implemented reports include:

- speaker-verification EER;
- spoofing EER;
- spoofing-aware speaker-verification EER, or SASV-EER;
- minimum a-DCF;
- t-DCF where configured;
- false-accept and false-reject quantities;
- precision, recall, and F1;
- Brier score and expected calibration error;
- bootstrap confidence intervals;
- paired comparisons against a baseline;
- breakdowns by emotion or attack system.

The evaluation logic makes a useful distinction: an EER needs both positive and negative classes. A spoof-only attack subset should not be given a conventional EER. The framework instead reports a spoof-acceptance rate at the development-selected threshold where appropriate.

### Fusion ladder

[src/easv/fusion.py](src/easv/fusion.py) contains five deliberately ordered fusion conditions:

| Model | Function in the comparison |
|---|---|
| Speaker-only | Simplest control using only the speaker score. |
| Linear baseline | Conventional speaker-score plus countermeasure-score fusion. |
| Emotion additive | Adds predicted emotion and uncertainty features. |
| Emotion interaction | Adds score-by-emotion interaction terms as a capacity control. |
| Emotion-gated | Produces non-negative, trial-specific weights for speaker and countermeasure evidence. |

The implementation uses L-BFGS for logistic fusion models. The gated model uses a small tanh hidden representation, softplus non-negative weighting, Adam optimisation, and training-only early stopping. Its saved state includes fitted parameters and scalers so new score files can be evaluated without refitting.

The source code does not train WavLM, HuBERT, ECAPA-TDNN, AASIST, or a speech-emotion recogniser. It begins after those upstream systems would have produced scores.

### Command-line and synthetic demo

[src/easv/cli.py](src/easv/cli.py) exposes audit, evaluation, demonstration, prediction, and manifest-template functions. [src/easv/demo.py](src/easv/demo.py) produces a synthetic end-to-end demonstration that simulates speaker-disjoint data roles, attack systems, multiple emotions, a held-out emotion, uncertainty variation, and every fusion arm.

The synthetic demo exercises the audit, training, metric, and reporting paths. It was designed for software verification, not for approximating a public speech corpus.

## Executed verification

### Automated test suite

The supplied test suite was executed with the repository virtual environment:

    ./.venv/bin/python -m pytest -q

Observed result:

    65 tests passed

The tests cover the project’s implemented functions, including manifests, score alignment, fusion fitting and serialisation, metric behaviour, command-line operations, and evaluation output. Passing tests mean that the code passed its encoded checks. They do not demonstrate performance on unseen human speech or real spoofing attacks.

### End-to-end synthetic demonstration

The demonstration was executed with:

    ./.venv/bin/python -m easv demo --out /tmp/easv-course-report/demo --config configs/demo.yaml

Observed result:

- 1,660 synthetic trials were generated;
- every fusion arm was fitted;
- evaluation reports were produced.

The generated test split contained:

| Measure | Observed value |
|---|---:|
| Target trials | 120 |
| Nontarget trials | 120 |
| Spoof trials | 240 |
| Speakers | 20 |
| Emotion conditions | 6 |
| Unseen test attack systems | 2 |

### Manifest audit of the generated data

The synthetic manifest was then audited:

    ./.venv/bin/python -m easv audit --manifest /tmp/easv-course-report/demo/data/manifest.csv --out /tmp/easv-course-report/audit

Observed result:

| Audit result | Count |
|---|---:|
| Errors | 0 |
| Warnings | 1 |

The one warning identifies the held-out emotion, surprise, in the test split. In the supplied demo this is intentional: it exercises cross-emotion reporting. A real experiment would need to disclose whether this test condition was planned and whether the upstream emotion recogniser had seen that category during its own training.

## Synthetic demonstration results

The following values came from the synthetic repository demo. They are not real-speech benchmark results:

| Model | Synthetic SASV-EER | 95% confidence interval | Minimum a-DCF | Parameters |
|---|---:|---|---:|---:|
| Speaker-only | 0.3583 | 0.3152–0.3833 | 0.9025 | 2 |
| Linear baseline | 0.3167 | 0.2819–0.3417 | 0.8328 | 3 |
| Emotion additive | 0.3083 | 0.2750–0.3460 | 0.8236 | 10 |
| Emotion interaction | 0.2833 | 0.2417–0.3167 | 0.7583 | 25 |
| Emotion gated | 0.2917 | 0.2583–0.3250 | 0.7207 | 47 |

Interpretation of these generated outputs:

- The interaction model had the lowest synthetic SASV-EER.
- The gated model had the lowest synthetic minimum a-DCF.
- The gated-versus-linear paired result had a mean gain of 0.0210, with confidence interval from -0.0083 to 0.0472.
- Because that interval crosses zero, it does not support a confirmed gain under the project’s stated comparison rule, even within the synthetic demonstration.
- At the development-selected threshold, the synthetic gated model had spoof-acceptance rates of 0.442 for the simulated TTS generator and 0.433 for the simulated voice-conversion generator.

The synthetic generator intentionally varies countermeasure reliability by emotion and uncertainty. That is useful for testing the intended code path, but the gated model is trained under that designed structure. The result cannot independently demonstrate practical security or superior performance.

## Literature and novelty review

The replacement report contains 35 references. The literature review is organised by these technical themes:

1. Speaker-verification representations;
2. Self-supervised speech representations;
3. Spoofing and countermeasure research;
4. Integrated SASV and score fusion;
5. Emotion-aware speech and the remaining question;
6. Evaluation discipline and reproducibility.

The references and citations address foundational work in factor analysis, x-vectors, ECAPA-TDNN, WavLM, HuBERT, wav2vec 2.0, RawNet2, AASIST, ASVspoof, SASV, t-DCF, a-DCF, emotion-related deepfake detection, emotional anti-spoofing, and style-dependent speaker verification.

The repository’s novelty gate remains open. The defensible status is:

- emotion may affect speaker-verification and anti-spoofing reliability;
- emotion-conditioned score fusion is a reasonable experimental hypothesis;
- the code supplies a leakage-aware way to evaluate it;
- the reviewed materials do not establish that the hypothesis is novel.

Before claiming novelty, the closest studies must be compared on exact input features, fusion point, upstream models, datasets, label taxonomy, attack taxonomy, splits, metrics, generalisation conditions, and statistical support. The existing [docs/NOVELTY_MATRIX.md](docs/NOVELTY_MATRIX.md) is the appropriate record for this work. Its existing user modification was preserved.

## Completed work

### Repository status

The supplied repository demonstrably includes:

- manifest handling;
- audit and leakage-oriented reports;
- score alignment;
- a controlled score-fusion ladder;
- metric and uncertainty reporting;
- command-line operations;
- automated tests;
- an end-to-end synthetic smoke test.

### Report status

The completed Word report includes:

- a formal cover and certificate;
- a 195-word abstract;
- a contents page with final pagination;
- a two-page narrative introduction;
- a themed literature review;
- a three-page methodology with flow and data table;
- a three-page results-and-discussion chapter with comparison table, bar chart, split table, and limitations;
- a conclusion;
- 35 references;
- an appendix recording the audit trail and executed commands.

### Visual and file validation

The final document was rendered to page images after layout changes and inspected visually. The final version:

- contains 19 pages;
- has no report placeholders;
- has a coherent contents page;
- keeps the principal comparison table intact;
- repeats the header for the split summary table when it continues to a new page;
- has a valid OOXML package: the archive integrity check reported no errors.

## Important limitations as of that snapshot (2026-10-21)

These were true of the supplied repository at the time of this audit. Rows marked
*closed* have since been addressed; the rest remain open and are tracked at the end
of [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md).

| Not demonstrated | Why it matters |
|---|---|
| Real speech-corpus experiment | Synthetic scores cannot establish real-world performance. **Closed** — 53,900 real trials executed. |
| ASV audio-model training or inference | The framework begins from pre-computed speaker scores. **Closed** — ECAPA-TDNN producer. |
| Anti-spoofing audio-model training or inference | The framework begins from pre-computed countermeasure scores. **Closed** — four regimes. |
| Emotion-recogniser training or inference | The framework accepts probabilities but does not produce them from audio. **Closed** — wav2vec2-SER producer. |
| Benchmark reproduction on ASVspoof, SASV, or another corpus | No approved corpus, score producers, or benchmark run is bundled. **Partly closed**: ASVspoof 2019 LA train used, EASV corpus built; no published baseline reproduced. |
| Security or accuracy improvement in deployment | The current results are generated by a designed simulation. |
| Verified novelty | The closest literature has not yet been fully compared. |
| Confirmed gated-model improvement over linear fusion | The generated paired confidence interval crosses zero. |

## Recommended next steps

1. Read the closest emotion-aware anti-spoofing and style-aware speaker-verification papers in full, then complete the novelty matrix.
2. Select licensed real speech, spoof, and emotion resources; record their versions, licences, label taxonomy, conditions, and attack types.
3. Choose reproducible speaker, countermeasure, and emotion score producers; document checkpoints, preprocessing, score direction, and calibration.
4. Define speaker-disjointness, attack-generator generalisation, and cross-emotion conditions before model selection.
5. Predefine success criteria, baseline, operating-point rules, and confidence-interval requirements before looking at final test outputs.
6. Run the manifest audit before fitting; resolve unexplained errors and record intentional warnings.
7. Fit on training data only, select thresholds on development data, and evaluate the test split once.
8. Report results by emotion and attack system, with calibration, uncertainty, and real operating-point failure modes.
9. Replace the synthetic results chapter only after the real-data protocol has been completed.

## Deliverables

| File | Description |
|---|---|
| [Course_Project_Report_EASV_Audited.docx](Course_Project_Report_EASV_Audited.docx) | Final 19-page course report based on the supplied template and actual repository audit. |
| [WORK_COMPLETION_AUDIT.md](WORK_COMPLETION_AUDIT.md) | This detailed work record. |

## Integrity statement

This audit is based on the code and synthetic outputs available on 2026-10-21. It avoids fabricated corpus experiments, benchmark results, novelty claims, and deployment-readiness assertions. The resulting report describes a working research infrastructure and an honest current evidence level. A research-paper manuscript claiming a new high-performing system should be prepared only after the real-data study has been run.
