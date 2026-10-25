"""Maintain the real-data section of the course-report DOCX.

The report was first written when every number came from the synthetic demo. This
script keeps its real-data section honest as the project moves: it replaces the
abstract that denies real data exists, refreshes the component table and the
section introduction, adds the three-generator regime, the ASVspoof-2019 LA result
and the held-out-emotion folds once they exist, and rewrites the closing claims so
they match the measured outcomes. It updates an existing section 4.4 rather than creating one, so the
document and the script cannot drift into two different versions.

Idempotent: every function checks for its own marker and re-running is safe.

Usage:
    .venv/bin/python scripts/update_report_real_results.py \
        [--docx Course_Project_Report_EASV_Audited.docx] \
        [--la-metrics /home/vct/data/easv/real/cm_aasist_la/metrics.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import docx
from docx.shared import Pt


BODY, SMALL = 12, 9


CROSSGEN_MARKER = "A third regime tests generalisation"
THREE_GEN_MARKER = "fourth regime"
LA_MARKER = "ASVspoof 2019 LA training set"


def _set_text(paragraph, text: str) -> None:
    """Replace a paragraph's text without touching the document's styles."""
    for run in paragraph.runs:
        run.text = ""
    paragraph.runs[0].text = text


def add_three_generator(doc) -> bool:
    """Add the completed three-generator regime and correct the earlier numbers."""
    if any(THREE_GEN_MARKER in p.text for p in doc.paragraphs):
        return False

    anchor = next((p for p in doc.paragraphs
                   if p.text.strip().startswith("A third regime tests generalisation")), None)
    if anchor is None:
        raise SystemExit("could not find the cross-generator paragraph to anchor against")

    lead = doc.add_paragraph()
    anchor._p.addnext(lead._p)
    run = lead.add_run(
        "A fourth regime uses the completed three-generator corpus and the countermeasure that "
        "treats StyleTTS2 as an unseen system: it is trained on the ESD bona fide, F5-TTS and "
        "CosyVoice trials only, with StyleTTS2 excluded from training and from development-based "
        "early stopping, and it detects the unseen generator at 2.50 % EER (1.60 % overall). "
        "Under this countermeasure the arm ladder is 27.1 % (speaker only), 2.27 % (linear), "
        "2.51 % (additive), 2.09 % (interaction) and 1.54 % (gated) SASV-EER. The gated arm "
        "beats linear by 0.68 points (95 % CI 0.15-1.12, p = 0.002) and, in a paired contrast the "
        "evaluation CLI does not run by default, the interaction control by 0.48 points "
        "(0.06-0.99). This is the one regime in which the proposed arm wins, and it wins "
        "significantly.")
    run.font.name = "Times New Roman"
    run.font.size = Pt(BODY)
    lead.paragraph_format.space_after = Pt(6)

    caveat = doc.add_paragraph()
    lead._p.addnext(caveat._p)
    run = caveat.add_run(
        "Two qualifications keep that result honest. First, the learned gate weights are nearly "
        "identical across all five emotion conditions (speaker weight 4.02-4.05, countermeasure "
        "weight 9.69-9.73, a spread below 0.5 %), so the model is not conditioning its decision "
        "on emotion; the gain comes from a better-fitting, better-calibrated decision function, "
        "and the same collapse appears in every other regime, including an exact tie across "
        "emotions in the in-domain regime. Second, the arm's edge over the interaction control is "
        "only significant once the third generator is present, so the two-generator and "
        "three-generator results have to be reported side by side rather than merged, and the "
        "paper's mechanism claim remains unsupported even where its predictive arm wins.")
    run.font.name = "Times New Roman"
    run.font.size = Pt(BODY)
    caveat.paragraph_format.space_after = Pt(6)

    # Correct the paragraph that still quoted the two-generator aggregate.
    for p in doc.paragraphs:
        if p.text.strip().startswith("Three results matter. First, the vulnerability is real"):
            _set_text(p, (
                "Three results matter. First, the vulnerability is real and both emotion- and "
                "generator-dependent. On the complete corpus (53,900 trials: 17,900 ESD bona fide "
                "plus 36,000 clones from StyleTTS2, F5-TTS and CosyVoice) the speaker front-end "
                "accepts 82.9 % of bona fide targets and 38.8 % of clones at the development-"
                "selected threshold, but the aggregate hides a generator effect: CosyVoice 59.0 %, "
                "F5-TTS 51.5 % and StyleTTS2 5.9 %. Inside each generator acceptance still follows "
                "emotion (angry 45.6 %, neutral 43.3 %, sad 33.2 %, happy 33.1 %). Second, "
                "conventional fusion removes almost all of that vulnerability when the countermeasure "
                "works, taking SASV-EER from 33.1 % to 0.60 % on the two-generator corpus. Third, "
                "on that corpus the proposed emotion-gated arm does not beat plain fusion: it is "
                "0.14 points worse and its paired bootstrap interval does not exclude zero "
                "(p = 0.98), so the hypothesis is not supported in the regime where the "
                "countermeasure already works. The completion of the third generator, reported "
                "below, changes that last statement only in the corpus where the countermeasure is "
                "deliberately kept away from one generator."))
            break
    return True


def add_la_aasist(doc, metrics_path: Path) -> bool:
    """Add the in-domain AASIST retrained on ASVspoof 2019 LA, if it has finished."""
    import json

    if not metrics_path.exists():
        return False
    if any(LA_MARKER in p.text for p in doc.paragraphs):
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "easv_test_cm_eer" not in metrics:
        return False

    anchor = next((p for p in doc.paragraphs
                   if p.text.strip().startswith("4.5 What has and has not")), None)
    if anchor is None:
        return False

    pct = lambda v: f"{100 * v:.2f}"  # noqa: E731
    text = (
        f"Finally, the ASVspoof 2019 LA training set ({metrics['n_la_train']:,} utterances, "
        f"{metrics['n_la_dev']:,} held out by speaker) was used to retrain the published AASIST "
        "architecture in-domain, so that the countermeasure's failure could not be blamed on a "
        f"missing checkpoint. The selected model reaches {pct(metrics['la_dev_cm_eer'])} % EER on "
        f"held-out LA speakers, i.e. it is a working LA countermeasure, yet it scores "
        f"{pct(metrics['easv_test_cm_eer'])} % SPF-EER on the EASV test split "
        f"({pct(metrics['easv_dev_cm_eer'])} % on dev). The negative transfer result therefore "
        "does not depend on which LA checkpoint is used: ASVspoof-2019 supervision, in either "
        "form, does not describe these neural clones, while a log-mel countermeasure trained on "
        "two adjacent generators detects the third at 2.50 % EER.")
    new = doc.add_paragraph()
    anchor._p.addprevious(new._p)
    run = new.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(BODY)
    new.paragraph_format.space_after = Pt(6)
    return True


HELDOUT_EMOTION_MARKER = "Two further folds test the cross-emotion claim directly"


def add_heldout_emotion(doc) -> bool:
    """Add the held-out-emotion folds to section 4.4.

    Placed after the LA paragraph when it exists, otherwise directly before
    section 4.5, so the two insertions do not fight over the same position.
    """
    if any(HELDOUT_EMOTION_MARKER in p.text for p in doc.paragraphs):
        return False
    la = next((p for p in doc.paragraphs
               if p.text.strip().startswith("Finally, the ASVspoof 2019 LA training set")), None)
    heading = next((p for p in doc.paragraphs
                    if p.text.strip().startswith("4.5 What has and has not")), None)
    if la is None and heading is None:
        raise SystemExit("could not find an anchor for the held-out-emotion paragraph")

    new = doc.add_paragraph()
    if la is not None:
        la._p.addnext(new._p)
    else:
        heading._p.addprevious(new._p)
    run = new.add_run(
        "Two further folds test the cross-emotion claim directly. In each, one emotion (angry, "
        "then neutral) is removed from every stage that could learn it: the countermeasure's "
        "training and development-based early stopping, and the fusion fit's train and dev "
        "splits, while the test split is left intact, so the held-out trials are scored by a "
        "pipeline that never encountered that emotion. Removing it costs nothing. The "
        "countermeasure detects the held-out emotion at 0.03 % EER (0.08 % overall), and on "
        "that emotion's 5,040 test trials the largest change in SASV-EER between fitting with "
        "and without it is 0.03 points, with every paired bootstrap interval containing zero "
        "(gated, angry: −0.01 points, interval −0.06 to 0.00; gated, neutral: +0.03, interval "
        "0.00 to 0.13). The pipeline therefore generalises to an emotion it has never seen. The "
        "reason is not that the emotion features generalise well but that they carry little "
        "emotion-specific structure: an arm that genuinely conditioned on emotion would lose "
        "something when an entire condition is removed, and these lose nothing measurable. "
        "That is the third independent line of evidence against the mechanism, alongside the "
        "flat gate weights and the confidence-scalar reproduction.")
    run.font.name = "Times New Roman"
    run.font.size = Pt(BODY)
    new.paragraph_format.space_after = Pt(6)
    return True


def add_crossgenerator(doc) -> None:
    """Append the leave-one-generator-out paragraph once, after the Regime B text."""
    if any(CROSSGEN_MARKER in p.text for p in doc.paragraphs):
        return
    anchor = next((p for p in doc.paragraphs
                   if p.text.strip().startswith("The second regime explains why")), None)
    if anchor is None:
        raise SystemExit("could not find the Regime B paragraph to anchor against")
    new = doc.add_paragraph()
    anchor._p.addnext(new._p)  # move it directly after the Regime B paragraph
    run = new.add_run(
        "A third regime tests generalisation instead of assuming it. The countermeasure was "
        "retrained with the F5-TTS generator excluded from training and from the development-"
        "based early stopping, so every F5-TTS clone is scored as an unseen attack system. That "
        "model reaches 6.27 % EER on the held-out generator (3.19 % on the generator it trained "
        "on, 4.28 % overall), against 50.8 % for the ASVspoof-pretrained AASIST: training on an "
        "adjacent neural generator transfers an order of magnitude better than the ASVspoof "
        "lineage. Under this countermeasure the fusion arms reach 4.11 % (linear), 3.81 % "
        "(interaction, +0.38 points, p < 0.001) and 3.93 % (gated, +0.27 points with an interval "
        "from -0.23 to +0.86, p = 0.23) SASV-EER. The proposed arm therefore still does not "
        "clear the significance bar even in the cross-generator condition, which is the condition "
        "most favourable to its original motivation.")
    run.font.name = "Times New Roman"
    run.font.size = Pt(BODY)
    new.paragraph_format.space_after = Pt(6)


def update_abstract(doc) -> bool:
    """Replace the synthetic-only abstract with the measured results.

    The original abstract was written before any real corpus existed and states,
    incorrectly, that no real corpus or pretrained backend is included. Leaving
    it would misrepresent the whole report.
    """
    for p in doc.paragraphs:
        if not p.text.strip().startswith("Voice authentication is useful only when"):
            continue
        if "82.9 % of targets and 38.8 % of clones" in p.text and len(p.text.split()) < 200:
            return False
        _set_text(p, (
            "Voice authentication must distinguish a genuine claimant from an impostor and "
            "from a synthetic voice. This project implements, audits and executes a score-level "
            "pipeline for emotion-conditioned spoofing-aware speaker verification on real "
            "speech: 53,900 speaker-disjoint trials combining ESD English bona fide recordings "
            "with 36,000 EmoSpoof-TTS clones from three generators. Speaker, emotion and "
            "countermeasure scores come from ECAPA-TDNN, wav2vec2-superb-ER and a log-mel CNN, "
            "with ASVspoof-2019 AASIST variants as transfer controls. Fusion is fitted on train, "
            "thresholds are chosen on dev, and test is reported once with grouped-bootstrap "
            "intervals. The speaker front-end accepts 82.9 % of targets and 38.8 % of clones, "
            "strongly generator-dependent (CosyVoice 59.0 %, F5-TTS 51.5 %, StyleTTS2 5.9 %). "
            "Conventional fusion cuts SASV-EER from 33.1 % to 0.60 % when the countermeasure "
            "works, while ASVspoof-2019 countermeasures do not transfer to these clones. With "
            "StyleTTS2 held out, the proposed gated arm is the best model (1.54 % against "
            "2.27 %), but its learned gate weights are flat across emotions, so the gain cannot "
            "be attributed to emotion conditioning. The report states these results and their "
            "limits; no novelty over published systems is claimed."))
        return True
    return False


def update_toc_pages(doc, pages: dict[str, str]) -> bool:
    """Re-point the static contents table after the real-data text was added.

    The contents page is a plain table, not a Word TOC field, so it does not
    update itself. Chapters 1-4 start on unchanged pages (the additions are all
    inside chapter 4); chapter 5, references and appendix move. The numbers are
    derived from the added word count at the document's measured words-per-page
    and should be confirmed in Word before submission.
    """
    for table in doc.tables:
        if not table.rows or table.rows[0].cells[0].text.strip() != "Abstract":
            continue
        for row in table.rows:
            key = row.cells[0].text.strip()
            for prefix, page in pages.items():
                if key.startswith(prefix):
                    row.cells[1].text = page
        return True
    return False


def update_ser_numbers(doc) -> None:
    """Give both SER measurements instead of the two-generator one alone."""
    for p in doc.paragraphs:
        if not p.text.strip().startswith("The second regime explains"):
            continue
        _set_text(p, (
            "The second regime explains why an apparent gain must be read carefully. The "
            "pretrained AASIST checkpoint, trained on ASVspoof-2019 attack families, scores at "
            "chance on these clones (SPF-EER 50.8 %). When that failing countermeasure is used, "
            "the emotion arms improve SASV-EER by about 5.5 points with p < 0.001. The mechanism "
            "is not emotion conditioning: the emotion recogniser is confidently wrong on "
            "synthetic speech — 30.7 % accuracy at 0.83 mean confidence on the two-generator "
            "corpus, 32.8 % at 0.82 on the full three-generator corpus, against 41.2 % on "
            "genuine speech — so its probability vector implicitly encodes whether the audio is "
            "synthetic. Emotion features act as a substitute spoof detector, which is a confound "
            "rather than the proposed effect. That was tested directly: replacing the four-way "
            "emotion vector with the single maximum predicted probability, while keeping the "
            "fusion ladder identical, recovers +1.6 of the +5.4 points (95 % CI 0.70-2.74, "
            "p < 0.001) on this regime, and the full vector remains significantly better than "
            "that scalar (−3.7 points, CI −7.68 to −0.78). Roughly a third of the apparent "
            "gain is therefore a "
            "spoof cue, and the remainder is the per-class structure of an out-of-domain "
            "emotion output rather than demonstrated emotion semantics. In the three-generator "
            "regime the same ablation is decisive against the mechanism: the gated arm reaches "
            "1.49 % with the scalar alone against 1.54 % with the full vector, a difference of "
            "0.02 points whose interval spans zero (p = 0.76), so its advantage does not "
            "require emotion information at all."))
        break


def update_closing_paragraphs(doc) -> None:
    """Replace the two closing paragraphs that still describe future work only."""
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith(("Future work should close the literature gate",
                         "Future work should close what this study leaves open")):
            _set_text(p, (
                "Future work should close what this study leaves open: a held-out-emotion "
                "fold against a failing countermeasure, since the two folds reported here were "
                "run only where the countermeasure already works; a "
                "pre-registered decision table, because the success margins were recorded only "
                "after the test report; the two ablation overlays on real data; repeated seeds "
                "and a second speaker split; reproduction of a published baseline under a "
                "matched protocol, which would also validate the metric code against reference "
                "numbers; and the ASVspoof 2019 evaluation partition, which would give the "
                "retrained countermeasure a standard benchmark instead of two held-out speakers. "
                "The ESD research licence also has to be settled before publication. The "
                "evidence gathered so far already points to the third of the original "
                "alternatives: the gating advantage is limited to the cross-generator corpus and "
                "is explained by capacity and calibration rather than by emotion conditioning."))
        elif t.startswith("This project implemented and audited a score-level research"):
            _set_text(p, (
                "This project implemented and audited a score-level research infrastructure for "
                "emotion-conditioned spoofing-aware speaker verification, and then executed it "
                "on real speech. Its practical contribution at this stage is methodological: it "
                "separates manifest truth from model inputs, checks leakage risks, preserves "
                "train/dev/test roles, compares a transparent fusion ladder, and records "
                "uncertainty and group-level behaviour. The full automated test suite passed, "
                "the synthetic demonstration completed as a pipeline smoke test, and the real "
                "corpus produced the measured component, fusion and transfer results in "
                "section 4.4."))


def update_audit_paragraph(doc) -> None:
    """Keep section 4.1's test count and scope statement current."""
    for p in doc.paragraphs:
        if not p.text.strip().startswith("The repository was executed in its supplied"):
            continue
        _set_text(p, (
            "The repository was executed in its supplied virtual environment. The full "
            "automated suite completed successfully: 83 tests passed. The tests cover "
            "manifest validation, score alignment, metric direction conventions, threshold "
            "separation, fusion serialization, grouped evaluation, command-line behaviour, "
            "the manifest builders, the arm-vs-arm comparison sign convention, the "
            "SER-confidence feature mode, and the held-out-emotion fold (that it leaves the "
            "test split intact and rejects an unknown emotion). Passing "
            "tests are evidence that the implemented functions meet their encoded "
            "expectations; they are not evidence that a biometric system works on unseen "
            "human speech, which is why section 4.4 executes the full pipeline on real audio."))
        break


def update_reproducibility_note(doc) -> None:
    """Replace the note that implies no real experiment exists."""
    for p in doc.paragraphs:
        if not p.text.strip().startswith("The synthetic demo should not be cited"):
            continue
        _set_text(p, (
            "The synthetic demo should not be cited as a benchmark result. Section 4.4 is the "
            "real experiment: externally sourced speech, documented score producers, metric "
            "definitions and split rules fixed before the test report, and a test split "
            "evaluated once. (The numeric success margins were not pre-registered before the "
            "first test report; PROTOCOL section 6 records that limitation explicitly.) Its "
            "per-trial scores, fitted models and reports live in results/real/ (git-ignored "
            "because they encode corpus paths) and the audio corpora live outside the "
            "repository; docs/REAL_RESULTS.md records every number with its split, threshold "
            "policy and interval, and docs/PROGRESS_LOG.md records how the corpora were "
            "obtained. The synthetic demonstration is kept only as the pipeline smoke test."))
        break


def update_appendix(doc) -> None:
    """Record the commands that actually produced the reported numbers."""
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("1. .venv/bin/python -m pytest -q"):
            _set_text(p, (
                "1. .venv/bin/python -m pytest -q  ->  83 tests passed.\n"
                "2. .venv/bin/python -m easv demo --out <temporary directory> --config "
                "configs/demo.yaml  ->  1,660 synthetic trials and five fitted models "
                "(pipeline smoke test only).\n"
                "3. .venv/bin/python -m easv audit --manifest results/e2e/merged_manifest.csv "
                "--out results/e2e/audit_real  ->  0 errors, 4 expected warnings on the "
                "53,900-trial real manifest.\n"
                "4. scripts/producer_asv.py and scripts/producer_ser.py over the merged "
                "manifest (53,900 score rows each); scripts/train_cm_indomain.py for the "
                "in-domain and StyleTTS2-held-out countermeasures; scripts/producer_cm_aasist.py "
                "for the pretrained checkpoint; scripts/train_cm_aasist_la.py plus "
                "scripts/build_manifests.py asvspoof for the LA retrain.\n"
                "5. .venv/bin/python -m easv.cli evaluate --manifest results/e2e/merged_manifest.csv "
                "--scores <combined scores> --config configs/real_run.yaml --out "
                "results/real/run4_3gen_crossgen  ->  the three-generator tables in 4.4.\n"
                "6. .venv/bin/python scripts/compare_arms.py --results "
                "results/real/run4_3gen_crossgen --manifest results/e2e/merged_manifest.csv "
                "--a emotion_gated --b emotion_interaction  ->  the arm-vs-arm contrast.\n"
                "7. scripts/train_cm_indomain.py --holdout-emotion <angry|neutral>, then "
                "easv.cli evaluate with and without --set folds.holdout_emotion=<emotion>, then "
                "scripts/compare_arms.py --emotion <emotion>  ->  the held-out-emotion folds."))
            break


def update_intro(doc) -> None:
    """Bring the section introduction up to the completed corpus."""
    for p in doc.paragraphs:
        if not p.text.strip().startswith("The pipeline was then executed on real speech"):
            continue
        _set_text(p, (
            "The pipeline was then executed on real speech. Bona fide trials come from the ESD "
            "English subset (speakers 0011-0020, emotion labels from the corpus folders); spoof "
            "trials come from the EmoSpoof-TTS generators StyleTTS2, F5-TTS and CosyVoice (12,000 "
            "utterances each). After the audit and the speaker-disjoint 4/2/4 split, the merged "
            "manifest holds 53,900 trials: 17,500 bona fide targets, 400 derived cross-speaker "
            "nontargets and 36,000 clones. Scores were produced by SpeechBrain ECAPA-TDNN "
            "(speaker), wav2vec2-base-superb-ER (emotion) and countermeasures: the official "
            "ASVspoof-2019 AASIST checkpoint, a log-mel CNN trained on the train split, a "
            "log-mel CNN trained with one generator held out, and an AASIST retrained in-domain "
            "on ASVspoof 2019 LA audio. All thresholds were chosen on the development split; the "
            "test split was reported once."))
        return


def dedupe(doc, prefix: str) -> int:
    """Drop repeated copies of a paragraph, keeping the first occurrence.

    An earlier version of the cross-generator guard looked for a marker the
    inserted text did not contain, so every invocation appended another copy.
    This repairs such documents and keeps the function idempotent.
    """
    seen = False
    removed = 0
    for p in list(doc.paragraphs):
        if not p.text.strip().startswith(prefix):
            continue
        if seen:
            p._element.getparent().remove(p._element)
            removed += 1
        seen = True
    return removed


def update_component_table(doc) -> bool:
    """Refresh Table 3 with the three-generator numbers and the retrained CM."""
    target = None
    for table in doc.tables:
        header = [c.text.strip() for c in table.rows[0].cells]
        if len(header) >= 4 and header[0].lower() == "component":
            target = table
            break
    if target is None:
        return False
    already = any("retrained" in row.cells[0].text for row in target.rows)

    for row in target.rows[1:]:
        name = row.cells[0].text.strip()
        if name.startswith("CM (in-domain"):
            row.cells[3].text = ("optimistic by construction: it trains on the same "
                                 "generators it is scored on")
        elif name == "ASV":
            row.cells[3].text = ("accepts 82.9 % of targets and 38.8 % of clones "
                                 "(CosyVoice 59.0 %, F5-TTS 51.5 %, StyleTTS2 5.9 %)")
        elif name == "SER":
            row.cells[2].text = "41.2 % / 32.8 % accuracy"
            row.cells[3].text = ("bona fide / clones; clones saturate it at 0.82 "
                                 "confidence")
    if not already:
        cells = target.add_row().cells
        values = ["CM (AASIST-LA retrained)", "AASIST trained on LA train audio",
                  "2.11 % SPOOF-EER on held-out LA",
                  "43.8 % SPOOF-EER on EASV test: no transfer either"]
        for cell, text in zip(cells, values):
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(SMALL)
            run.font.name = "Times New Roman"
    return True


def update_conclusions(doc) -> None:
    """Refresh the two closing claims that the three-generator run has overtaken."""
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("The completed work demonstrates a reproducible score-level"):
            _set_text(p, (
                "The completed work demonstrates a reproducible score-level evaluation pipeline, an "
                "explicit fusion ablation ladder, data audits, metric implementations, trained ASV, "
                "SER and countermeasure front-ends, and a real experiment on ESD and EmoSpoof-TTS "
                "data spanning three generators. What it does not show is that the proposed "
                "emotion-conditioned mechanism is better than conventional fusion because of its "
                "emotion conditioning: the gated arm does beat both controls on the three-generator "
                "corpus, but its learned gate weights are flat across emotion conditions, so the "
                "gain is attributable to the decision function rather than to the mechanism, and "
                "on the two-generator corpus the same arm is worse than the baseline. It also does "
                "not show a cross-generator advantage in the two-generator setting. Each of those "
                "gaps can change the conclusion, so the mixed result in section 4.4 stands as the "
                "finding rather than as a placeholder."))
        elif t.startswith("The synthetic run provides no basis for claiming real authentication gains"):
            _set_text(p, (
                "The synthetic run provides no basis for claiming real authentication gains, and "
                "the real runs give a mixed verdict that is reported as measured. On the held-out "
                "test split, plain fusion cut SASV-EER from 33.1 % to 0.60 % when the "
                "countermeasure worked, which is a genuine security improvement over ASV alone, "
                "while the proposed arm was slightly worse there. On the complete three-generator "
                "corpus with a countermeasure that has never seen the third generator, the proposed "
                "arm is the best model (1.54 % against 2.27 % for plain fusion), significantly so "
                "over both the interaction control (+0.48 points) and the capacity control "
                "(+0.85 points), but its gate weights do not vary with emotion. The defensible "
                "claim is therefore narrower than the original hypothesis: emotion-conditioned "
                "architecture choices can improve spoofing-aware fusion on a hard, "
                "cross-generator corpus, but this data does not show that the improvement comes "
                "from using emotion. The held-out-emotion folds remove the last alternative "
                "reading: excluding an entire emotion from the countermeasure's training and from "
                "the fusion fit leaves that emotion's error rate unchanged to within 0.03 points, "
                "so the pipeline is not relying on memorised emotion identities either, and it "
                "loses nothing because it was hardly using them. The failure of ASVspoof-2019 "
                "countermeasures to transfer to "
                "modern neural clones (50.8 % EER pretrained, 43.8 % retrained in-domain) is arguably "
                "the most useful finding, because it shows that countermeasure generalisation, not "
                "fusion architecture, is the binding constraint."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", type=Path,
                    default=Path("Course_Project_Report_EASV_Audited.docx"))
    ap.add_argument("--la-metrics", type=Path,
                    default=Path("/home/vct/data/easv/real/cm_aasist_la/metrics.json"))
    args = ap.parse_args()

    doc = docx.Document(str(args.docx))
    texts = [p.text.strip() for p in doc.paragraphs]
    if not any(t.startswith("4.4 Real-data results") for t in texts):
        raise SystemExit(
            "this report has no section 4.4; this script maintains an existing real-data "
            "section so the two versions cannot drift. Regenerate the section from the course "
            "template first, then re-run.")

    update_abstract(doc)
    update_intro(doc)
    update_audit_paragraph(doc)
    update_ser_numbers(doc)
    update_closing_paragraphs(doc)
    update_toc_pages(doc, {"Chapter 5": "18", "References": "19", "Appendix": "21"})
    update_reproducibility_note(doc)
    update_appendix(doc)
    added = add_three_generator(doc)
    added_la = add_la_aasist(doc, args.la_metrics)
    added_fold = add_heldout_emotion(doc)
    update_component_table(doc)
    update_conclusions(doc)
    removed = dedupe(doc, CROSSGEN_MARKER)
    doc.save(str(args.docx))
    print(f"updated {args.docx} (three-generator={added}, la-aasist={added_la}, "
          f"held-out-emotion={added_fold}, removed-duplicates={removed})")


if __name__ == "__main__":
    main()
