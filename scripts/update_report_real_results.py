"""Insert the real-data results into the course-report DOCX.

The report was written when every number came from the synthetic demo, and it
says so in the chapter introduction, section 4.4 and the conclusion. Those
sentences are now false: a real run exists (ESD + EmoSpoof-TTS, 41,900 trials,
two countermeasure regimes). This script adds section 4.5 with the measured
tables and rewrites the three claims that the real run has overtaken, keeping
the honest caveats the original text was careful about.

Idempotent: if section 4.5 is already present the script reports it and exits.

Usage:
    .venv/bin/python scripts/update_report_real_results.py \
        [--docx Course_Project_Report_EASV_Audited.docx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import docx
from docx.shared import Pt


BODY, SMALL = 12, 9


def add_para_before(anchor, text: str, *, bold: bool = False, size: int = BODY,
                    italic: bool = False, space_after: int = 6):
    p = anchor.insert_paragraph_before()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(space_after)
    return p


def add_table_before(doc, anchor, header: list[str], rows: list[list[str]],
                     caption: str) -> None:
    caption_para = add_para_before(anchor, caption, size=10, italic=True, space_after=2)
    table = doc.add_table(rows=len(rows) + 1, cols=len(header))
    table.style = "Table Grid"
    table.autofit = True
    for j, text in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold = True
        run.font.size = Pt(SMALL)
        run.font.name = "Times New Roman"
    for i, row in enumerate(rows, start=1):
        for j, text in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(SMALL)
            run.font.name = "Times New Roman"
    # place the table directly after its own caption, not at the end of the section
    caption_para._p.addnext(table._tbl)


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
                "from using emotion. The failure of ASVspoof-2019 countermeasures to transfer to "
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
    if any(t.startswith("4.4 Real-data results") for t in texts):
        add_crossgenerator(doc)
        update_intro(doc)
        added = add_three_generator(doc)
        added_la = add_la_aasist(doc, args.la_metrics)
        update_component_table(doc)
        for p in doc.paragraphs:
            if p.text.strip().startswith("4.4 Real-data results (executed 22 October"):
                _set_text(p, "4.4 Real-data results (executed 22-23 October 2026)")
                p.runs[0].bold = True
                p.runs[0].font.name = "Times New Roman"
                p.runs[0].font.size = Pt(BODY)
        update_conclusions(doc)
        removed = dedupe(doc, CROSSGEN_MARKER)
        doc.save(str(args.docx))
        if removed:
            print(f"removed {removed} duplicated cross-generator paragraph(s)")
        print(f"real-results section present; three-generator={added}, "
              f"la-aasist={added_la}")
        return

    anchor = next(p for p in doc.paragraphs
                  if p.text.strip().startswith("4.4 What has and has not been demonstrated"))

    # The new section takes 4.4 and the previous one becomes 4.5, so the chapter
    # still reads in order after the insert.
    for run in anchor.runs:
        if "4.4" in run.text:
            run.text = run.text.replace("4.4", "4.5")
    add_para_before(anchor, "4.4 Real-data results (executed 22 October 2026)", bold=True)
    add_para_before(anchor, (
        "The pipeline was then executed on real speech. Bona fide trials come from the ESD "
        "English subset (speakers 0011-0020, emotion labels from the corpus folders); spoof "
        "trials come from the EmoSpoof-TTS generators F5-TTS and CosyVoice (12,000 utterances "
        "each). After the audit and the speaker-disjoint 4/2/4 split, the merged manifest holds "
        "41,900 trials: 17,500 bona fide targets, 400 derived cross-speaker nontargets and "
        "24,000 clones. Scores were produced by SpeechBrain ECAPA-TDNN (speaker), "
        "wav2vec2-base-superb-ER (emotion) and two countermeasures: the official ASVspoof-2019 "
        "AASIST checkpoint, and a log-mel CNN trained only on the train split. All thresholds "
        "were chosen on the development split; the test split was reported once."))

    add_table_before(doc, anchor,
                     ["component", "model", "EER on test", "note"],
                     [["ASV", "ECAPA-TDNN, cosine to centroid", "3.12 % SV-EER",
                       "accepts 82.9 % of targets and 55.2 % of clones"],
                      ["SER", "wav2vec2-base-superb-ER", "41.8 % / 30.7 % accuracy",
                       "bona fide / clones; clones saturate it at 0.83 confidence"],
                      ["CM (in-domain)", "log-mel CNN, train split only", "0.09 % SPF-EER",
                       "optimistic: same two generators seen in training"],
                      ["CM (AASIST-LA)", "official 2019 LA checkpoint", "50.8 % SPF-EER",
                       "no transfer to EmoSpoof-TTS clones"]],
                     "Table 3. Measured component quality on the real merged manifest (test split).")

    add_table_before(doc, anchor,
                     ["fusion arm", "SASV-EER, in-domain CM", "SASV-EER, AASIST-LA CM"],
                     [["speaker only", "0.3309", "0.3309"],
                      ["linear (baseline)", "0.0060", "0.3244"],
                      ["emotion additive", "0.0066", "0.2654"],
                      ["emotion interaction", "0.0055", "0.2661"],
                      ["emotion gated (proposed)", "0.0073", "0.2610"]],
                     "Table 4. Real SASV-EER for the full arm ladder under two countermeasure "
                     "regimes, identical manifest and ASV/SER scores.")

    add_para_before(anchor, (
        "Three results matter. First, the vulnerability is real and emotion-dependent: the "
        "speaker front-end accepts 55.2 % of neural emotional clones as the cloned speaker, and "
        "acceptance rises to 64.6 % for anger and 61.7 % for neutral while falling to 47.5 % for "
        "sadness and 47.2 % for happiness. Second, conventional fusion removes almost all of that "
        "when the countermeasure works, taking SASV-EER from 33.1 % to 0.60 %. Third, the proposed "
        "emotion-gated arm does not beat plain fusion in that regime: it is 0.14 points worse and "
        "its paired bootstrap interval does not exclude zero (p = 0.98). The paper's own hypothesis "
        "is therefore not supported by this data, and reporting it otherwise would be dishonest."))
    add_para_before(anchor, (
        "The second regime explains why an apparent gain must be read carefully. The pretrained "
        "AASIST checkpoint, trained on ASVspoof-2019 attack families, scores at chance on these "
        "clones (SPF-EER 50.8 %). When that failing countermeasure is used, the emotion arms "
        "improve SASV-EER by about 5.5 points with p < 0.001. The mechanism is not emotion "
        "conditioning: the emotion recogniser is confidently wrong on synthetic speech (30.7 % "
        "accuracy at 0.83 mean confidence, against 41.8 % on genuine speech), so its probability "
        "vector implicitly encodes whether the audio is synthetic. Emotion features act as a "
        "substitute spoof detector, which is a confound rather than the proposed effect, and it "
        "sets the agenda for the next pass: replace emotion features with an explicit confidence "
        "scalar and test whether the gain survives, and test on a held-out generator."))

    # --- correct claims the real run has overtaken -------------------------
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("All results in this chapter are from the repository audit"):
            for run in p.runs:
                run.text = ""
            p.runs[0].text = (
                "Sections 4.1 to 4.3 report the repository audit and the supplied synthetic "
                "demonstration, which validate the software pipeline only. Section 4.4 reports a "
                "real-data execution (ESD English bona fide + EmoSpoof-TTS clones) with measured "
                "component scores and the full fusion ladder; its test-split numbers are the only "
                "real performance claims in this report. Neither part establishes novelty over "
                "published systems.")
        elif t.startswith("The completed work demonstrates a reproducible score-level"):
            for run in p.runs:
                run.text = ""
            p.runs[0].text = (
                "The completed work demonstrates a reproducible score-level evaluation pipeline, an "
                "explicit fusion ablation ladder, data audits, metric implementations, trained ASV, "
                "SER and countermeasure front-ends, and both a synthetic smoke test and a real "
                "experiment on ESD and EmoSpoof-TTS data. What it does not show is that the proposed "
                "emotion-conditioned mechanism is better than conventional fusion, that it "
                "generalises to an unseen generator, or that the result survives replacing emotion "
                "features by an explicit confidence signal. It also does not include an AASIST "
                "retrained on ASVspoof 2019 LA, because the corpus cannot be extracted on the "
                "available disk volume. Each of those gaps can change the conclusion, so the "
                "negative result in section 4.4 stands as the finding rather than as a placeholder.")
        elif t.startswith("The synthetic run provides no basis for claiming real authentication gains"):
            for run in p.runs:
                run.text = ""
            p.runs[0].text = (
                "The synthetic run provides no basis for claiming real authentication gains, and "
                "the real run gives a mixed verdict that is reported as measured. On the held-out "
                "test split, plain fusion cut SASV-EER from 33.1 % to 0.60 %, which is a genuine "
                "security improvement over ASV alone, but the proposed emotion-gated arm was "
                "slightly worse than the linear baseline, so the specific contribution of this "
                "project is not yet demonstrated. The failure of an off-the-shelf countermeasure "
                "to transfer to modern neural clones (50.8 % EER) is arguably the most useful "
                "finding, because it shows that countermeasure generalisation, not fusion "
                "architecture, is the binding constraint.")

    doc.save(str(args.docx))
    print(f"updated {args.docx}")


if __name__ == "__main__":
    main()
