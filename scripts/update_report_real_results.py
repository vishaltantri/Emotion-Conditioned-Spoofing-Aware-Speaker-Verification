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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docx", type=Path,
                    default=Path("Course_Project_Report_EASV_Audited.docx"))
    args = ap.parse_args()

    doc = docx.Document(str(args.docx))
    texts = [p.text.strip() for p in doc.paragraphs]
    if any(t.startswith("4.4 Real-data results") for t in texts):
        print("section 4.5 already present; nothing to do")
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
