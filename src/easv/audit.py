"""Manifest audits: coverage statistics and leakage checks.

The research plan requires speaker-disjoint, generator-controlled and
emotion-controlled splits, with all exclusions documented. These checks do not
know the intent of a particular experiment, so they *report* overlaps as
findings instead of forbidding them: a cross-generator track wants generator
overlap to be absent, while an in-domain track does not care.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from easv.manifest import DEFAULT_SPLITS, Finding, Manifest, Trial


@dataclass
class AuditReport:
    counts: dict[str, dict[str, dict[str, int]]]
    speakers: dict[str, list[str]]
    attacks: dict[str, list[str]]
    emotions: dict[str, list[str]]
    findings: tuple[Finding, ...] = ()
    allow_speaker_overlap: tuple[str, ...] = ()

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "counts": self.counts,
            "speakers": self.speakers,
            "attacks": self.attacks,
            "emotions": self.emotions,
            "findings": [
                {"level": f.level, "code": f.code, "message": f.message, "trial_id": f.trial_id}
                for f in self.findings
            ],
        }


def _tally(trials: Iterable[Trial], key) -> dict[str, int]:
    return dict(sorted(Counter(key(t) for t in trials).items()))


def audit_manifest(
    manifest: Manifest,
    *,
    splits: Sequence[str] = DEFAULT_SPLITS,
    train_split: str = "train",
    test_splits: Sequence[str] = ("dev", "test"),
    allow_speaker_overlap: Sequence[str] = (),
) -> AuditReport:
    """Build a coverage/leakage report for a manifest.

    ``allow_speaker_overlap`` names trials-disjointness exceptions, e.g.
    ``("train-dev",)`` when the development set deliberately reuses training
    speakers for score-fusion fitting.
    """

    findings: list[Finding] = list(manifest.findings)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    speakers: dict[str, list[str]] = {}
    attacks: dict[str, list[str]] = {}
    emotions: dict[str, list[str]] = {}

    for split in splits:
        trials = manifest.by_split(split)
        counts[split] = {
            "trial_type": _tally(trials, lambda t: t.trial_type),
            "emotion": _tally(trials, lambda t: t.emotion),
            "attack": _tally(trials, lambda t: t.attack),
            "source": _tally(trials, lambda t: t.dataset or "unspecified"),
        }
        speakers[split] = sorted({t.speaker_id for t in trials})
        attacks[split] = sorted({t.attack for t in trials if t.is_spoof})
        emotions[split] = sorted({t.emotion for t in trials})
        if not trials:
            findings.append(Finding("warning", "empty_split", f"split '{split}' has no trials"))

    for split in splits:
        trials = manifest.by_split(split)
        if not trials:
            continue
        types = Counter(t.trial_type for t in trials)
        for required, why in (
            ("target", "EER and t-DCF are undefined without target trials"),
            ("nontarget", "SV-EER and t-DCF are undefined without non-target trials"),
            ("spoof", "SASV-EER and spoof detection are undefined without spoof trials"),
        ):
            if types.get(required, 0) == 0:
                findings.append(
                    Finding("error", "missing_trial_type", f"split '{split}' has no '{required}' trials; {why}")
                )

    for test_split in test_splits:
        if test_split not in splits:
            continue
        pair = f"{train_split}-{test_split}"
        overlap = sorted(set(speakers.get(train_split, [])) & set(speakers.get(test_split, [])))
        if overlap and pair not in allow_speaker_overlap:
            findings.append(
                Finding(
                    "warning",
                    "speaker_overlap",
                    f"{len(overlap)} speaker(s) appear in both '{train_split}' and '{test_split}': "
                    f"{', '.join(overlap[:5])}{' ...' if len(overlap) > 5 else ''}",
                )
            )
        shared_attacks = sorted(set(attacks.get(train_split, [])) & set(attacks.get(test_split, [])))
        if shared_attacks:
            findings.append(
                Finding(
                    "warning",
                    "generator_overlap",
                    f"attack system(s) shared by '{train_split}' and '{test_split}': "
                    f"{', '.join(shared_attacks)} (cross-generator claims need held-out systems)",
                )
            )
        unseen_emotions = sorted(set(emotions.get(test_split, [])) - set(emotions.get(train_split, [])))
        if unseen_emotions:
            findings.append(
                Finding(
                    "warning",
                    "unseen_emotion_in_test",
                    f"'{test_split}' contains emotion(s) absent from '{train_split}': "
                    f"{', '.join(unseen_emotions)} (make sure this is intentional)",
                )
            )

    if train_split not in splits and any(manifest.by_split(s) for s in test_splits):
        findings.append(
            Finding(
                "warning",
                "missing_train_split",
                f"no '{train_split}' trials; fusion models and thresholds must come from another split",
            )
        )

    return AuditReport(
        counts=counts,
        speakers=speakers,
        attacks=attacks,
        emotions=emotions,
        findings=tuple(findings),
        allow_speaker_overlap=tuple(allow_speaker_overlap),
    )


def audit_markdown(report: AuditReport) -> str:
    """Render an audit report as a Markdown table block."""

    lines: list[str] = ["| split | target | nontarget | spoof | emotions | attacks | speakers |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split, block in report.counts.items():
        types = block.get("trial_type", {})
        lines.append(
            "| {split} | {t} | {n} | {s} | {e} | {a} | {sp} |".format(
                split=split,
                t=types.get("target", 0),
                n=types.get("nontarget", 0),
                s=types.get("spoof", 0),
                e=len(report.emotions.get(split, [])),
                a=len(report.attacks.get(split, [])),
                sp=len(report.speakers.get(split, [])),
            )
        )
    if report.findings:
        lines.append("")
        lines.append("Findings:")
        lines.extend(f"- `{f.level}` {f.code}: {f.message}" for f in report.findings)
    return "\n".join(lines) + "\n"
