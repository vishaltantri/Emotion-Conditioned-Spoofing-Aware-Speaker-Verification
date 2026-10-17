"""Trial manifests: schema, loading and validation.

A *manifest* is the single source of truth for which trials exist, what the
ground truth is, and which split each trial belongs to. Score files are joined
onto it by ``trial_id`` (see :mod:`easv.scores`). Nothing in this module reads
audio; audio paths are optional metadata.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

TARGET = "target"
NON_TARGET = "nontarget"
SPOOF = "spoof"
TRIAL_TYPES = (TARGET, NON_TARGET, SPOOF)

DEFAULT_SPLITS = ("train", "dev", "test")

REQUIRED_COLUMNS = (
    "trial_id",
    "split",
    "speaker_id",
    "claimed_speaker",
    "trial_type",
    "emotion",
    "attack",
)

OPTIONAL_COLUMNS = (
    "path",
    "duration_sec",
    "dataset",
    "source_trial_id",
    "notes",
)

UNKNOWN = "unknown"


@dataclass(frozen=True)
class Trial:
    """One verification trial.

    ``speaker_id`` is the speaker that actually produced the audio (the *source*
    speaker for spoofed speech, where known). ``claimed_speaker`` is the enrolled
    identity the audio is presented as. ``target`` trials therefore have
    ``speaker_id == claimed_speaker``, and ``nontarget`` trials must differ.
    """

    trial_id: str
    split: str
    speaker_id: str
    claimed_speaker: str
    trial_type: str
    emotion: str
    attack: str
    path: str | None = None
    duration_sec: float | None = None
    dataset: str | None = None
    source_trial_id: str | None = None
    notes: str | None = None

    @property
    def is_target(self) -> bool:
        return self.trial_type == TARGET

    @property
    def is_bona_fide(self) -> bool:
        return self.trial_type in (TARGET, NON_TARGET)

    @property
    def is_spoof(self) -> bool:
        return self.trial_type == SPOOF


@dataclass(frozen=True)
class Finding:
    """A validation or audit finding."""

    level: str  # "error" | "warning"
    code: str
    message: str
    trial_id: str | None = None

    def format(self) -> str:
        where = f" [{self.trial_id}]" if self.trial_id else ""
        return f"{self.level.upper()}:{where} {self.code}: {self.message}"


@dataclass
class Manifest:
    trials: tuple[Trial, ...]
    findings: tuple[Finding, ...] = ()
    source: str | None = None

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == "warning")

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted({t.split for t in self.trials}))

    def by_split(self, split: str) -> tuple[Trial, ...]:
        return tuple(t for t in self.trials if t.split == split)

    def __len__(self) -> int:
        return len(self.trials)


def _first(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name]:
            return row[name]
    return ""


def _normalize_row(row: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        out[key.strip().lower()] = (value or "").strip()
    return out


def load_manifest(
    path: str | Path,
    splits: Sequence[str] = DEFAULT_SPLITS,
) -> Manifest:
    """Load and validate a manifest CSV.

    Validation findings are returned rather than raised so that the audit CLI
    can report every problem at once. Callers that require a clean manifest
    should check :attr:`Manifest.errors`.
    """

    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_header = reader.fieldnames or []
        header = [h.strip().lower() for h in raw_header]
        rows = [_normalize_row(row) for row in reader if row]

    findings: list[Finding] = []
    for column in REQUIRED_COLUMNS:
        if column not in header:
            findings.append(
                Finding("error", "missing_column", f"manifest is missing required column '{column}'")
            )

    trials: list[Trial] = []
    seen: dict[str, int] = {}
    duration_bad = 0

    for line_no, row in enumerate(rows, start=2):
        trial_id = _first(row, "trial_id")
        if not trial_id:
            findings.append(
                Finding("error", "missing_trial_id", f"row {line_no} has no trial_id")
            )
            continue
        if trial_id in seen:
            findings.append(
                Finding(
                    "error",
                    "duplicate_trial_id",
                    f"trial_id '{trial_id}' first seen on row {seen[trial_id]}",
                    trial_id,
                )
            )
            continue
        seen[trial_id] = line_no

        split = _first(row, "split").lower()
        if not split:
            findings.append(Finding("error", "missing_split", "split is empty", trial_id))
            split = UNKNOWN
        elif split not in splits:
            findings.append(
                Finding(
                    "warning",
                    "unexpected_split",
                    f"split '{split}' is not one of {tuple(splits)}",
                    trial_id,
                )
            )

        trial_type = _first(row, "trial_type").lower()
        if trial_type not in TRIAL_TYPES:
            findings.append(
                Finding(
                    "error",
                    "bad_trial_type",
                    f"trial_type '{trial_type}' not in {TRIAL_TYPES}",
                    trial_id,
                )
            )
            continue

        speaker_id = _first(row, "speaker_id") or UNKNOWN
        claimed_speaker = _first(row, "claimed_speaker") or UNKNOWN
        if not speaker_id or not claimed_speaker:
            findings.append(
                Finding("error", "missing_speaker", "speaker_id/claimed_speaker empty", trial_id)
            )

        emotion = _first(row, "emotion").lower() or UNKNOWN
        attack = _first(row, "attack").lower() or "none"

        if trial_type == TARGET and speaker_id != claimed_speaker:
            findings.append(
                Finding(
                    "error",
                    "target_speaker_mismatch",
                    f"target trial has speaker_id='{speaker_id}' != claimed_speaker='{claimed_speaker}'",
                    trial_id,
                )
            )
        if trial_type == NON_TARGET and speaker_id == claimed_speaker:
            findings.append(
                Finding(
                    "error",
                    "nontarget_speaker_overlap",
                    "nontarget trial reuses the claimed speaker as source speaker",
                    trial_id,
                )
            )
        if trial_type == SPOOF and attack in ("", "none"):
            findings.append(
                Finding("error", "spoof_without_attack", "spoof trial must name an attack", trial_id)
            )
        if trial_type in (TARGET, NON_TARGET) and attack not in ("", "none"):
            findings.append(
                Finding(
                    "error",
                    "bonafide_with_attack",
                    f"bona fide trial declares attack '{attack}'",
                    trial_id,
                )
            )

        duration: float | None = None
        raw_duration = _first(row, "duration_sec")
        if raw_duration:
            try:
                duration = float(raw_duration)
                if duration <= 0:
                    raise ValueError
            except ValueError:
                duration_bad += 1
                findings.append(
                    Finding("warning", "bad_duration", f"could not parse duration '{raw_duration}'", trial_id)
                )
                duration = None

        trials.append(
            Trial(
                trial_id=trial_id,
                split=split,
                speaker_id=speaker_id,
                claimed_speaker=claimed_speaker,
                trial_type=trial_type,
                emotion=emotion,
                attack=attack,
                path=_first(row, "path") or None,
                duration_sec=duration,
                dataset=_first(row, "dataset") or None,
                source_trial_id=_first(row, "source_trial_id") or None,
                notes=_first(row, "notes") or None,
            )
        )

    if not trials:
        findings.append(Finding("error", "empty_manifest", "manifest contains no usable trials"))

    return Manifest(trials=tuple(trials), findings=tuple(findings), source=str(path))


def manifest_template() -> str:
    """Return an empty manifest CSV with the canonical header and one demo row."""

    header = list(REQUIRED_COLUMNS) + ["path", "duration_sec", "dataset"]
    example = [
        "demo_target_0001",
        "train",
        "spk001",
        "spk001",
        "target",
        "neutral",
        "none",
        "/path/to/audio.wav",
        "4.2",
        "EXAMPLE",
    ]
    return ",".join(header) + "\n" + ",".join(example) + "\n"


def merge_manifests(manifests: Iterable[Manifest]) -> Manifest:
    """Concatenate manifests, keeping the first occurrence of each trial_id."""

    trials: list[Trial] = []
    findings: list[Finding] = []
    seen: set[str] = set()
    for manifest in manifests:
        findings.extend(manifest.findings)
        for trial in manifest.trials:
            if trial.trial_id in seen:
                findings.append(
                    Finding("warning", "duplicate_trial_id", "duplicate trial_id dropped", trial.trial_id)
                )
                continue
            seen.add(trial.trial_id)
            trials.append(trial)
    return Manifest(trials=tuple(trials), findings=tuple(findings))
