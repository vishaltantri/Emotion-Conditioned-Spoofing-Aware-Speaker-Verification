"""Score files and the join between scores and manifest trials.

Score-direction conventions, fixed once and used everywhere:

* ``speaker_score`` - higher means the utterance is *more* like the claimed
  speaker.
* ``cm_score``      - higher means *more* bona fide. This follows the ASVspoof
  convention that a countermeasure score supports the bona fide class. The
  spoof evidence used by fusion is therefore ``-cm_score``.
* ``emo_<label>``   - optional predicted probability of an emotion label, one
  column per label the emotion model can output. Columns are normalised to sum
  to one per row when they do not already.

Ground-truth emotion for *analysis* lives in the manifest (``emotion``), never
in the score file. Keeping predicted and ground-truth emotion in separate files
makes leakage of evaluation labels into the fusion inputs visible.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from easv.manifest import Finding, Manifest, Trial

EMOTION_PREFIX = "emo_"
REQUIRED_SCORE_COLUMNS = ("trial_id", "speaker_score", "cm_score")


@dataclass(frozen=True)
class ScoreRow:
    trial_id: str
    speaker_score: float
    cm_score: float
    emotion_probs: tuple[float, ...] | None = None


@dataclass
class ScoreTable:
    rows: dict[str, ScoreRow]
    emotion_labels: tuple[str, ...] = ()
    findings: tuple[Finding, ...] = ()
    source: str | None = None

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == "error")

    def __len__(self) -> int:
        return len(self.rows)


def load_scores(path: str | Path) -> ScoreTable:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip().lower() for h in (reader.fieldnames or [])]
        raw_rows = [row for row in reader if row]

    findings: list[Finding] = []
    for column in REQUIRED_SCORE_COLUMNS:
        if column not in header:
            findings.append(
                Finding("error", "missing_score_column", f"score file is missing column '{column}'")
            )

    emotion_labels = tuple(h[len(EMOTION_PREFIX) :] for h in header if h.startswith(EMOTION_PREFIX))
    if len(set(emotion_labels)) != len(emotion_labels):
        findings.append(Finding("error", "duplicate_emotion_column", "duplicate emo_ columns"))
    if len(emotion_labels) == 0:
        findings.append(
            Finding(
                "warning",
                "no_emotion_scores",
                "no emo_<label> columns found; emotion-conditioned models cannot run",
            )
        )

    rows: dict[str, ScoreRow] = {}
    bad_values = 0
    bad_sums = 0
    for line_no, raw in enumerate(raw_rows, start=2):
        row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
        trial_id = row.get("trial_id", "")
        if not trial_id:
            findings.append(Finding("error", "missing_trial_id", f"score row {line_no} has no trial_id"))
            continue
        if trial_id in rows:
            findings.append(
                Finding("error", "duplicate_score_trial", "duplicate score row dropped", trial_id)
            )
            continue
        try:
            speaker_score = float(row["speaker_score"])
            cm_score = float(row["cm_score"])
        except (KeyError, ValueError):
            bad_values += 1
            findings.append(
                Finding("error", "bad_score_value", "speaker_score/cm_score not numeric", trial_id)
            )
            continue

        probs: tuple[float, ...] | None = None
        if emotion_labels:
            try:
                values = [float(row.get(EMOTION_PREFIX + label, "nan")) for label in emotion_labels]
            except ValueError:
                values = [float("nan")] * len(emotion_labels)
            if any(np.isnan(values)) or any(v < 0 for v in values):
                findings.append(
                    Finding(
                        "error",
                        "bad_emotion_probability",
                        f"emotion probabilities must be finite and non-negative: {values}",
                        trial_id,
                    )
                )
                continue
            total = float(sum(values))
            if total <= 0:
                findings.append(
                    Finding("error", "zero_emotion_probability", "emotion probabilities sum to zero", trial_id)
                )
                continue
            if abs(total - 1.0) > 0.05:
                bad_sums += 1
            values = [v / total for v in values]
            probs = tuple(values)

        rows[trial_id] = ScoreRow(
            trial_id=trial_id,
            speaker_score=speaker_score,
            cm_score=cm_score,
            emotion_probs=probs,
        )

    if bad_sums:
        findings.append(
            Finding(
                "warning",
                "emotion_probs_renormalised",
                f"{bad_sums} row(s) did not sum to 1.0 within 0.05 and were renormalised",
            )
        )
    if bad_values:
        findings.append(
            Finding("warning", "dropped_score_rows", f"{bad_values} row(s) dropped for non-numeric scores")
        )
    if not rows:
        findings.append(Finding("error", "empty_score_table", "score file contains no usable rows"))

    return ScoreTable(
        rows=rows,
        emotion_labels=emotion_labels,
        findings=tuple(findings),
        source=str(path),
    )


@dataclass(frozen=True)
class JoinedItem:
    trial: Trial
    speaker_score: float
    cm_score: float
    emotion_probs: tuple[float, ...] | None


@dataclass(frozen=True)
class TrialBatch:
    """Immutable, numpy-friendly view of a set of joined trials."""

    trial_ids: tuple[str, ...]
    speaker: np.ndarray
    cm: np.ndarray
    emotion_probs: np.ndarray | None
    emotion_labels: tuple[str, ...]
    trial_types: tuple[str, ...]
    emotions: tuple[str, ...]
    attacks: tuple[str, ...]
    speakers: tuple[str, ...]
    claimed_speakers: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.trial_ids)

    @property
    def n(self) -> int:
        return len(self.trial_ids)

    @property
    def is_target(self) -> np.ndarray:
        return np.array([t == "target" for t in self.trial_types], dtype=bool)

    @property
    def is_nontarget(self) -> np.ndarray:
        return np.array([t == "nontarget" for t in self.trial_types], dtype=bool)

    @property
    def is_bona_fide(self) -> np.ndarray:
        return np.array([t in ("target", "nontarget") for t in self.trial_types], dtype=bool)

    @property
    def is_spoof(self) -> np.ndarray:
        return np.array([t == "spoof" for t in self.trial_types], dtype=bool)

    @property
    def has_emotion(self) -> bool:
        return self.emotion_probs is not None and self.emotion_probs.size > 0

    def subset(self, mask: np.ndarray | Sequence[bool]) -> "TrialBatch":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.n,):
            raise ValueError(f"mask shape {mask.shape} does not match {self.n} trials")
        pick = lambda seq: tuple(v for v, keep in zip(seq, mask) if keep)  # noqa: E731
        return TrialBatch(
            trial_ids=pick(self.trial_ids),
            speaker=self.speaker[mask],
            cm=self.cm[mask],
            emotion_probs=None if self.emotion_probs is None else self.emotion_probs[mask],
            emotion_labels=self.emotion_labels,
            trial_types=pick(self.trial_types),
            emotions=pick(self.emotions),
            attacks=pick(self.attacks),
            speakers=pick(self.speakers),
            claimed_speakers=pick(self.claimed_speakers),
        )


@dataclass
class JoinedData:
    items: tuple[JoinedItem, ...]
    emotion_labels: tuple[str, ...]
    findings: tuple[Finding, ...] = ()

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.level == "error")

    @property
    def has_emotion_scores(self) -> bool:
        return len(self.emotion_labels) > 0 and all(i.emotion_probs is not None for i in self.items)

    def batch(self, split: str | None = None) -> TrialBatch:
        items = self.items if split is None else tuple(i for i in self.items if i.trial.split == split)
        return self._build(items)

    def split_batch(self, split: str) -> TrialBatch:
        return self.batch(split)

    def _build(self, items: Iterable[JoinedItem]) -> TrialBatch:
        items = tuple(items)
        probs: np.ndarray | None
        if self.emotion_labels and all(i.emotion_probs is not None for i in items):
            probs = np.asarray([i.emotion_probs for i in items], dtype=float)
        else:
            probs = None
        return TrialBatch(
            trial_ids=tuple(i.trial.trial_id for i in items),
            speaker=np.asarray([i.speaker_score for i in items], dtype=float),
            cm=np.asarray([i.cm_score for i in items], dtype=float),
            emotion_probs=probs,
            emotion_labels=self.emotion_labels,
            trial_types=tuple(i.trial.trial_type for i in items),
            emotions=tuple(i.trial.emotion for i in items),
            attacks=tuple(i.trial.attack for i in items),
            speakers=tuple(i.trial.speaker_id for i in items),
            claimed_speakers=tuple(i.trial.claimed_speaker for i in items),
        )


def join(manifest: Manifest, scores: ScoreTable, require_all: bool = True) -> JoinedData:
    """Inner-join a manifest with a score table on ``trial_id``."""

    findings: list[Finding] = []
    items: list[JoinedItem] = []
    missing: list[str] = []

    for trial in manifest.trials:
        row = scores.rows.get(trial.trial_id)
        if row is None:
            missing.append(trial.trial_id)
            continue
        items.append(
            JoinedItem(
                trial=trial,
                speaker_score=row.speaker_score,
                cm_score=row.cm_score,
                emotion_probs=row.emotion_probs,
            )
        )

    if missing:
        level = "error" if require_all else "warning"
        preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        findings.append(
            Finding(level, "missing_scores", f"{len(missing)} manifest trial(s) have no score row: {preview}")
        )

    extra = set(scores.rows) - {t.trial_id for t in manifest.trials}
    if extra:
        preview = ", ".join(sorted(extra)[:5]) + (" ..." if len(extra) > 5 else "")
        findings.append(
            Finding("warning", "unused_scores", f"{len(extra)} score row(s) not in manifest: {preview}")
        )

    return JoinedData(
        items=tuple(items),
        emotion_labels=scores.emotion_labels,
        findings=tuple(findings),
    )
