"""Synthetic demonstration data.

This module exists so the full pipeline can be executed, tested and reviewed
without downloading restricted speech corpora. It simulates:

* disjoint speaker sets for train/dev/test;
* recognised attack systems in train/dev and **unseen** generators in test;
* an emotion taxonomy whose reliability differs by emotion, with a held-out
  emotion (``surprise``) that the simulated emotion classifier never saw;
* countermeasure scores whose reliability degrades exactly when the emotion
  classifier is uncertain, and speaker scores that degrade for emotional speech
  (the real-world effects the research plan worries about).

The simulated mechanism is *assumed*, not measured. Any improvement of the
gated model on this data validates the pipeline, not the research hypothesis.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TRAIN_EMOTIONS = ("neutral", "happy", "sad", "angry", "fear")
HELD_OUT_EMOTION = "surprise"
ALL_EMOTIONS = TRAIN_EMOTIONS + (HELD_OUT_EMOTION,)

# Simulated reliability of countermeasure evidence per emotion (1 = highly reliable).
# The contrast is deliberately wide: some emotion conditions make countermeasure
# evidence almost useless, which is the regime the proposed gating targets.
EMOTION_CM_RELIABILITY = {
    "neutral": 0.97,
    "angry": 0.90,
    "happy": 0.55,
    "sad": 0.40,
    "fear": 0.30,
    "surprise": 0.22,
}

CM_SIGMA_FLOOR = 0.35
CM_SIGMA_SPAN = 4.65  # sigma = floor + span * (1 - reliability)
UNCERTAINTY_SIGMA_GAIN = 1.6
SPOOF_EVASION_PER_UNCERTAINTY = 0.8  # spoof scores drift toward bona fide when uncertain

TRAIN_GENERATORS = ("g1_tts", "g2_vc", "g3_tts")
DEV_GENERATORS = ("g4_vc",)
TEST_GENERATORS = ("g5_tts", "g6_vc")

MANIFEST_HEADER = [
    "trial_id",
    "split",
    "speaker_id",
    "claimed_speaker",
    "trial_type",
    "emotion",
    "attack",
    "dataset",
    "path",
    "duration_sec",
    "notes",
]


@dataclass
class DemoData:
    manifest_path: Path
    scores_path: Path
    n_trials: int
    speakers: dict[str, list[str]] = field(default_factory=dict)
    generators: dict[str, list[str]] = field(default_factory=dict)


def _emotion_cm_sigma(emotion: str) -> float:
    reliability = EMOTION_CM_RELIABILITY.get(emotion, 0.5)
    return CM_SIGMA_FLOOR + CM_SIGMA_SPAN * (1.0 - reliability)


def _emotion_prediction(
    rng: np.random.Generator,
    true_emotion: str,
    uncertainty: float,
    known_emotions: tuple[str, ...],
) -> np.ndarray:
    """Simulate an emotion classifier trained only on ``known_emotions``."""

    k = len(known_emotions)
    focused = true_emotion in known_emotions and rng.random() < 0.82 - 0.35 * uncertainty
    if focused:
        chosen = known_emotions.index(true_emotion)
    else:
        # Unseen or ambiguous emotion: the classifier is confused but not uniform.
        chosen = int(rng.integers(0, k))
    concentration = float(np.clip(0.95 - 0.6 * uncertainty - 0.25 * (not focused), 0.12, 0.95))
    probs = rng.dirichlet(np.full(k, 0.35))
    probs = (1.0 - concentration) * probs
    probs[chosen] += concentration
    return probs / probs.sum()


def _trial_score(
    rng: np.random.Generator,
    *,
    is_target: bool,
    is_bona_fide: bool,
    emotion: str,
    generator: str | None,
    generator_seen: bool,
    uncertainty: float,
) -> tuple[float, float]:
    """Simulate ``(speaker_score, cm_score)`` for one trial."""

    sigma_cm = _emotion_cm_sigma(emotion) * (1.0 + UNCERTAINTY_SIGMA_GAIN * uncertainty)
    sigma_spk = 1.1 + (0.35 if emotion != "neutral" else 0.0)

    # ASV score. Spoofed trials *imitate the claimed speaker*, so the ASV branch
    # alone cannot separate them: this is what makes the countermeasure branch,
    # and its emotion-dependent reliability, matter.
    if is_target:
        speaker_mu = 2.0 - (0.35 if emotion != "neutral" else 0.0)
    elif is_bona_fide:
        speaker_mu = -2.0
    else:
        speaker_mu = 1.0
    speaker_score = float(rng.normal(speaker_mu, sigma_spk))

    if is_bona_fide:
        cm_mu = 1.2
    else:
        # Unseen generators are harder to detect; evasive spoofs drift toward
        # bona fide as the condition becomes more uncertain.
        cm_mu = (-0.7 if not generator_seen else -1.2) - SPOOF_EVASION_PER_UNCERTAINTY * uncertainty
    cm_score = float(rng.normal(cm_mu, sigma_cm))
    return speaker_score, cm_score


def generate_demo_data(
    out_dir: str | Path,
    *,
    seed: int = 7,
    n_train_speakers: int = 40,
    n_dev_speakers: int = 12,
    n_test_speakers: int = 20,
) -> DemoData:
    """Write ``manifest.csv`` and ``scores.csv`` for the synthetic scenario."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    speakers = {
        "train": [f"spk{i:03d}" for i in range(1, n_train_speakers + 1)],
        "dev": [f"spk{i:03d}" for i in range(100, 100 + n_dev_speakers)],
        "test": [f"spk{i:03d}" for i in range(200, 200 + n_test_speakers)],
    }
    generators = {"train": list(TRAIN_GENERATORS), "dev": list(DEV_GENERATORS), "test": list(TEST_GENERATORS)}
    train_generators = set(TRAIN_GENERATORS)

    manifest_rows: list[list[str]] = []
    score_rows: list[list[str]] = []
    score_header = ["trial_id", "speaker_score", "cm_score"] + [
        f"emo_{emotion}" for emotion in TRAIN_EMOTIONS
    ]

    for split in ("train", "dev", "test"):
        emotions = ALL_EMOTIONS if split == "test" else TRAIN_EMOTIONS
        split_speakers = speakers[split]
        for speaker_index, claimed in enumerate(split_speakers):
            # A stable partner speaker supplies genuine "other speaker" audio.
            partner = split_speakers[(speaker_index + 1) % len(split_speakers)]
            for emotion in emotions:
                uncertainty = float(np.clip(rng.beta(2.0, 3.0), 0.0, 0.95))

                # 1) target trial: the claimed speaker, bona fide.
                _emit(
                    manifest_rows,
                    score_rows,
                    rng,
                    trial_id=f"{split}_{claimed}_{emotion}_tar",
                    split=split,
                    speaker_id=claimed,
                    claimed_speaker=claimed,
                    trial_type="target",
                    emotion=emotion,
                    attack="none",
                    generator=None,
                    generator_seen=True,
                    uncertainty=uncertainty,
                    is_target=True,
                    is_bona_fide=True,
                )

                # 2) non-target trial: a different speaker presented as the claimed one.
                _emit(
                    manifest_rows,
                    score_rows,
                    rng,
                    trial_id=f"{split}_{claimed}_{emotion}_non",
                    split=split,
                    speaker_id=partner,
                    claimed_speaker=claimed,
                    trial_type="nontarget",
                    emotion=emotion,
                    attack="none",
                    generator=None,
                    generator_seen=True,
                    uncertainty=uncertainty,
                    is_target=False,
                    is_bona_fide=True,
                )

                # 3) spoof trials: one per generator configured for this split.
                for generator in generators[split]:
                    _emit(
                        manifest_rows,
                        score_rows,
                        rng,
                        trial_id=f"{split}_{claimed}_{emotion}_{generator}",
                        split=split,
                        speaker_id=partner,
                        claimed_speaker=claimed,
                        trial_type="spoof",
                        emotion=emotion,
                        attack=generator,
                        generator=generator,
                        generator_seen=generator in train_generators,
                        uncertainty=uncertainty,
                        is_target=False,
                        is_bona_fide=False,
                    )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_HEADER)
        writer.writerows(manifest_rows)

    scores_path = out_dir / "scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(score_header)
        writer.writerows(score_rows)

    return DemoData(
        manifest_path=manifest_path,
        scores_path=scores_path,
        n_trials=len(manifest_rows),
        speakers=speakers,
        generators=generators,
    )


def _emit(
    manifest_rows: list[list[str]],
    score_rows: list[list[str]],
    rng: np.random.Generator,
    *,
    trial_id: str,
    split: str,
    speaker_id: str,
    claimed_speaker: str,
    trial_type: str,
    emotion: str,
    attack: str,
    generator: str | None,
    generator_seen: bool,
    uncertainty: float,
    is_target: bool,
    is_bona_fide: bool,
) -> None:
    speaker_score, cm_score = _trial_score(
        rng,
        is_target=is_target,
        is_bona_fide=is_bona_fide,
        emotion=emotion,
        generator=generator,
        generator_seen=generator_seen,
        uncertainty=uncertainty,
    )
    probs = _emotion_prediction(rng, emotion, uncertainty, TRAIN_EMOTIONS)
    manifest_rows.append(
        [
            trial_id,
            split,
            speaker_id,
            claimed_speaker,
            trial_type,
            emotion,
            attack,
            "SYNTHETIC",
            f"synthetic/{trial_id}.wav",
            "3.0",
            "simulated scores only; no audio file exists",
        ]
    )
    score_rows.append([trial_id, f"{speaker_score:.6f}", f"{cm_score:.6f}"] + [f"{p:.6f}" for p in probs])
