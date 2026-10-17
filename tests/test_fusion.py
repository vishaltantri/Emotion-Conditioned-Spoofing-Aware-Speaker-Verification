import numpy as np
import pytest

from easv.fusion import (
    EmotionAdditiveFusion,
    EmotionGatedFusion,
    FusionFeatures,
    LinearFusion,
    SpeakerOnlyFusion,
    Standardizer,
    create_model,
    load_model,
    normalized_entropy,
)
from easv.metrics import compute_eer


def separable_features(n: int = 400, seed: int = 0, n_emotions: int = 3):
    rng = np.random.default_rng(seed)
    labels = rng.random(n) < 0.4
    speaker = np.where(labels, 2.0, -2.0) + rng.normal(0, 0.6, n)
    cm = np.where(labels, 1.5, -1.5) + rng.normal(0, 0.6, n)
    probs = rng.dirichlet(np.ones(n_emotions), size=n)
    return FusionFeatures(speaker, cm, emotion=probs, emotion_labels=tuple(f"e{i}" for i in range(n_emotions))), labels


def test_standardizer_zero_mean_unit_scale_and_constant_column():
    x = np.column_stack([np.arange(10.0), np.full(10, 3.0)])
    scaler = Standardizer().fit(x)
    z = scaler.transform(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(z[:, 0].std(), 1, atol=1e-9)
    assert np.allclose(z[:, 1], 0, atol=1e-9)


def test_standardizer_roundtrip():
    x = np.random.default_rng(0).normal(size=(20, 3))
    scaler = Standardizer().fit(x)
    restored = Standardizer.from_dict(scaler.to_dict())
    assert np.allclose(scaler.transform(x), restored.transform(x))


def test_normalized_entropy_bounds():
    uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
    certain = np.array([[1.0, 0.0, 0.0, 0.0]])
    assert normalized_entropy(uniform)[0] == pytest.approx(1.0)
    assert normalized_entropy(certain)[0] == pytest.approx(0.0, abs=1e-9)


def test_fusion_features_derive_uncertainty_columns():
    probs = np.array([[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]])
    features = FusionFeatures(np.zeros(2), np.zeros(2), emotion=probs)
    assert features.entropy.shape == (2,)
    assert features.confidence.tolist() == pytest.approx([0.7, 1 / 3])


def test_speaker_only_fusion_learns_separable_scores():
    features, labels = separable_features()
    model = SpeakerOnlyFusion(l2=1e-3).fit(features, labels)
    scores = model.decision_function(features)
    assert compute_eer(scores, labels).eer < 0.05
    assert model.weights[0] > 0


def test_linear_fusion_improves_on_speaker_only_when_cm_is_informative():
    rng = np.random.default_rng(2)
    n = 600
    labels = rng.random(n) < 0.4
    speaker = np.where(labels, 1.2, 0.8) + rng.normal(0, 1.0, n)  # weak evidence
    cm = np.where(labels, 1.8, -1.8) + rng.normal(0, 0.7, n)
    features = FusionFeatures(speaker, cm)
    only = SpeakerOnlyFusion().fit(features, labels)
    both = LinearFusion().fit(features, labels)
    assert compute_eer(both.decision_function(features), labels).eer < compute_eer(
        only.decision_function(features), labels
    ).eer


def test_emotion_models_require_emotion_features():
    features = FusionFeatures(np.zeros(5), np.zeros(5))
    for name in ("emotion_additive", "emotion_interaction", "emotion_gated"):
        model = create_model(name, epochs=5, max_iter=5)
        with pytest.raises(ValueError):
            model.fit(features, np.array([True, False, True, False, True]))


def test_gated_fusion_prefers_countermeasure_evidence_where_it_is_reliable():
    """The gate should learn *different* CM weights per emotion condition."""

    rng = np.random.default_rng(3)
    n = 4000
    reliable = rng.random(n) < 0.5
    labels = rng.random(n) < 0.35
    probs = np.where(reliable[:, None], np.array([0.92, 0.04, 0.04]), np.array([0.04, 0.04, 0.92]))
    probs = probs + rng.normal(0, 0.01, probs.shape).clip(min=0)
    probs = np.clip(probs, 1e-6, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    speaker = np.where(labels, 2.0, 1.2) + rng.normal(0, 1.0, n)
    sigma = np.where(reliable, 0.6, 3.0)
    cm = np.where(labels, 1.5, -1.5) + rng.normal(0, 1.0, n) * sigma
    features = FusionFeatures(speaker, cm, emotion=probs, emotion_labels=("a", "b", "c"))

    model = EmotionGatedFusion(hidden=2, l2=1e-3, lr=0.03, epochs=1200, seed=5).fit(features, labels)
    w_s, w_c = model.gate_weights(features)
    assert w_c[reliable].mean() > w_c[~reliable].mean()
    assert (w_c > 0).all()
    assert compute_eer(model.decision_function(features), labels).eer < 0.15


def test_models_are_deterministic_for_a_fixed_seed():
    features, labels = separable_features(seed=7)
    first = EmotionGatedFusion(hidden=3, epochs=50, seed=13).fit(features, labels)
    second = EmotionGatedFusion(hidden=3, epochs=50, seed=13).fit(features, labels)
    assert np.allclose(first.decision_function(features), second.decision_function(features))


def test_state_dict_roundtrip_preserves_scores(tmp_path):
    features, labels = separable_features(seed=11)
    cases = [
        LinearFusion(l2=1e-3).fit(features, labels),
        EmotionAdditiveFusion(l2=1e-3).fit(features, labels),
        EmotionGatedFusion(hidden=2, epochs=40, seed=3).fit(features, labels),
    ]
    for model in cases:
        path = tmp_path / f"{model.name}.json"
        model.save(path)
        restored = load_model(path)
        assert restored.name == model.name
        assert np.allclose(restored.decision_function(features), model.decision_function(features), atol=1e-9)


def test_unknown_model_name_raises():
    with pytest.raises(KeyError):
        create_model("emotion_magic")


def test_create_model_ignores_irrelevant_kwargs():
    model = create_model("linear", epochs=9999, hidden=7, l2=0.01)
    assert isinstance(model, LinearFusion)
