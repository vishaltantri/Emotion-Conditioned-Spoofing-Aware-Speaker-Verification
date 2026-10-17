"""Score-fusion models over speaker, countermeasure and emotion evidence.

All models consume :class:`FusionFeatures` and emit a *logit* decision score in
which larger values support acceptance. Sign conventions:

* ``speaker`` is the ASV score (higher = more like the claimed speaker);
* ``cm`` is the countermeasure score (higher = more bona fide), so the spoof
  evidence of the research plan is ``-cm``;
* ``emotion`` is a matrix of predicted emotion probabilities (ground-truth
  emotion never enters the model), with derived normalised entropy and
  confidence when uncertainty features are enabled.

Model ladder (each is an ablation arm required by the research plan):

1. ``speaker_only``        - ASV score only.
2. ``linear``              - conventional speaker + countermeasure fusion,
   the primary baseline.
3. ``emotion_additive``    - the same linear model with emotion features added
   but *no* interaction, isolating "more inputs" from conditioning.
4. ``emotion_interaction`` - linear model with speaker/CM x emotion interaction
   terms, a capacity control for the gated model.
5. ``emotion_gated``       - the proposed emotion-conditioned fusion: a small
   gate maps the emotion representation (including uncertainty) to
   trial-specific non-negative weights on the speaker and countermeasure
   evidence, plus an emotion-conditioned bias.

Everything is trained only on the training split; scaling statistics are fitted
there too, so no evaluation statistic leaks into the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

MODEL_ORDER = (
    "speaker_only",
    "linear",
    "emotion_additive",
    "emotion_interaction",
    "emotion_gated",
)

MODEL_DESCRIPTIONS = {
    "speaker_only": "ASV score only (no countermeasure, no emotion)",
    "linear": "conventional calibrated speaker + countermeasure fusion",
    "emotion_additive": "linear fusion with additive emotion features (no conditioning)",
    "emotion_interaction": "linear fusion with speaker/CM x emotion interactions",
    "emotion_gated": "proposed emotion-conditioned gated fusion with uncertainty",
}


@dataclass
class FusionFeatures:
    """Model inputs for one set of trials."""

    speaker: np.ndarray
    cm: np.ndarray
    emotion: np.ndarray | None = None
    emotion_labels: tuple[str, ...] = ()
    entropy: np.ndarray | None = None
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.speaker = np.asarray(self.speaker, dtype=float).ravel()
        self.cm = np.asarray(self.cm, dtype=float).ravel()
        if self.speaker.shape != self.cm.shape:
            raise ValueError("speaker and cm arrays must align")
        if self.emotion is not None:
            self.emotion = np.asarray(self.emotion, dtype=float)
            if self.emotion.ndim != 2 or self.emotion.shape[0] != self.speaker.shape[0]:
                raise ValueError("emotion matrix must be (n_trials, n_emotions)")
            if self.entropy is None:
                self.entropy = normalized_entropy(self.emotion)
            if self.confidence is None:
                self.confidence = np.max(self.emotion, axis=1)

    def __len__(self) -> int:
        return int(self.speaker.size)

    @property
    def has_emotion(self) -> bool:
        return self.emotion is not None and self.emotion.size > 0

    def mask(self, index: np.ndarray) -> "FusionFeatures":
        index = np.asarray(index)
        return FusionFeatures(
            speaker=self.speaker[index],
            cm=self.cm[index],
            emotion=None if self.emotion is None else self.emotion[index],
            emotion_labels=self.emotion_labels,
            entropy=None if self.entropy is None else self.entropy[index],
            confidence=None if self.confidence is None else self.confidence[index],
        )


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row divided by ``log K`` (0 = certain, 1 = flat)."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("entropy expects a (n, K) matrix")
    k = probabilities.shape[1]
    if k <= 1:
        return np.zeros(probabilities.shape[0], dtype=float)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / np.log(k)


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        x = np.asarray(x, dtype=float)
        self.mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.scale = scale
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("standardizer has not been fitted")
        return (np.asarray(x, dtype=float) - self.mean) / self.scale

    def to_dict(self) -> dict:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "scale": None if self.scale is None else self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, state: Mapping) -> "Standardizer":
        mean = state.get("mean")
        scale = state.get("scale")
        return cls(
            mean=None if mean is None else np.asarray(mean, dtype=float),
            scale=None if scale is None else np.asarray(scale, dtype=float),
        )


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _log_loss(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


class BaseFusionModel:
    """Common interface: fit / decision_function / serialization."""

    name = "base"
    requires_emotion = False

    def raw_features(self, features: FusionFeatures) -> np.ndarray:
        raise NotImplementedError

    def fit(self, features: FusionFeatures, y: Sequence[bool]) -> "BaseFusionModel":
        raise NotImplementedError

    def decision_function(self, features: FusionFeatures) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, features: FusionFeatures) -> np.ndarray:
        return _sigmoid(self.decision_function(features))

    # -- serialization -------------------------------------------------
    def state_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "BaseFusionModel":
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BaseFusionModel":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return load_model_from_state(state)


class LogisticFusion(BaseFusionModel):
    """L2-regularised logistic regression trained with L-BFGS."""

    def __init__(self, *, l2: float = 1e-3, max_iter: int = 2000, tol: float = 1e-9):
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.scaler = Standardizer()
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.n_features: int = 0
        self.n_iter: int = 0
        self.final_objective: float | None = None

    def raw_features(self, features: FusionFeatures) -> np.ndarray:
        raise NotImplementedError

    def fit(self, features: FusionFeatures, y: Sequence[bool]) -> "LogisticFusion":
        x = self.raw_features(features)
        y = np.asarray(y, dtype=float).ravel()
        if np.unique(y).size < 2:
            raise ValueError("fusion training needs both positive and negative trials")
        self.scaler.fit(x)
        xs = self.scaler.transform(x)
        n, d = xs.shape
        self.n_features = d

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            w, b = theta[:-1], theta[-1]
            z = xs @ w + b
            loss = _log_loss(z, y) + self.l2 * float(w @ w)
            grad_w = xs.T @ (_sigmoid(z) - y) / n + 2.0 * self.l2 * w
            grad_b = float(np.mean(_sigmoid(z) - y))
            return loss, np.concatenate([grad_w, [grad_b]])

        theta0 = np.zeros(d + 1, dtype=float)
        result = minimize(
            lambda t: objective(t)[0],
            theta0,
            jac=lambda t: objective(t)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )
        self.weights = result.x[:-1].copy()
        self.bias = float(result.x[-1])
        self.n_iter = int(result.nit)
        self.final_objective = float(result.fun)
        return self

    def decision_function(self, features: FusionFeatures) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError(f"{self.name}: model is not fitted")
        xs = self.scaler.transform(self.raw_features(features))
        return xs @ self.weights + self.bias

    def state_dict(self) -> dict:
        if self.weights is None:
            raise RuntimeError(f"{self.name}: model is not fitted")
        return {
            "name": self.name,
            "kind": "logistic",
            "hyperparameters": {"l2": self.l2, "max_iter": self.max_iter},
            "scaler": self.scaler.to_dict(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "n_features": self.n_features,
            "n_iter": self.n_iter,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "LogisticFusion":
        model = cls(**{k: v for k, v in state.get("hyperparameters", {}).items() if k in ("l2", "max_iter")})
        model.scaler = Standardizer.from_dict(state["scaler"])
        model.weights = np.asarray(state["weights"], dtype=float)
        model.bias = float(state["bias"])
        model.n_features = int(state.get("n_features", model.weights.size))
        model.n_iter = int(state.get("n_iter", 0))
        return model


class SpeakerOnlyFusion(LogisticFusion):
    name = "speaker_only"
    requires_emotion = False

    def raw_features(self, features: FusionFeatures) -> np.ndarray:
        return features.speaker.reshape(-1, 1)


class LinearFusion(LogisticFusion):
    """Conventional speaker + anti-spoofing fusion baseline."""

    name = "linear"
    requires_emotion = False

    def raw_features(self, features: FusionFeatures) -> np.ndarray:
        return np.column_stack([features.speaker, features.cm])


class EmotionAdditiveFusion(LogisticFusion):
    """Linear fusion with emotion evidence added additively (capacity ablation)."""

    name = "emotion_additive"
    requires_emotion = True

    def __init__(self, *, use_uncertainty: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_uncertainty = bool(use_uncertainty)

    def _emotion_block(self, features: FusionFeatures) -> np.ndarray:
        if not features.has_emotion:
            raise ValueError(f"{self.name} requires emotion probability scores (emo_* columns)")
        columns = [features.emotion]
        if self.use_uncertainty:
            columns.append(features.entropy.reshape(-1, 1))
            columns.append(features.confidence.reshape(-1, 1))
        return np.column_stack(columns)

    def raw_features(self, features: FusionFeatures) -> np.ndarray:
        return np.column_stack([features.speaker, features.cm, self._emotion_block(features)])

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["hyperparameters"]["use_uncertainty"] = self.use_uncertainty
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "EmotionAdditiveFusion":
        model = cls(
            **{
                k: v
                for k, v in state.get("hyperparameters", {}).items()
                if k in ("l2", "max_iter", "use_uncertainty")
            }
        )
        model.scaler = Standardizer.from_dict(state["scaler"])
        model.weights = np.asarray(state["weights"], dtype=float)
        model.bias = float(state["bias"])
        model.n_features = int(state.get("n_features", model.weights.size))
        return model


class EmotionInteractionFusion(EmotionAdditiveFusion):
    """Linear fusion with speaker/CM x emotion interaction features.

    Serves as the capacity-control arm: it tests whether any gain of the gated
    model comes merely from more parameters rather than from conditioning.
    """

    name = "emotion_interaction"
    requires_emotion = True

    def raw_features(self, features: FusionFeatures) -> np.ndarray:
        block = self._emotion_block(features)
        s = features.speaker.reshape(-1, 1)
        c = features.cm.reshape(-1, 1)
        interactions = [s * block, c * block, s * c]
        return np.column_stack([features.speaker, features.cm, block] + interactions)


class EmotionGatedFusion(BaseFusionModel):
    """Proposed emotion-conditioned gated fusion.

    ``score = w_s(phi) * s + w_c(phi) * c + b(phi)`` where ``phi`` is the emotion
    representation (probabilities plus uncertainty), ``w_s``/``w_c`` are
    non-negative weights produced by a small tanh gate, and ``b`` is an
    emotion-conditioned bias. The weights are trained end-to-end with the
    logistic loss, so the model can down-weight countermeasure evidence in
    emotion conditions where that evidence is unreliable instead of applying a
    single global weight. Training uses Adam on the full training split with L2
    penalties on the gate parameters.
    """

    name = "emotion_gated"
    requires_emotion = True

    def __init__(
        self,
        *,
        hidden: int = 8,
        l2: float = 1e-3,
        lr: float = 0.05,
        epochs: int = 600,
        seed: int = 0,
        use_uncertainty: bool = True,
        early_stopping: bool = True,
        validation_fraction: float = 0.15,
        patience: int = 40,
        min_delta: float = 1e-6,
    ):
        self.hidden = int(hidden)
        self.l2 = float(l2)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.seed = int(seed)
        self.use_uncertainty = bool(use_uncertainty)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.base_scaler = Standardizer()
        self.gate_scaler = Standardizer()
        self.params: dict[str, np.ndarray | float] = {}
        self.history: list[float] = []
        self.val_history: list[float] = []
        self.best_epoch: int | None = None

    # -- features ------------------------------------------------------
    def _base_matrix(self, features: FusionFeatures) -> np.ndarray:
        return np.column_stack([features.speaker, features.cm])

    def _gate_matrix(self, features: FusionFeatures) -> np.ndarray:
        if not features.has_emotion:
            raise ValueError("emotion_gated requires emotion probability scores (emo_* columns)")
        columns = [features.emotion]
        if self.use_uncertainty:
            columns.append(features.entropy.reshape(-1, 1))
            columns.append(features.confidence.reshape(-1, 1))
        return np.column_stack(columns)

    def raw_features(self, features: FusionFeatures) -> np.ndarray:
        return np.column_stack([self._base_matrix(features), self._gate_matrix(features)])

    # -- forward/backward ---------------------------------------------
    def _forward(self, base: np.ndarray, gate: np.ndarray):
        p = self.params
        pre = gate @ p["A"] + p["a"]
        h = np.tanh(pre)
        h = h if self.hidden > 0 else gate
        q_s = h @ p["u"] + p["c_s"]
        q_c = h @ p["v"] + p["c_c"]
        w_s = _softplus(q_s)
        w_c = _softplus(q_c)
        bias = h @ p["vb"] + p["d"]
        z = w_s * base[:, 0] + w_c * base[:, 1] + bias
        cache = {"h": h, "q_s": q_s, "q_c": q_c, "w_s": w_s, "w_c": w_c}
        return z, cache

    def _init_params(self, p_gate: int) -> None:
        rng = np.random.default_rng(self.seed)
        hidden = self.hidden
        scale = 0.1
        self.params = {
            "A": rng.normal(0.0, scale, size=(p_gate, hidden)),
            "a": np.zeros(hidden),
            "u": rng.normal(0.0, scale, size=(hidden,)),
            "v": rng.normal(0.0, scale, size=(hidden,)),
            "vb": rng.normal(0.0, scale, size=(hidden,)),
            "c_s": np.array(np.log(np.e - 1.0)),  # softplus(0.5413) == 1
            "c_c": np.array(np.log(np.e - 1.0)),
            "d": np.array(0.0),
        }

    def fit(self, features: FusionFeatures, y: Sequence[bool]) -> "EmotionGatedFusion":
        y = np.asarray(y, dtype=float).ravel()
        if np.unique(y).size < 2:
            raise ValueError("fusion training needs both positive and negative trials")

        base = self._base_matrix(features)
        gate_raw = self._gate_matrix(features)
        self.base_scaler.fit(base)
        self.gate_scaler.fit(gate_raw)
        base = self.base_scaler.transform(base)
        gate = self.gate_scaler.transform(gate_raw)
        self._init_params(gate.shape[1])

        n = y.size
        index = np.arange(n)
        val_index = np.array([], dtype=int)
        train_index = index
        if self.early_stopping and n >= 20 and 0.0 < self.validation_fraction < 0.5:
            rng = np.random.default_rng(self.seed + 1)
            shuffled = rng.permutation(n)
            n_val = max(5, int(round(n * self.validation_fraction)))
            val_index = np.sort(shuffled[:n_val])
            train_index = np.sort(shuffled[n_val:])

        m = {k: np.zeros_like(v, dtype=float) for k, v in self.params.items()}
        v2 = {k: np.zeros_like(v, dtype=float) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        best_val = np.inf
        best_params: dict[str, np.ndarray] | None = None
        best_epoch = 0
        self.history = []
        self.val_history = []

        for epoch in range(1, self.epochs + 1):
            grads = self._gradient(base[train_index], gate[train_index], y[train_index])
            for k in self.params:
                g = grads[k]
                m[k] = beta1 * m[k] + (1 - beta1) * g
                v2[k] = beta2 * v2[k] + (1 - beta2) * g * g
                m_hat = m[k] / (1 - beta1**epoch)
                v_hat = v2[k] / (1 - beta2**epoch)
                self.params[k] = self.params[k] - self.lr * m_hat / (np.sqrt(v_hat) + eps)

            self.history.append(self._objective(base[train_index], gate[train_index], y[train_index]))

            if val_index.size:
                with np.errstate(over="ignore"):
                    z_val, _ = self._forward(base[val_index], gate[val_index])
                val_loss = self._log_loss_(z_val, y[val_index])
                self.val_history.append(val_loss)
                if val_loss < best_val - self.min_delta:
                    best_val = val_loss
                    best_epoch = epoch
                    best_params = {k: np.array(v, dtype=float, copy=True) for k, v in self.params.items()}
                elif epoch - best_epoch >= self.patience:
                    break

        if best_params is not None:
            self.params = best_params
            self.best_epoch = best_epoch
        return self

    def _log_loss_(self, z: np.ndarray, y: np.ndarray) -> float:
        return _log_loss(z, y)

    def _objective(self, base: np.ndarray, gate: np.ndarray, y: np.ndarray) -> float:
        z, _ = self._forward(base, gate)
        penalty = sum(float(np.sum(np.asarray(self.params[k]) ** 2)) for k in ("A", "u", "v", "vb"))
        return _log_loss(z, y) + self.l2 * penalty

    def _gradient(self, base: np.ndarray, gate: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
        z, cache = self._forward(base, gate)
        n = y.size
        p = self.params
        dz = (_sigmoid(z) - y) / n
        s, c = base[:, 0], base[:, 1]

        dw_s = dz * s
        dw_c = dz * c
        dq_s = dw_s * _sigmoid(cache["q_s"])
        dq_c = dw_c * _sigmoid(cache["q_c"])
        dbias = dz

        h = cache["h"]
        dh = np.outer(dq_s, p["u"]) + np.outer(dq_c, p["v"]) + np.outer(dbias, p["vb"])
        if self.hidden > 0:
            dpre = dh * (1.0 - h**2)
            dA = gate.T @ dpre
            da = dpre.sum(axis=0)
        else:
            dA = np.zeros_like(p["A"])
            da = gate.T @ dh

        grads = {
            "A": dA + 2.0 * self.l2 * p["A"],
            "a": da,
            "u": h.T @ dq_s + 2.0 * self.l2 * p["u"],
            "c_s": np.array(dq_s.sum()),
            "v": h.T @ dq_c + 2.0 * self.l2 * p["v"],
            "c_c": np.array(dq_c.sum()),
            "vb": h.T @ dbias + 2.0 * self.l2 * p["vb"],
            "d": np.array(dbias.sum()),
        }
        return grads

    def decision_function(self, features: FusionFeatures) -> np.ndarray:
        if not self.params:
            raise RuntimeError("emotion_gated: model is not fitted")
        base = self.base_scaler.transform(self._base_matrix(features))
        gate = self.gate_scaler.transform(self._gate_matrix(features))
        z, _ = self._forward(base, gate)
        return z

    def gate_weights(self, features: FusionFeatures) -> tuple[np.ndarray, np.ndarray]:
        """Return per-trial ``(speaker_weight, cm_weight)`` for inspection."""

        if not self.params:
            raise RuntimeError("emotion_gated: model is not fitted")
        base = self.base_scaler.transform(self._base_matrix(features))
        gate = self.gate_scaler.transform(self._gate_matrix(features))
        _, cache = self._forward(base, gate)
        return cache["w_s"], cache["w_c"]

    def state_dict(self) -> dict:
        if not self.params:
            raise RuntimeError("emotion_gated: model is not fitted")
        return {
            "name": self.name,
            "kind": "emotion_gated",
            "hyperparameters": {
                "hidden": self.hidden,
                "l2": self.l2,
                "lr": self.lr,
                "epochs": self.epochs,
                "seed": self.seed,
                "use_uncertainty": self.use_uncertainty,
            },
            "base_scaler": self.base_scaler.to_dict(),
            "gate_scaler": self.gate_scaler.to_dict(),
            "params": {k: np.asarray(v).tolist() for k, v in self.params.items()},
            "best_epoch": self.best_epoch,
            "final_train_loss": self.history[-1] if self.history else None,
            "best_val_loss": min(self.val_history) if self.val_history else None,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping) -> "EmotionGatedFusion":
        model = cls(**{k: v for k, v in state.get("hyperparameters", {}).items()})
        model.base_scaler = Standardizer.from_dict(state["base_scaler"])
        model.gate_scaler = Standardizer.from_dict(state["gate_scaler"])
        model.params = {k: np.asarray(v, dtype=float) for k, v in state["params"].items()}
        model.best_epoch = state.get("best_epoch")
        return model


_MODEL_TYPES = (SpeakerOnlyFusion, LinearFusion, EmotionAdditiveFusion, EmotionInteractionFusion, EmotionGatedFusion)
_REGISTRY = {m.name: m for m in _MODEL_TYPES}


def create_model(name: str, **kwargs) -> BaseFusionModel:
    """Instantiate a fusion model by name, passing only relevant kwargs."""

    if name not in _REGISTRY:
        raise KeyError(f"unknown fusion model '{name}'; available: {', '.join(_REGISTRY)}")
    cls = _REGISTRY[name]
    if issubclass(cls, EmotionGatedFusion):
        accepted = {
            "hidden",
            "l2",
            "lr",
            "epochs",
            "seed",
            "use_uncertainty",
            "early_stopping",
            "validation_fraction",
            "patience",
        }
    elif issubclass(cls, EmotionAdditiveFusion):
        accepted = {"l2", "max_iter", "use_uncertainty"}
    else:
        accepted = {"l2", "max_iter"}
    return cls(**{k: v for k, v in kwargs.items() if k in accepted})


def load_model_from_state(state: Mapping) -> BaseFusionModel:
    name = state.get("name")
    if name not in _REGISTRY:
        raise KeyError(f"unknown fusion model '{name}' in state")
    return _REGISTRY[name].from_state_dict(state)


def load_model(path: str | Path) -> BaseFusionModel:
    return load_model_from_state(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ModelReport:
    """Training summary for one fusion model."""

    name: str
    description: str
    n_features: int
    n_train: int
    training: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "n_features": self.n_features,
            "n_train": self.n_train,
            "training": self.training,
        }
