"""Generate the manuscript figure set from stored evaluation output.

Every plotted number is read from a file that an evaluation run wrote. Nothing in
this script contains a hand-typed result, so the figures cannot drift away from
`results/*/results.json` — re-running the evaluation regenerates them.

Figures:

1. `fig1_arms_by_regime`   SASV-EER with grouped-bootstrap intervals for the five
                           fusion arms, one panel per countermeasure regime. Shows
                           the sign flip between regimes.
2. `fig2_gate_weights`     the paper's negative finding: per-emotion speaker and
                           countermeasure gate weights are flat in every regime.
3. `fig3_vulnerability`    ASV spoof acceptance by generator, and the emotion x
                           generator matrix (StyleTTS2 separated, not averaged away).
4. `fig4_det_curves`       DET curves for `linear` vs `emotion_gated` per regime.
5. `fig5_asv_scores`       per-generator ASV score distributions against the
                           dev-selected threshold.

Figures 3 (heatmap) and 5 need the raw ASV score file, which lives with the
corpora rather than in the repository. Without it they fall back to the stored
marginals (fig3a) or are skipped with a notice, so the script still runs on a
clean checkout.

Usage:
    .venv/bin/python scripts/make_figures.py \
        [--manifest results/e2e/merged_manifest.csv] \
        [--results-root results/real] [--out figures] \
        [--asv-scores $DATA_ROOT/real/scores_asv_merged.csv] \
        [--asv-summary results/real/asv_summary.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

MODELS = ("speaker_only", "linear", "emotion_additive", "emotion_interaction", "emotion_gated")
MODEL_LABELS = {
    "speaker_only": "speaker only",
    "linear": "linear (baseline)",
    "emotion_additive": "additive",
    "emotion_interaction": "interaction",
    "emotion_gated": "gated (proposed)",
}
MODEL_COLORS = {
    "speaker_only": "#999999",
    "linear": "#1f4e79",
    "emotion_additive": "#2e75b6",
    "emotion_interaction": "#548235",
    "emotion_gated": "#c00000",
}

REGIMES = (
    ("run1", "A · in-domain CM\n(two generators)"),
    ("run2_oodcm", "B · ASVspoof-2019 AASIST\n(failing CM)"),
    ("run3_crossgen", "C · F5-TTS held out\n(two generators)"),
    ("run4_3gen_crossgen", "D · StyleTTS2 held out\n(three generators)"),
)

GENERATOR_LABELS = {
    "cosyvoice": "CosyVoice",
    "f5_tts": "F5-TTS",
    "styletts2": "StyleTTS2",
}
EMOTION_ORDER = ("angry", "happy", "neutral", "sad")


def save(fig, out_dir: Path, stem: str, plt) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def load_runs(root: Path) -> list[dict]:
    runs = []
    for key, label in REGIMES:
        path = root / key / "results.json"
        if not path.exists():
            raise SystemExit(
                f"missing {path}: run the evaluations first (see docs/REAL_RESULTS.md, "
                "section Reproduce)")
        runs.append({"key": key, "label": label, "dir": root / key,
                     "payload": json.loads(path.read_text(encoding="utf-8"))})
    return runs


def arm(run: dict, name: str) -> dict | None:
    for entry in run["payload"]["models"]:
        if entry["name"] == name:
            return entry
    return None


def fig_arms_by_regime(runs, out_dir, plt, np):
    fig, axes = plt.subplots(1, len(runs), figsize=(13.2, 3.9), sharey=True,
                             layout="constrained")
    x = np.arange(len(MODELS))
    for ax, run in zip(axes, runs):
        values, lo, hi, colors = [], [], [], []
        for name in MODELS:
            entry = arm(run, name)
            assert entry is not None, f"{run['key']} has no arm {name}"
            value = 100 * entry["test"]["sasv_eer"]
            boot = entry.get("bootstrap_sasv_eer") or {}
            values.append(value)
            lo.append(value - 100 * boot.get("lo", value / 100))
            hi.append(100 * boot.get("hi", value / 100) - value)
            colors.append(MODEL_COLORS[name])
        ax.bar(x, values, color=colors, width=0.68)
        ax.errorbar(x, values, yerr=[lo, hi], fmt="none", ecolor="#222222",
                    elinewidth=0.9, capsize=2.5)
        ax.set_yscale("log")
        gated = 100 * arm(run, "emotion_gated")["test"]["sasv_eer"]
        linear = 100 * arm(run, "linear")["test"]["sasv_eer"]
        ax.set_title(f"{run['label']}\ngated {gated:.2f} %  ·  linear {linear:.2f} %",
                     fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m].replace(" (", "\n(") for m in MODELS],
                           rotation=45, ha="right", fontsize=7.5)
        ax.grid(axis="y", which="both", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("SASV-EER (%) — log scale, lower is better", fontsize=8.5)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Five-arm fusion ladder across four countermeasure regimes "
                 "(test split, grouped-bootstrap 95 % intervals)", fontsize=10)
    return save(fig, out_dir, "fig1_arms_by_regime", plt)


def fig_gate_weights(runs, out_dir, plt, np):
    """Per-emotion gate weights, normalised within each regime.

    Plotted as a percentage deviation from the regime's own mean weight rather
    than as absolute weights: the regimes sit at very different weight levels
    (the failing countermeasure needs a different balance), and the claim under
    test is that each regime's weights do not vary *with emotion*. Absolute
    levels would bury that in scale differences.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), layout="constrained")
    any_weights = False
    emotions: list[str] = []
    for run in runs:
        entry = arm(run, "emotion_gated")
        weights = (entry or {}).get("gate_weights") or {}
        if not weights:
            continue
        any_weights = True
        emotions = [e for e in EMOTION_ORDER if e in weights]
        emotions += [e for e in weights if e not in emotions]
        x = np.arange(len(emotions))
        style = dict(marker="o", markersize=4, linewidth=1.5)
        for ax, key in ((axes[0], "mean_speaker_weight"), (axes[1], "mean_cm_weight")):
            series = np.array([weights[e][key] for e in emotions], dtype=float)
            deviation = 100 * (series / series.mean() - 1)
            spread = 100 * (series.max() - series.min()) / series.mean()
            ax.plot(x, deviation, label=f"{run['label'].split(chr(10))[0]} "
                                        f"({spread:.2f} % spread)", **style)
    if not any_weights:
        raise SystemExit("no gate weights in any run; cannot draw the gate figure")
    for ax, title in zip(axes, ("speaker weight $w_s(\\phi)$",
                                "countermeasure weight $w_c(\\phi)$")):
        ax.set_xticks(np.arange(len(emotions)))
        ax.set_xticklabels(emotions, fontsize=8)
        ax.set_xlabel("ground-truth emotion of the trial", fontsize=8.5)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(-20, 20)
        ax.axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_ylabel("weight deviation from the regime mean (%)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7.5, frameon=False, ncol=2,
               loc="outside lower center")
    fig.suptitle("Gate weights are close to flat across emotions in every regime "
                 "(spread in legend); no regime orders them by emotion", fontsize=10)
    return save(fig, out_dir, "fig2_gate_weights", plt)


def joint_acceptance(manifest_path: Path, asv_path: Path, threshold: float):
    """Emotion x generator spoof acceptance on the test split, from raw ASV scores."""
    with open(manifest_path, encoding="utf-8") as fh:
        trials = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    with open(asv_path, encoding="utf-8") as fh:
        scores = {r["trial_id"]: float(r["speaker_score"]) for r in csv.DictReader(fh)}
    cells: dict[str, dict[str, list[bool]]] = {}
    for trial in trials:
        if trial["trial_type"] != "spoof":
            continue
        if trial["emotion"] not in EMOTION_ORDER:
            continue
        accepted = scores[trial["trial_id"]] >= threshold
        cells.setdefault(trial["emotion"], {}).setdefault(trial["attack"], []).append(accepted)
    return cells


def generator_scores(manifest_path: Path, asv_path: Path):
    """Speaker-score arrays for bona fide targets and for each generator's clones."""
    with open(manifest_path, encoding="utf-8") as fh:
        trials = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    with open(asv_path, encoding="utf-8") as fh:
        scores = {r["trial_id"]: float(r["speaker_score"]) for r in csv.DictReader(fh)}
    bona = [scores[t["trial_id"]] for t in trials if t["trial_type"] == "target"]
    clones: dict[str, list[float]] = {}
    for trial in trials:
        if trial["trial_type"] != "spoof":
            continue
        clones.setdefault(trial["attack"], []).append(scores[trial["trial_id"]])
    return bona, clones


def fig_vulnerability(runs, out_dir, plt, np, manifest, asv_scores, asv_summary):
    summary = json.loads(asv_summary.read_text(encoding="utf-8")) if asv_summary.exists() else {}
    by_generator = (summary.get("spoof_acceptance") or {}).get("by_generator_test") or {}
    has_matrix = manifest is not None and asv_scores is not None and summary

    if has_matrix:
        cells = joint_acceptance(manifest, asv_scores, summary["dev_sv_threshold"])
        generators = [g for g in GENERATOR_LABELS if g in cells.get("angry", {})]
        grids = np.array([
            [100 * np.mean(cells[e][g]) for g in generators] for e in EMOTION_ORDER
        ])
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5),
                                gridspec_kw={"width_ratios": [1.0, 1.35]})
        bars = axes[0]
        names = [g for g in GENERATOR_LABELS if g in by_generator]
        values = [100 * by_generator[g]["accept_rate"] for g in names]
        counts = [by_generator[g]["n"] for g in names]
        bars.barh([GENERATOR_LABELS[g] for g in names], values,
                  color=["#c00000", "#ed7d31", "#2e75b6"][:len(names)], height=0.6)
        for y, (value, n) in enumerate(zip(values, counts)):
            bars.annotate(f"{value:.1f} % (n={n:,})", xy=(value, y), xytext=(2, 0),
                          textcoords="offset points", va="center", fontsize=7.5)
        bars.set_xlim(0, max(values) * 1.35)
        bars.set_xlabel("clones accepted as the cloned speaker (%)", fontsize=8.5)
        bars.set_title("pooled acceptance hides the generator", fontsize=9)
        bars.grid(axis="x", alpha=0.25, linewidth=0.6)
        bars.set_axisbelow(True)

        heat = axes[1]
        image = heat.imshow(grids, cmap="Reds", vmin=0, vmax=max(60, grids.max() * 1.05),
                            aspect="auto")
        heat.set_xticks(np.arange(len(generators)))
        heat.set_xticklabels([GENERATOR_LABELS[g] for g in generators], fontsize=8)
        heat.set_yticks(np.arange(len(EMOTION_ORDER)))
        heat.set_yticklabels(EMOTION_ORDER, fontsize=8)
        for i in range(grids.shape[0]):
            for j in range(grids.shape[1]):
                heat.text(j, i, f"{grids[i, j]:.1f}", ha="center", va="center",
                          fontsize=7.5,
                          color="white" if grids[i, j] > grids.max() * 0.55 else "#222222")
        heat.set_title("acceptance by emotion and generator (%)", fontsize=9)
        fig.colorbar(image, ax=heat, fraction=0.035, pad=0.02, label="% accepted")
        fig.suptitle("ASV vulnerability to emotional clones is generator-dependent and "
                     "emotion-dependent inside each generator", fontsize=9.5)
        return save(fig, out_dir, "fig3_vulnerability", plt)

    if not by_generator:
        print("skip fig3: no ASV summary; run scripts/summarize_real_scores.py first")
        return []
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    names = [g for g in GENERATOR_LABELS if g in by_generator]
    values = [100 * by_generator[g]["accept_rate"] for g in names]
    ax.barh([GENERATOR_LABELS[g] for g in names], values, color="#c00000", height=0.6)
    ax.set_xlabel("clones accepted as the cloned speaker (%)", fontsize=8.5)
    ax.set_title("ASV spoof acceptance by generator (marginals only: the raw ASV score "
                 "file is unavailable, so the emotion x generator panel is omitted)",
                 fontsize=8)
    return save(fig, out_dir, "fig3_vulnerability", plt)


DET_TICKS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)


def _probit(p):
    from scipy.stats import norm

    return norm.ppf(p)


def setup_det_axes(ax):
    ticks = [_probit(t / 100) for t in DET_TICKS]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in DET_TICKS])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in DET_TICKS])
    lo, hi = _probit(0.001), _probit(0.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def sasv_curve(scores, labels, max_points: int = 800):
    """FAR/FRR pairs over thresholds. Positives are targets; negatives are everything else.

    Sorting the scores descending and walking the list means every prefix is the set
    of trials *accepted* at that threshold, so the cumulative positive count is the
    true-accept count — the rejection rate is one minus it.
    """
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    pos = np.cumsum(labels)
    neg = np.cumsum(~labels)
    n_pos, n_neg = pos[-1], neg[-1]
    far = neg / n_neg
    frr = 1.0 - pos / n_pos
    keep = np.r_[True, np.diff(far) > 0]
    far, frr = far[keep], frr[keep]
    if len(far) > max_points:
        idx = np.unique(np.linspace(0, len(far) - 1, max_points).astype(int))
        far, frr = far[idx], frr[idx]
    return np.clip(far, 1e-5, 1 - 1e-5), np.clip(frr, 1e-5, 1 - 1e-5)


def fig_det_curves(runs, out_dir, plt, np, manifest_path):
    """DET curves on probit axes, the conventional speech-verification rendering."""
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = {r["trial_id"]: r for r in csv.DictReader(fh)}
    fig, axes = plt.subplots(1, len(runs), figsize=(13.2, 3.9), sharey=True,
                             layout="constrained")
    for ax, run in zip(axes, runs):
        with open(run["dir"] / "test_trial_scores.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        labels = np.array([manifest[r["trial_id"]]["trial_type"] == "target" for r in rows])
        for name, color in (("linear", MODEL_COLORS["linear"]),
                            ("emotion_gated", MODEL_COLORS["emotion_gated"])):
            scores = np.array([float(r[f"{name}_score"]) for r in rows])
            far, frr = sasv_curve(scores, labels)
            ax.plot(_probit(far), _probit(frr), color=color, linewidth=1.3,
                    label=MODEL_LABELS[name])
            eer = arm(run, name)["test"]["sasv_eer"]
            if 0.001 < eer < 0.4:
                ax.plot(_probit(eer), _probit(eer), marker="o", markersize=3.5,
                        color=color, markerfacecolor="white", markeredgewidth=1.1)
        setup_det_axes(ax)
        ax.set_title(run["label"], fontsize=9)
        ax.set_xlabel("false acceptance rate (%)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=2,
               loc="outside lower center")
    axes[0].set_ylabel("false rejection rate (%)", fontsize=8.5)
    fig.suptitle("DET curves, linear vs gated; open circles mark the equal-error point "
                 "(targets are positives, nontargets and spoofs are negatives)", fontsize=10)
    return save(fig, out_dir, "fig4_det_curves", plt)


def fig_asv_scores(out_dir, plt, np, manifest, asv_scores, asv_summary):
    if manifest is None or asv_scores is None:
        print("skip fig5: raw ASV scores unavailable (they live with the corpora)")
        return []
    summary = json.loads(asv_summary.read_text(encoding="utf-8")) if asv_summary.exists() else {}
    threshold = summary.get("dev_sv_threshold")
    bona, clones = generator_scores(manifest, asv_scores)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bins = np.linspace(-0.1, 1.0, 56)
    ax.hist(bona, bins=bins, density=True, alpha=0.45, color="#404040",
            label=f"bona fide targets (n={len(bona):,})")
    for generator, color in zip(GENERATOR_LABELS, ("#ed7d31", "#2e75b6", "#1f9e89")):
        values = clones.get(generator)
        if not values:
            continue
        ax.hist(values, bins=bins, density=True, alpha=0.40, color=color,
                label=f"{GENERATOR_LABELS[generator]} clones (n={len(values):,})")
    if threshold is not None:
        ax.axvline(threshold, color="#c00000", linewidth=1.2, linestyle="--")
        ax.annotate(f"dev threshold {threshold:+.3f}", xy=(threshold, 0.98),
                    xycoords=("data", "axes fraction"), fontsize=7.5,
                    color="#c00000", rotation=90, va="top", ha="right")
    ax.set_xlabel("ECAPA-TDNN speaker score (higher = more like the claimed speaker)", fontsize=8.5)
    ax.set_ylabel("density", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Clones overlap the bona fide distribution, and the overlap is "
                 "generator-specific", fontsize=9.5)
    return save(fig, out_dir, "fig5_asv_scores", plt)


def main() -> None:
    ap = argparse.ArgumentParser(__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("results/e2e/merged_manifest.csv"))
    ap.add_argument("--results-root", type=Path, default=Path("results/real"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--asv-scores", type=Path, default=None,
                    help="raw ASV score CSV (default: $EASV_DATA_ROOT/real/scores_asv_merged.csv)")
    ap.add_argument("--asv-summary", type=Path, default=Path("results/real/asv_summary.json"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9,
    })

    asv_scores = args.asv_scores
    if asv_scores is None:
        data_root = Path(os.environ.get("EASV_DATA_ROOT", "/home/vct/data/easv"))
        candidate = data_root / "real" / "scores_asv_merged.csv"
        asv_scores = candidate if candidate.exists() else None
    manifest = args.manifest if args.manifest.exists() else None
    if manifest is None:
        print(f"notice: {args.manifest} missing; skipping the manifest-driven figures")

    runs = load_runs(args.results_root)
    written = []
    written += fig_arms_by_regime(runs, args.out, plt, np)
    written += fig_gate_weights(runs, args.out, plt, np)
    if manifest is not None:
        written += fig_vulnerability(runs, args.out, plt, np, manifest, asv_scores,
                                     args.asv_summary)
        written += fig_det_curves(runs, args.out, plt, np, manifest)
        written += fig_asv_scores(args.out, plt, np, manifest, asv_scores, args.asv_summary)
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
