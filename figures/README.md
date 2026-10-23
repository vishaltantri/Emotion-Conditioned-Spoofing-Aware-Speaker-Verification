# Figures

**No standalone figure set exists yet.** The course report carries two diagrams
inside the DOCX; nothing in this directory has been generated. Any figure added here
must be produced by a script from `results/*/results.json` and
`test_trial_scores.csv` — never hand-edited numbers.

Planned figures, in the order they would be most useful:

1. **System diagram** — ASV, SER and countermeasure producers feeding the fusion
   ladder, with the train/dev/test boundary drawn explicitly.
2. **Arm comparison** — SASV-EER with grouped-bootstrap intervals for the five arms,
   one panel per countermeasure regime (in-domain, LA-pretrained, F5-held-out,
   StyleTTS2-held-out), so the sign flip between regimes is visible.
3. **Gate weights by emotion** — the plot that carries the paper's negative finding:
   per-emotion speaker and countermeasure weights are flat, in every regime.
4. **Spoof acceptance by generator and emotion** — the ASV vulnerability, with the
   StyleTTS2 outlier separated instead of averaged away.
5. **DET curves** for the best arm per regime, and an error analysis by attack family.

The data for all five is already stored; the generator script is the missing piece.
