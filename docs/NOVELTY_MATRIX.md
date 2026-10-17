# Prior-art matrix and novelty gate

**Status: unverified.** Every cell below is deliberately `TODO`. Filling a cell
requires reading the full paper and recording what it actually does; cells must
never be filled by inference from the title or abstract. This file is Phase 1 of
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) and it blocks any novelty claim.

## Comparison matrix

| Paper | Speaker verification | Spoof detection | Emotion information | SSL encoder | Fusion method | Unseen-generator eval | Cross-emotion eval | Source read? |
|---|---|---|---|---|---|---|---|---|
| Sensors 2025 — utterance-style-dependent SV with emotional embeddings | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| Interspeech 2025 — *Can Emotion Fool Anti-spoofing?* (emotion gating) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| ICASSP 2022 — deepfake detection through emotion recognition | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| IEEE GCCE 2023 — emotion-conditioned authentication | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| SASV 2022 challenge baselines | yes | yes | not central | varies | score fusion baselines | protocol-dependent | not central | no |
| Multi-task spoofing-robust ASV (2022) | TODO | TODO | TODO | TODO | TODO | TODO | TODO | no |
| WavLM | n/a (encoder) | n/a | n/a | yes | n/a | n/a | n/a | no |
| ECAPA-TDNN | yes | no | no | no | n/a | n/a | n/a | no |
| AASIST | no | yes | no | no | n/a | n/a | n/a | no |
| Proposed method | yes | yes | yes (probs + uncertainty) | planned (WavLM/HuBERT) | emotion-conditioned gated fusion (implemented, untested on real data) | must test | must test | partially (this repo) |

For each row, also record: features, encoder, training objective, where fusion
happens, losses, datasets and splits, metrics, baselines, stated limitations, and
limitations visible from the protocol but not stated.

## Search strings

```
"emotion-aware" "speaker verification" spoofing
"emotion-conditioned" spoofing-aware speaker verification
"emotion" "SASV" speaker verification
"emotion recognition" audio deepfake detection
"emotional speech" anti-spoofing TTS
"emotion gating" anti-spoofing
WavLM emotion spoof speaker verification fusion
"self-supervised" "spoofing-aware speaker verification"
"cross-generator" emotional speech deepfake detection
"unseen emotion" spoofing detection
"speaker verification" "emotion embedding" pretrained
"SASV" score fusion emotion
```

Sources: Google Scholar, IEEE Xplore, ISCA Archive, ACL Anthology, SpringerLink,
ScienceDirect, arXiv — with date filters and both exact-title and concept searches.

## Starting references (verify metadata before citing)

* Emotion-dependent speaker verification, Sensors 2025 — <https://doi.org/10.3390/s25175284>
* Can Emotion Fool Anti-spoofing?, Interspeech 2025 — <https://www.isca-archive.org/interspeech_2025/mahapatra25_interspeech.html>
* Emotion-based deepfake detection, ICASSP 2022 — <https://doi.org/10.1109/ICASSP43922.2022.9747186>
* Emotion-conditioned verification, IEEE GCCE 2023 — <https://doi.org/10.1109/GCCE59613.2023.10315461>
* SASV 2022 challenge — <https://www.isca-archive.org/interspeech_2022/jung22c_interspeech.html>
* WavLM — <https://arxiv.org/abs/2110.13900>
* ECAPA-TDNN — <https://arxiv.org/abs/2005.07143>
* AASIST — <https://arxiv.org/abs/2110.01200>
* a-DCF (architecture-agnostic DCF) — <https://arxiv.org/html/2403.01355v1>
* ASVspoof 2021 evaluation plan — <https://arxiv.org/html/2109.00535v1>

## Novelty gate checklist (all must be checked before implementation claims)

- [ ] The closest papers were read in full, not skimmed.
- [ ] The closest papers do not already implement this method and evaluation protocol.
- [ ] The proposed fusion is technically specified and meaningfully different from
      existing emotion gating or score-fusion approaches — not a renamed variant.
- [ ] Strong published baselines are reproducible under a matched protocol.
- [ ] The claimed gap is traceable to specific verified limitations.
- [ ] The contribution fits in one precise sentence.
- [ ] Cross-emotion *and* cross-generator generalization are evaluated, or the
      omission is justified.

## Method-distinction paragraph (to write after the gate)

> Unlike [closest prior work], which [specific verified limitation], this study
> [specific method or protocol] and evaluates it under [specific, justified
> generalization condition]. The contribution is supported by [specific results
> and ablations].

Do not use "first", "novel" or "state of the art" unless the evidence recorded in
this file supports it.

## Go / no-go decision

| date | decision | justification | who |
|---|---|---|---|
| — | pending | gate not closed | — |
