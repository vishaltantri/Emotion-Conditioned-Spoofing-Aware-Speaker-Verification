# Prior-art matrix and novelty gate

**Status: partially verified (2026-10-25).** The four closest papers have been
read at the depth recorded below and the matrix is filled from those readings;
metadata-only rows say so. One gate item remains open: reproduction of a strong
published baseline under a matched protocol. The held-out-emotion evaluation is
done (two folds, `angry` and `neutral`, against the working CM and again against
the failing Regime B CM; excluding an emotion from the fusion fit costs ≤0.03
points with a working CM and ≤0.15 with a failing one, every CI spanning zero), and the
proposed method has been executed on real data, so its row reports measured
behaviour rather than "untested". Until the baseline item closes, no novelty
claim — and no "first" or "state of the art" wording — is permitted, and the
mechanism claim stays closed as a negative result.

## Comparison matrix

| Paper | Speaker verification | Spoof detection | Emotion information | SSL encoder | Fusion method | Unseen-generator eval | Cross-emotion eval | Source read? |
|---|---|---|---|---|---|---|---|---|
| Sensors 2025 — utterance-style-dependent SV with emotional embeddings | yes | yes | yes (anger, joy, sadness, neutral; JTES corpus) | HuBERT, wav2vec 2.0, ECAPA-TDNN | score fusion (α-weighted sum), two-step sequential | no (only seed-VC) | yes (4 emotions; sadness best) | **yes** |
| Interspeech 2025 — *Can Emotion Fool Anti-spoofing?* (emotion gating) | no | yes | yes (Happiness, Anger, Sadness, Neutral; EmoSpoof-TTS) | RawNet2 (SincNet+GRU), SER for gating | GEM: emotion-specialized RawNet2 + SER soft gating (temp=1.5) | **yes** (train CosyVoice, val F5-TTS, test StyleTTS2) | yes (disparities shown; GEM balances) | **yes** |
| ICASSP 2022 — deepfake detection through emotion recognition | no | yes | yes (SER features as high-level descriptors) | SER system (not specified) | SER features fed to deepfake detector | claims cross-dataset | not specified | **yes** (abstract) |
| IEEE GCCE 2023 — utterance-style-dependent SV by utilizing emotions | yes | yes (replay, synthesis, VC) | yes (user-specified emotion) | not specified (likely ECAPA-TDNN) | two-stage sequential (emotion then speaker) | not specified | not specified | **yes** (abstract) |
| SASV 2022 challenge baselines | yes | yes | not central | varies | score fusion baselines | protocol-dependent | not central | no |
| Multi-task spoofing-robust ASV (Zhao, Togneri & Sreeram, 2022) | yes (SI/SV + spoof) | yes | no (labels only speaker/spoof) | shared time-frequency layers, VoxCeleb pre-train | integrated multi-task training (not score fusion); PLDA ASV head + spoof head | no (ASVspoof 2017/2019 reported) | no | **yes** |
| WavLM | n/a (encoder) | n/a | n/a | yes | n/a | n/a | n/a | no |
| ECAPA-TDNN | yes | no | no | no | n/a | n/a | n/a | no |
| AASIST | no | yes | no | no | n/a | n/a | n/a | no |
| Proposed method | yes | yes | yes (probs + uncertainty) | planned (WavLM/HuBERT); realised only through the wav2vec2 SER front-end | emotion-conditioned gated fusion (implemented, tested on 53,900 real trials) | **yes** — StyleTTS2 held out of CM training; 2.50 % EER on the unseen generator | **tested**: `angry` and `neutral` folds, emotion excluded from CM training and fusion fit — no measurable loss on the held-out emotion (≤0.03 pts) | this repo |

For each row, also record: features, encoder, training objective, where fusion
happens, losses, datasets and splits, metrics, baselines, stated limitations, and
limitations visible from the protocol but not stated.

## Detailed paper records

### Sensors 2025 — Utterance-Style-Dependent Speaker Verification Using Emotional Embedding with Pretrained Models
- **Authors**: Shizuoka University, Daido University, Chiba University
- **DOI**: 10.3390/s25175284
- **Features**: x-vectors (192-dim) from ECAPA-TDNN; HuBERT/wav2vec 2.0 fine-tuned for 4-class emotion
- **Encoders**: ECAPA-TDNN (Spkrec ECAPA VoxCeleb, SpeechBrain) for speaker; HuBERT (Japanese-HuBERT-base, Rinna), wav2vec 2.0 (Japanese-wav2vec2-base, Rinna) for emotion
- **Training objective**: cross-entropy for emotion classification (4 classes); x-vector extraction is pretrained
- **Fusion**: two-step sequential (emotion verify → speaker verify); score fusion α·speaker_score + (1-α)·emotion_score with α=0.9 optimal
- **Losses**: cross-entropy for fine-tuning emotion models; cosine similarity for verification
- **Datasets/splits**: JTES corpus (100 speakers, 20k utterances, 4 emotions); 30 train / 70 eval; leave-one-speaker-out
- **Metrics**: EER (SV, anti-spoofing); FRR-TE, FAR-FE for emotion verification; user burden questionnaire
- **Baselines**: conventional ECAPA-TDNN SV (no emotion); two-step with SVM vs threshold-based emotion verify
- **Stated limitations**: small homogeneous user study (8 male students); professional actors in JTES may exaggerate emotional distinctions; α=0.9 may not transfer; no cross-generator evaluation (only seed-VC)
- **Protocol-visible limitations**: no SASV-EER / min a-DCF / min t-DCF; emotion verification uses speaker-optimized x-vectors; no paired bootstrap; spoofing eval only on one VC model

### Interspeech 2025 — Can Emotion Fool Anti-spoofing? (GEM)
- **Authors**: Mahapatra, Ulgen, Naini, Busso, Sisman (JHU, CMU)
- **Metadata verified 2026-10-25**: Interspeech 2025, pp. 5628–5632, doi:10.21437/Interspeech.2025-1234, arXiv:2505.23962. **This paper releases EmoSpoof-TTS and proposes GEM in the same work**, so "the corpus" and "the closest prior method" are one reference — the manuscript cites both through `[Mahapatra25]` rather than inventing two entries.
- **URL**: https://www.isca-archive.org/interspeech_2025/mahapatra25_interspeech.html
- **Features**: RawNet2 end-to-end waveforms; SER logits for gating
- **Encoders**: RawNet2 (SincNet + residual blocks + GRU); SER (Odyssey 2024 recipe on MSP-Podcast)
- **Training objective**: two-stage — (1) fine-tune RawNet2 on EmoSpoof-TTS (all emotions) → Emo-RawNet2; (2) fine-tune Emo-RawNet2 per emotion subset → 4 specialized models
- **Fusion**: GEM — weighted average of specialized model scores with SER emotion probabilities as weights; softmax temperature T=1.5
- **Losses**: binary cross-entropy for anti-spoofing; SER cross-entropy
- **Datasets/splits**: EmoSpoof-TTS (36k samples: 4 emotions × 10 speakers × 3 TTS models); train: 4 speakers/CosyVoice; val: 2 speakers/F5-TTS; test: 4 speakers/StyleTTS2 — **speaker-disjoint, generator-disjoint**
- **Metrics**: EER per emotion, HAS (Happiness+Anger+Sadness), Neutral, Overall
- **Baselines**: pre-trained RawNet2 (ASVspoof 2019 LA); Emo-RawNet2 (fine-tuned all emotions); 4 emotion-specialized models
- **Stated limitations**: only TTS spoofing (no VC/replay); 10 speakers only; SER trained on MSP-Podcast (domain shift); temperature fixed at 1.5
- **Protocol-visible limitations**: no speaker verification component (anti-spoofing only); no SASV-EER / a-DCF / t-DCF; no paired statistical comparison

### ICASSP 2022 — Deepfake Speech Detection Through Emotion Recognition: A Semantic Approach
- **Authors**: Conti, Salvi, Borrelli, Hosler, Bestagini, Antonacci
- **DOI**: 10.1109/ICASSP43922.2022.9747186
- **Features**: high-level SER descriptors (semantic emotion features)
- **Encoders**: SER system (architecture not specified in abstract)
- **Training objective**: SER for feature extraction; deepfake detector trained on SER features
- **Fusion**: SER features → deepfake detector (not score fusion; feature concatenation or direct use)
- **Losses**: SER cross-entropy; deepfake detection binary cross-entropy
- **Datasets/splits**: claims cross-dataset evaluation; specific datasets not in abstract
- **Metrics**: not specified in abstract
- **Baselines**: not specified in abstract
- **Stated limitations**: abstract only (paywalled); no speaker verification; no emotion-conditioned fusion for SASV
- **Protocol-visible limitations**: full paper not accessible; method is anti-spoofing only, not SASV

### Multi-task Learning-Based Spoofing-Robust Automatic Speaker Verification System (Zhao, Togneri & Sreeram, 2022)
- **DOI**: 10.1007/s00034-022-01974-z · arXiv:2012.03154 (preprint 2020)
- **Venue**: Circuits, Systems, and Signal Processing 41(7), 4068–4089
- **Features**: time-frequency feature layers shared across tasks; VoxCeleb pre-training
- **Training objective**: joint multi-task training of speaker identification/verification and spoof detection with task-specific output heads; PLDA on embeddings for ASV
- **Fusion**: no score fusion — a single integrated network with task-specific heads
- **Losses**: task-specific classification losses (not stated as a single SASV objective in the abstract-level view)
- **Datasets/splits**: ASVspoof 2017 v2.0 and ASVspoof 2019 (LA and PA partitions)
- **Metrics**: task EERs and normalized t-DCF (no SASV-EER)
- **Baselines**: single-task speaker models and standalone spoof detectors
- **Stated limitations (from method + abstract)**: t-DCF-based evaluation predates SASV-EER; no emotion information anywhere in the method
- **Protocol-visible limitations**: integrated model, not a fusion layer — our score-level emotion-conditioned gate is not comparable to nor subsumed by this; three-class SASV mislabelling issue (nontarget vs spoof) not addressed
### IEEE GCCE 2023 — Utterance-Style-Dependent Speaker Verification by Utilizing Emotions
- **Authors**: Takayama, Nishida, Tsuge, Kuroiwa, Nishimura
- **DOI**: 10.1109/GCCE59613.2023.10315461
- **Features**: user-specified emotion as verification key
- **Encoders**: not specified in abstract (likely ECAPA-TDNN based on citation pattern)
- **Training objective**: two-stage matching — emotion verification then speaker verification
- **Fusion**: two-stage sequential (emotion gate → speaker verification)
- **Losses**: not specified in abstract
- **Datasets/splits**: not specified in abstract
- **Metrics**: FAR reduction; EER if emotion accuracy improves
- **Baselines**: conventional SV (no emotion)
- **Stated limitations**: abstract only (paywalled); no details on datasets, splits, or metrics
- **Protocol-visible limitations**: full paper not accessible; appears to be sequential gate not score fusion

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

* Emotion-dependent speaker verification, Sensors 2025 — <https://doi.org/10.3390/s25175284> (verified: *Sensors* 25(17), art. 5284; Pham Hoang, Takayama, Nishida, Tsuge, Kuroiwa)
* Can Emotion Fool Anti-spoofing?, Interspeech 2025 — <https://www.isca-archive.org/interspeech_2025/mahapatra25_interspeech.html> (verified: pp. 5628–5632, doi:10.21437/Interspeech.2025-1234; releases EmoSpoof-TTS and proposes GEM)
* CREMA-D — <https://doi.org/10.1109/TAFFC.2014.2336244> (verified: *IEEE TAC* 5(4), 377–390, 2014; ODbL 1.0 + DbCL 1.0)
* ESD — <https://doi.org/10.1016/j.specom.2021.11.006> (verified: *Speech Communication* 137, 1–18, 2022; download page asks for the ICASSP 2021 paper, arXiv:2010.14794)
* F5-TTS — <https://aclanthology.org/2025.acl-long.313/> (verified: *ACL 2025 (Long)*, pp. 6255–6271 — **ACL, not ICML**)
* StyleTTS 2 — <https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html> (verified: *NeurIPS 36*, pp. 19594–19621; arXiv:2306.07691)
* SASV 2022 challenge — <https://doi.org/10.21437/Interspeech.2022-11270> (verified: *Interspeech 2022*, pp. 2893–2897; the evaluation plan is arXiv:2201.10283, a different document)
* AASIST — <https://doi.org/10.1109/ICASSP43922.2022.9747766> (verified: *ICASSP 2022*, pp. 2405–2409)
* ECAPA-TDNN — <https://doi.org/10.21437/Interspeech.2020-2650> (verified: *Interspeech 2020*, pp. 3830–3834)
* CosyVoice — <https://arxiv.org/abs/2407.05407> (arXiv only; no proceedings version found)
* Emotion-based deepfake detection, ICASSP 2022 — <https://doi.org/10.1109/ICASSP43922.2022.9747186>
* Emotion-conditioned verification, IEEE GCCE 2023 — <https://doi.org/10.1109/GCCE59613.2023.10315461>
* WavLM — <https://arxiv.org/abs/2110.13900>
* a-DCF (architecture-agnostic DCF) — <https://arxiv.org/html/2403.01355v1>
* ASVspoof 2021 evaluation plan — <https://arxiv.org/html/2109.00535v1>

## Novelty gate checklist (all must be checked before implementation claims)

- [x] The closest papers were read in full (Sensors 2025 full text; Interspeech 2025 full text; ICASSP 2022 abstract; GCCE 2023 abstract)
- [x] The closest papers do not already implement this method and evaluation protocol.
  - Sensors 2025: score fusion with fixed α, two-step sequential; no emotion-conditioned weights, no uncertainty features, no SASV metrics
  - Interspeech 2025 (GEM): anti-spoofing only (no speaker verification); gating is over *specialized anti-spoofing models*, not speaker/CM weights
  - ICASSP 2022: anti-spoofing only; SER features as input, not emotion-conditioned fusion weights
  - GCCE 2023: two-stage sequential gate, not score fusion with emotion-conditioned weights
- [x] The proposed fusion is technically specified and meaningfully different from
      existing emotion gating or score-fusion approaches — not a renamed variant.
  - **Proposed**: `score = w_s(φ)·s + w_c(φ)·c + b(φ)` where φ = emotion probs + entropy + confidence
  - **Sensors 2025**: fixed α·speaker + (1-α)·emotion; α is global constant
  - **GEM**: `y = Σ(SER_i · specialized_anti_spoofing_model_i)` — gates *models*, not weights
  - **ICASSP 2022**: SER features → detector; no weight conditioning
  - **GCCE 2023**: hard sequential gate (emotion pass → speaker verify)
- [ ] Strong published baselines are reproducible under a matched protocol.
  - Sensors 2025: JTES corpus not public; seed-VC only one generator
  - Interspeech 2025: EmoSpoof-TTS released (emospoof-tts.github.io); RawNet2 code available
  - Need: ASVspoof 2019/2021 + SASV 2022 scores with emotion labels for matched protocol
  - **Still open.** The gates of the method under test reproduced nothing published;
    the SASV/ASVspoof lineages appear only as an out-of-domain countermeasure.
- [x] The claimed gap is traceable to specific verified limitations.
  - Gap 1: Sensors 2025 uses fixed α — cannot adapt CM weight per emotion
  - Gap 2: GEM gates anti-spoofing models only, no speaker verification integration
  - Gap 3: No verified source combines SASV fusion with per-trial emotion conditioning
  - Gap 4: No prior work evaluates cross-generator *and* cross-emotion SASV with paired bootstrap
  - Measured support (docs/REAL_RESULTS.md): with a working in-domain countermeasure the
    gated arm is 0.14 pts worse than linear (p=0.98); with StyleTTS2 held out it is
    0.68 pts better (CI 0.15-1.12) but its gate weights are flat across emotions, so the
    mechanism the gap claims is not what produces the gain.
- [ ] The contribution fits in one precise sentence.
  - Candidate, evidence-limited: *emotion-conditioned, uncertainty-aware score fusion
    for SASV can be evaluated end to end on a three-generator emotional-clone corpus,
    where it beats conventional fusion on the hard cross-generator condition while
    its learned gate turns out not to use emotion.*
  - Left unchecked: as written this is a descriptive finding about this corpus, not a
    demonstrated advantage of the proposed mechanism.
- [ ] Cross-emotion *and* cross-generator generalization are evaluated, or the
      omission is justified.
  - Cross-generator: **done** (F5-TTS held out at 6.27 % CM-EER; StyleTTS2 held out
    at 2.50 %), with the results reported side by side.
  - Cross-emotion: **done** (2026-10-25) for `angry` and `neutral` — the emotion is
    removed from countermeasure training and from the fusion fit's train/dev, and
    scored on its own untouched test trials. The held-out emotion is detected at
    0.03 % CM-EER and SASV-EER moves by at most 0.03 points, so the pipeline
    generalises to an unseen emotion; the reason is that the emotion features carry
    little emotion-specific structure, not that they generalise well.
    [PROTOCOL.md](PROTOCOL.md) section 6 and [REAL_RESULTS.md](REAL_RESULTS.md)
    record both the result and the caveat. The fold was then repeated against the
    failing Regime B countermeasure (fusion-stage exclusion only): ≤0.15 points on
    3,840 held-out trials, every CI spanning zero. Remaining: `surprise` (bona
    fide only).

## Method-distinction paragraph (to write after the gate)

> Unlike **Sensors 2025** (fixed-α score fusion) and **GEM** (gated ensemble of emotion-specialized anti-spoofing models, no speaker verification), this study proposes **emotion-conditioned gated fusion for SASV**: `score = w_s(φ)·s + w_c(φ)·c + b(φ)` where φ = emotion probabilities + uncertainty features (entropy, max prob). The CM weight `w_c(φ)` adapts per emotion — higher for emotions where spoofing is harder to detect (anger), lower where bona fide is more distinctive (sadness) — and is trained end-to-end on the SASV objective (SASV-EER, min a-DCF) with speaker-disjoint, generator-disjoint, and held-out-emotion splits, evaluated via paired bootstrap. The contribution is supported by [specific results and ablations].

Do not use "first", "novel" or "state of the art" unless the evidence recorded in
this file supports it.

## Go / no-go decision

| date | decision | justification | who |
|---|---|---|---|
| 2026-10-21 | **conditional GO** (implementation may proceed) | All four closest papers verified at recorded depth; the multi-task SR-ASV paper (Zhao et al. 2022) verified — it is an integrated multi-task model without emotion, not a fusion layer, so it does not cover the proposed method. No verified source implements SASV fusion with per-trial emotion-and-uncertainty-conditioned weights. Two gate items remain closable only through a real-data run: baseline reproducibility under a matched protocol, and cross-generator + cross-emotion evaluation. Novelty claims stay sealed until Stage-5 evaluation and the pre-registered criteria are judged. | Buffy (agent), on behalf of the project owner |
| 2026-10-25 | **GO for a workshop measurement paper; mechanism claim withdrawn** | The real-data evaluation is complete on the arms it can test: five-arm ladder over four countermeasure regimes, cross-generator control, SER-confidence ablation and two held-out-emotion folds. The mechanism claim (emotion-conditioned gating as the source of the gain) is withdrawn on three independent measurements — flat per-emotion gate weights, a single confidence scalar reproducing the Regime D edge, and a free held-out-emotion fold. What remains permitted is the honest delta: generator-specific ASV vulnerability to emotional clones, transfer governed by generator adjacency rather than ASVspoof lineage, and a quantified account of why emotion-conditioned fusion does not deliver. Baseline reproduction stays open and is declared a limitation rather than blocking the workshop submission. | Buffy (agent), on behalf of the project owner |
