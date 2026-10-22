# First real results: EmoSpoof-TTS clones vs ESD bona fide

Provenance: **real**. Corpus: ESD English speakers 0011–0020 (bona fide, emotion
labels from the corpus folders) + EmoSpoof-TTS F5-TTS and CosyVoice subsets
(spoof clones, emotion labels from the generator archives; 12,000 wavs each).
Speaker-disjoint 4/2/4 split. 41,900 trials in total.

Component scores, all measured on this data:

| component | model | EER (test) | notes |
|---|---|---:|---|
| ASV | SpeechBrain ECAPA-TDNN, cosine to train-split centroid | **3.12 %** (SV-EER) | accepts 82.9 % of bona fide targets, **55.2 % of clones** |
| SER | wav2vec2-base-superb-er, 4-way softmax | 41.8 % acc. on bona fide, 30.7 % on clones | clones saturate it: mean max prob 0.83 while wrong |
| CM (in-domain) | log-mel CNN, train split only | **0.09 %** (SPF-EER) | optimistic by construction: sees these two generators during training |
| CM (AASIST, LA-pretrained) | official 2019-LA checkpoint | **50.8 %** | no transfer to EmoSpoof clones at all |

## Regime A — usable countermeasure (in-domain CM)

`results/real/run1`, `configs/real_run.yaml`, thresholds from dev only.

| model | SASV-EER | 95 % CI | min a-DCF | ECE | params |
|---|---:|---|---:|---:|---:|
| speaker_only | 0.3309 | 0.2605–0.3816 | 0.7294 | 0.125 | 2 |
| linear | 0.0060 | 0.0029–0.0074 | 0.0385 | 0.051 | 3 |
| emotion_additive | 0.0066 | 0.0037–0.0083 | 0.0410 | 0.051 | 9 |
| emotion_interaction | 0.0055 | 0.0031–0.0066 | 0.0359 | 0.026 | 22 |
| emotion_gated | 0.0073 | 0.0031–0.0093 | 0.0353 | 0.004 | 43 |

Paired bootstrap against `linear`: additive −0.05 pts (p=0.73), interaction
+0.05 pts (p=0.34), gated **−0.14 pts (p=0.98, i.e. slightly worse)**. No emotion
arm beats plain fusion; the countermeasure already removes almost all spoof
acceptance, so the remaining 0.6 % is ASV error and emotion carries no usable
extra signal. The gated model is the best calibrated (ECE 0.004) but that is not
a performance win.

## Regime B — countermeasure that does not transfer (AASIST-LA)

`results/real/run2_oodcm`, same manifest, same ASV/SER scores, CM swapped for the
out-of-domain AASIST checkpoint (SPF-EER 50.8 %, min t-DCF 0.999).

| model | SASV-EER | 95 % CI |
|---|---:|---|
| speaker_only | 0.3309 | 0.2605–0.3816 |
| linear | 0.3244 | 0.2516–0.3806 |
| emotion_additive | 0.2654 | 0.1882–0.3506 |
| emotion_interaction | 0.2661 | 0.1913–0.3470 |
| emotion_gated | 0.2610 | 0.1786–0.3483 |

Paired bootstrap vs `linear`: additive **+5.28 pts** (2.97–8.88, p<0.001),
interaction +5.19 (2.94–8.57, p<0.001), gated **+5.57** (2.86–9.85, p<0.001).

## The catch: why the Regime B gain is not (only) emotion conditioning

The SER front-end is confidently wrong on clones: 30.7 % accuracy — barely above
the 25 % chance level — with a mean maximum probability of 0.83, against 41.8 %
and 0.777 on bona fide speech. The emotion probability vector therefore carries
a **spoof cue**: "confidently wrong emotion" correlates with being synthetic.

That is enough to explain a 5-point SASV-EER gain from emotion features without
any genuine emotion-conditioned speaker/countermeasure modelling, and it is a
labelling hazard for any paper claim built on a weak CM. The honest reading of
Regime B is: *when the countermeasure fails, the emotion recogniser's
out-of-distribution behaviour acts as a substitute detector*. Confirmatory test
for a future pass: retrain the fusion arms with emotion features replaced by an
explicit "SER confidence" scalar and see whether the gain survives; and evaluate
on the third generator (StyleTTS2) once its subset lands, where SER behaviour may
differ.

## What these results do and do not establish

Established (measured, real audio, dev-only thresholds, test reported once):

* an off-the-shelf ECAPA ASV accepts 55.2 % of neural emotional clones as the
  cloned speaker, and acceptance is emotion-dependent (angry 64.6 %, neutral
  61.7 %, sad 47.5 %, happy 47.2 %);
* standard spoofing-aware fusion removes nearly all of that vulnerability when
  the countermeasure works (33.1 % → 0.6 % SASV-EER);
* the LA-trained AASIST checkpoint does not transfer to EmoSpoof-TTS clones
  (EER 50.8 %) — a CM trained on ASVspoof-2019 attack families is not a
  countermeasure for modern neural TTS in a different recording domain;
* with an in-domain CM, emotion-conditioned fusion adds nothing (and the gated
  arm is marginally worse);
* with a failing CM, emotion features cut SASV-EER by ~5.5 points, but the
  mechanism is most likely SER's OOD behaviour acting as a spoof cue.

Not established:

* any cross-generator benefit (third generator still downloading);
* any claim that the proposed gated fusion is better than plain fusion — on this
  data it is not, and saying otherwise would be dishonest;
* an in-domain AASIST re-trained on ASVspoof 2019 LA (blocked on disk space).

## Reproduce

```bash
D=/home/vct/data/easv
python scripts/build_manifests.py esd       --esd-root $D/esd --out results/e2e/m_esd_real.csv
python scripts/build_manifests.py emospoof  --emo-root $D/emospoof --out results/e2e/m_emospoof.csv
python scripts/merge_manifests.py --esd results/e2e/m_esd_real.csv \
    --emospoof results/e2e/m_emospoof.csv --out results/e2e/merged_manifest.csv
python scripts/producer_asv.py --manifest results/e2e/merged_manifest.csv --out $D/real/scores_asv_merged.csv
python scripts/train_cm_indomain.py --manifest results/e2e/merged_manifest.csv --out-dir $D/real/cm_indomain
python scripts/producer_ser.py --manifest results/e2e/merged_manifest.csv --out $D/real/scores_ser_merged.csv
python scripts/combine_scores.py --speaker $D/real/scores_asv_merged.csv \
    --cm $D/real/cm_indomain/scores_cm_indomain.csv \
    --emotion $D/real/scores_ser_merged.csv --out $D/real/scores_combined.csv
easv evaluate --manifest results/e2e/merged_manifest.csv \
    --scores $D/real/scores_combined.csv --config configs/real_run.yaml --out results/real/run1
```
