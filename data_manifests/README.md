# Manifests and score files

Real manifests are **not** committed: they encode speaker identities and dataset
splits, and some corpora forbid redistribution. Only this schema documentation and
generated examples (which are git-ignored) live here.

Generate a header to start from:

```bash
easv manifest-template --out data_manifests/example.csv
```

## Manifest columns

Required:

| column | meaning |
|---|---|
| `trial_id` | unique id; the join key with score files |
| `split` | `train`, `dev`, `test` (other values produce a warning) |
| `speaker_id` | speaker who produced the audio (source speaker for spoofs, where known) |
| `claimed_speaker` | enrolled identity the trial is presented as |
| `trial_type` | `target`, `nontarget`, `spoof` |
| `emotion` | ground-truth emotion label; `unknown` is allowed but weakens analysis |
| `attack` | `none` for bona fide trials, otherwise the attack/generator id (`a01`, `g5_tts`, …) |

Optional: `path`, `duration_sec`, `dataset`, `source_trial_id`, `notes`.

Validation rules enforced by `easv audit`:

* `target` requires `speaker_id == claimed_speaker`;
* `nontarget` requires them to differ;
* `spoof` requires a named attack, and bona fide trials must have `attack = none`;
* duplicate `trial_id`s are errors; unexpected splits and unparseable durations are warnings;
* every split needs at least one target, one non-target and one spoof trial, otherwise
  SASV-EER, SV-EER and t-DCF are undefined there;
* speaker overlap between `train` and `dev`/`test`, shared generators, and test
  emotions absent from train are reported as findings. Use
  `--allow-speaker-overlap train-dev` when reusing training speakers for
  score-fusion fitting is intentional.

## Score file columns

| column | meaning |
|---|---|
| `trial_id` | must match the manifest |
| `speaker_score` | higher = more like the claimed speaker |
| `cm_score` | higher = more bona fide (ASVspoof convention) |
| `emo_<label>` | optional predicted emotion probability per label; rows are renormalised if they do not sum to 1 (warning) |

Scores are produced by the ASV, countermeasure and emotion models. Keep
ground-truth emotion out of the score file — the pipeline treats the score file as
system output, and mixing labels into it is the easiest way to leak evaluation
information into fusion inputs.

## Manifests built for this project

All three were produced by the builders in `scripts/` and audited with `easv audit`:

| manifest | rows | builder | contents |
|---|---:|---|---|
| `m_esd_real.csv` | 17,900 | `build_manifests.py esd` | 17,500 ESD English targets (10 speakers × 5 emotions × 350) + 400 derived cross-speaker nontargets |
| `m_emospoof.csv` | 36,000 | `build_manifests.py emospoof` | clones from StyleTTS2, F5-TTS and CosyVoice; emotion from the generator's folder tree; `attack` = generator |
| `merged_manifest.csv` | **53,900** | `merge_manifests.py` | the two above with the generator split demoted to the ESD speaker split (4/2/4), so every split has all three generators |
| `m_asvspoof_la.csv` | 25,580 | `build_manifests.py asvspoof` | ASVspoof 2019 LA train only (2,580 bona fide, 22,800 spoof, 200 derived nontargets) |

`easv audit` on the merged manifest returns **0 errors and 4 warnings**: speaker
overlap for the derived nontarget rows, and generator overlap because all three
generators appear in every split. The second is intentional in a corpus this size;
the leave-one-generator-out test is implemented by excluding the generator from
countermeasure *training* (`train_cm_indomain.py --holdout-attack`), not by removing
it from the manifest.

Two rules from the ingestion checklist still deserve stating, because they shaped
these files:

* **Generator identity is never inferred.** `attack` comes from the EmoSpoof folder
  tree or the LA protocol's system column; where the protocol does not disclose the
  source voice, the row says so in `notes` instead of guessing.
* **Emotion labels are ground truth, not predictions.** The emotion *probabilities*
  used by fusion come from the SER producer and live in the score file; the
  manifest's `emotion` column is used for splits, breakdowns and audit only.

## Open data issues

* **ESD licence.** The audio came from a public Hugging Face mirror; a mirror is a
  distribution route, not a rights grant. The ESD research-licence form was never
  submitted ([draft](../docs/ESD_LICENCE_EMAIL.md)), so this must be resolved before
  publishing beyond the course report.
* **No pinned corpus manifest.** The Zenodo md5s from the download campaign are in
  [docs/PROGRESS_LOG.md](../docs/PROGRESS_LOG.md), but the Hugging Face parquet
  shards used for the three generators and for LA are not checksummed or version
  pinned. A rerun months later may not get identical bytes.
* **ASVspoof 2019 evaluation partition and SASV 2022 protocols** were never
  fetched, so no published-baseline reproduction exists.

## Example

```csv
trial_id,split,speaker_id,claimed_speaker,trial_type,emotion,attack,dataset,path,duration_sec,notes
train_spk001_neutral_tar,train,spk001,spk001,target,neutral,none,EXAMPLE,/audio/spk001_0001.wav,3.4,
dev_spk014_angry_a07,dev,spk099,spk014,spoof,angry,a07,EXAMPLE,/audio/a07_0012.wav,4.1,held-out generator for dev
```
