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

## Building a real manifest

1. Confirm access, licences and the official partitions for every corpus involved
   (ASVspoof 2019 LA/PA, ASVspoof 2021 LA/PA/DF, SASV 2022, EmoSpoof-TTS, and an
   emotion corpus such as RAVDESS or CREMA-D if separate emotion labels are needed).
2. Map bona-fide/spoof labels, speaker identities and attack ids from the official
   protocol files. Do not infer generator identity when the protocol does not
   provide it.
3. Decide how emotion labels are obtained. A corpus without emotion annotations
   does not supply them; using an emotion classifier on unlabelled audio turns the
   evaluation of emotion conditioning into an evaluation of that classifier's
   errors, which must be reported as such.
4. Keep splits speaker-disjoint unless the protocol says otherwise, and make the
   held-out generator/emotion set explicit in the manifest `notes`.
5. Run `easv audit --manifest <file> --out results/audit` and resolve errors before
   scoring anything.
6. Publish the split *procedure* (not restricted audio) so the evaluation is
   reproducible.

## Example

```csv
trial_id,split,speaker_id,claimed_speaker,trial_type,emotion,attack,dataset,path,duration_sec,notes
train_spk001_neutral_tar,train,spk001,spk001,target,neutral,none,EXAMPLE,/audio/spk001_0001.wav,3.4,
dev_spk014_angry_a07,dev,spk099,spk014,spoof,angry,a07,EXAMPLE,/audio/a07_0012.wav,4.1,held-out generator for dev
```
