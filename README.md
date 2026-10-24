# Swar Rakshak — Phase 1

Phase 1 provides dataset configuration, pure protocol/metadata parsing, resumable manifests, validation, and one shared audio preprocessing path for future training, evaluation, and live inference.

## Scope and label convention

- `label = 1` means synthetic/spoof.
- `label = 0` means genuine/bonafide.
- ASVspoof 2019 LA train/dev/eval are represented by `train.csv`, `dev.csv`, and `eval.csv`.
- ASVspoof eval and In-the-Wild are test-only. In-the-Wild is always written with `split = test` to `itw_test.csv`; neither source is used for training, tuning, or thresholds in this phase.
- Dataset paths are read from `config.yaml` with `pathlib`; no dataset path is hard-coded in code.

## Dataset sources and licenses

- **ASVspoof 2019 LA:** Edinburgh DataShare / Zenodo record `6906306`, ODC Attribution license. Obtain the dataset from the official record and follow its terms.
- **In-the-Wild:** Hugging Face dataset `mueller91/In-The-Wild`, Apache 2.0. Obtain the dataset from the dataset page and follow its terms.

No dataset is downloaded by this repository. The paths in `config.yaml` are placeholders until the archives are extracted locally.

## Shared preprocessing policy

`src.audio.preprocess()` is the single entry point intended for training, evaluation, and inference. It:

1. converts to mono by averaging channels;
2. replaces NaN/Inf with zero;
3. resamples to 16 kHz using **librosa**, selected because it provides a simple, well-tested high-quality resampling API without introducing a PyTorch dependency into this preprocessing-only phase;
4. warns on empty/near-silent/clipped input;
5. RMS-normalizes to `-20 dBFS`, capped at `+20 dB` gain, while leaving clips below `-60 dBFS` unamplified;
6. creates 4-second windows with a 2-second hop, repeat-tiling clips shorter than 4 seconds and truncating clips longer than 10 seconds.

Constants are defined at the top of `src/audio.py`: target sample rate `16000`, target RMS `-20 dBFS`, maximum gain `20 dB`, near-silence threshold `-60 dBFS`, clipping threshold `0.99`, and silence-statistics threshold `-50 dBFS` over 20 ms RMS frames.

`silence_stats()` returns RMS dBFS plus leading/trailing durations measured using that fixed 20 ms frame-RMS threshold. Manifest `rms_dbfs`, `peak_dbfs`, and silence values are computed from sanitized mono, 16 kHz audio **before RMS normalization**, preserving original loudness differences for later ablations. The `--limit` smoke-test option uses a reproducible seed (`42`) and label-stratified random sampling where possible.

## Development checks with synthetic data

The tests generate all audio in temporary directories; real datasets are not required:

```bash
pytest
python scripts/demo_audio.py
```

The demo prints the normalized waveform shape, 4-second window shapes, and warnings for synthetic 2 s, 4 s, and 10 s clips.

## RUN LATER — after data extraction

1. Fill `config.yaml`:

```yaml
asvspoof_la_root: "path/to/asvspoof2019_la"
itw_root: "path/to/in_the_wild"
```

2. Inspect protocol formats and local paths:

```bash
python -m src.build_manifests inspect
```

3. Build a small smoke-test set:

```bash
python -m src.build_manifests build --limit 200
```

4. Build the full manifests:

```bash
python -m src.build_manifests build
```

5. Validate paths, labels, overlap, missing values, and count references:

```bash
python -m src.validate_manifests
```

The builder requires exactly one `*cm.train*`, `*cm.dev*`, and `*cm.eval*` protocol file under `ASVspoof2019_LA_cm_protocols/`. It extracts duration/sample rate from `soundfile.info`, computes shared-preprocessing features with a multiprocessing `tqdm` pass, and stores a resumable `.feature_cache.csv` under `manifests/`.
