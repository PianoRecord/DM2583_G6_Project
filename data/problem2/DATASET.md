# MAESTRO v3.0.0 subset

Source: [MAESTRO dataset](https://magenta.withgoogle.com/datasets/maestro).
License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Attribution: Google LLC and International Piano-e-Competition.
Citation: Curtis Hawthorne et al., *Enabling Factorized Piano Music Modeling and
Generation with the MAESTRO Dataset*, ICLR 2019.

## Recordings and selection

The subset contains 50 full classical solo-piano recordings, totaling 178.2
minutes and about 188.9 MB of FLAC audio. Original WAV recordings were converted
to mono by stereo averaging, resampled to 22050 Hz using polyphase resampling,
and saved as PCM16 FLAC without time cropping. ZIP entries passed CRC checks;
the source records include SHA256 hashes.

Selection used metadata durations of 90-300 seconds, seed 42, round-robin sampling
across classical composers, and unique composer/title strings. There are 40
recordings from the official training split and 10 from the official test split.
The preprocessing preserves these labels and checks source hashes and cross-split
duplicates. It does not randomly split the subset again.

[playlist.csv](playlist.csv) lists composers, titles, splits, and local filenames.
[sources.json](sources.json) records selection criteria, provenance, and ZIP entry
offsets. This convenience sample is not representative of all classical music.
Different titles may refer to the same work, so grouping of alternate performances
depends on the official split; 50 recordings should not be described as 50
independent works.

## Feature extraction

The pipeline loads complete recordings at 22050 Hz in mono. It uses librosa beat
tracking and `chroma_stft(norm=None)` with `hop_length=512`, averaging chroma frames
between consecutive detected beat boundaries. Leading/trailing audio without a
complete beat interval is excluded. Pitch classes are ordered as
`C, C#, D, D#, E, F, F#, G, G#, A, A#, B`.

Each recording uses the fixed low-energy threshold
`max(1e-5, peak_frame_RMS * 0.001)`. A beat at or below this RMS threshold, or with
chroma total at most `1e-8`, is invalid. Valid beats are L1-normalized over the 12
pitch classes. No global normalization parameters are fitted across recordings.

Starting at the first beat interval, the pipeline cuts non-overlapping 16-beat
windows. A window containing any invalid beat is discarded. Invalid beats keep
their original positions, so windows never concatenate across silence. Incomplete
tails are discarded. The thresholds were fixed before evaluating the test data.

## Processed arrays and validation

- [processed/train.npy](processed/train.npy): `float32`, `(1129, 16, 12)`, from 40 recordings.
- [processed/test.npy](processed/test.npy): `float32`, `(297, 16, 12)`, from 10 recordings.
- [processed/manifest.json](processed/manifest.json): parameters, source/code hashes,
  per-recording statistics, and each window's recording, timestamps, beat indices,
  and array row index.

From the repository root:

```python
import numpy as np
train = np.load('data/problem2/processed/train.npy')
test = np.load('data/problem2/processed/test.npy')
```

Both arrays have finite, non-negative values. Maximum per-beat sum errors are
about `2.98e-7` for train and `2.38e-7` for test, below the `1e-5` tolerance. All 50
recordings contribute valid windows, none were skipped entirely, and the train/test
file sets are disjoint. Training had 12 invalid beats and 5 rejected windows;
test had none. The discarded tails total 329 training beats and 92 test beats.
Every retained window has contiguous original beat indices and no invalid beats.

Data diagnostics are in [results/problem2/data](../../results/problem2/data):

- [real_chroma_heatmap.png](../../results/problem2/data/real_chroma_heatmap.png): three training windows.
- [beat_tracking_spotcheck.png](../../results/problem2/data/beat_tracking_spotcheck.png): onset strength and detected beats for three training recordings.
- [song_summary.csv](../../results/problem2/data/song_summary.csv): per-recording statistics.
- [validation.json](../../results/problem2/data/validation.json): numerical checks and spot-check settings.

The beat spot checks use the training recordings with lowest, median, and highest
beat-interval coefficient of variation: `maestro_034`, `maestro_038`, and
`maestro_017`, each shown for about 12 seconds. Beats often align with onset peaks,
but weak notes, sustained notes, and pauses can be misaligned, including around
4-5 seconds in the `maestro_038` excerpt and parts of `maestro_017`. These are
visual checks without ground-truth beats. Rubato, half/double-tempo errors, and
metrical ambiguity remain possible. Test data were not filtered after visual
inspection. Longer or faster recordings contribute more windows and therefore
receive more weight in evaluation; windows are not independent works.

## Rebuilding

Prepared arrays are included in Git, while raw audio is excluded. Normal training
and evaluation can use the arrays directly. To restore the audio and rebuild
features from the repository root:

```sh
python scripts/download_maestro_subset.py
python scripts/prepare_person_a.py
python scripts/check_setup.py
python -m unittest discover -s tests -v
```

The downloader uses validated HTTP range requests for selected ZIP entries
(about 1.63 GB transferred). The manifest records preprocessing-code and source
list hashes. If the audio, source list, or preprocessing changes, regenerate the
cache so that those hashes match. Data preparation does not train the model.
Preserve attribution, the license, and conversion details when sharing recordings
or derived data.
