# MAESTRO v3.0.0 — 50 classical solo-piano recordings

Source: https://magenta.withgoogle.com/datasets/maestro
License: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/
Attribution: Google LLC and International Piano-e-Competition.
Citation: Curtis Hawthorne et al., “Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset”, ICLR 2019.

## Local collection

50 full recordings, 178.2 minutes; 188.9 MB in `audio/`.
Converted from the original WAV recordings to mono 22050 Hz PCM16 FLAC
by stereo averaging and polyphase resampling. No time cropping.
Each source ZIP entry passed its CRC check; local SHA256 values are recorded.

`playlist.csv` lists composers, titles, official splits and local filenames.
`sources.json` contains provenance, deterministic selection criteria and download offsets.
To restore missing files, run `python scripts/download_maestro_subset.py` from the project root.

Selection: metadata duration 90–300 seconds, seed 42, round-robin across
classical composers, unique composer/title pairs, 40 official train and 10 official test recordings.
This is a convenience sample, not a representative sample of classical music.
Selection does not establish beat-tracking quality; rubato may still affect extraction.

The preprocessing now reads the official 40-train / 10-test labels from `sources.json`,
verifies source audio hashes, and never rerandomizes this MAESTRO subset.
Exact composer/title strings are unique in the source selection, but different strings
can describe the same work; these remain in their official split.
See `PERSON_A_HANDOFF.md` for preprocessing and output details.

Audio remains excluded from Git. Preserve attribution, license and conversion details
when sharing this subset or derived data. Model training has not been run by this download task.
