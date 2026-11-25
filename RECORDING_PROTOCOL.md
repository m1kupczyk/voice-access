# Own-corpus recording protocol

Parameters (durations, number of prompts, devices) are set in `config.yaml` under `own_corpus`.

## Consent
- Each speaker signs a written consent form before recording (purpose: thesis research; voice data is a GDPR special category; raw audio stays local; withdrawal possible at any time, which deletes the speaker's files and index rows).
- The signed form is stored on paper or offline, never in the repository. It is linked to the speaker only through a private key sheet (name ↔ `spkNN`) kept outside the repository.
- The recorder refuses to run without `--consent`.

## Sessions
- At least 2 sessions per speaker, on different days (`--session 1`, `--session 2`).
- Quiet room, same microphone for the whole session; the device label goes to `--device`.
- Language: Polish. Distance to the microphone about 20–30 cm.

## Per session
1. Free speech (enrollment material): talk freely, for example describe your day or a picture. Do not read.
2. Random digit strings: read each displayed string aloud as separate digits in Polish. Prompts are seeded per speaker and session, so they differ between sessions.
3. After each clip: Enter keeps it, `r` re-records, `s` skips.

## Command
```
python -m src.data.record --speaker spk01 --session 1 --device laptop_mic --consent
python -m src.data.own          # validate index, write manifest, warn if < 2 sessions
```

## Storage
- Audio: `data/raw/own/<speaker_id>/s<session>/<kind>_<nn>.wav` (gitignored).
- Index: `data/raw/own/recordings.csv` (speaker_id, session, date, kind, prompt, device, path).
