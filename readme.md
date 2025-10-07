# AI DJ with Model

AI DJ with Model is a small toolkit for analysing music files, training a song recommender, and exploring the results in a Streamlit-powered virtual DJ interface inspired by the DDJ-FLX4 controller. The workflow starts by extracting rich audio features from your library, continues with model training, and ends with an interactive browser app that can recommend follow-up tracks and render simple transitions.

## Highlights
- End-to-end pipeline: audio feature extraction, model training, and live interface.
- Rich musical descriptors (rhythm, tonal, dynamics, mood, instrumentation, embeddings).
- Optional power-ups when Essentia, madmom, pyloudnorm, OpenL3, or Spleeter are installed.
- Recommender bundle saved as `models/recommender.joblib` for reuse in the UI.
- Streamlit front end with two compact decks, central mix controls, and instant mix preview.

## Repository Layout
- `feauture_extraction.py` – CLI and library for generating detailed JSON reports from audio files.
- `train_recommender.py` – trains the scaler, nearest-neighbor model, and clustering on saved reports.
- `dj_interface.py` – Streamlit application that loads the bundle plus analysis reports.
- `models/recommender.joblib` – example trained bundle (regenerate by running the training script).
- `tests/` – unit tests for the feature extraction helpers.
- `Music_Data/` (optional) – expected location for your music library if you use defaults.

## Requirements
- Python 3.9+
- Required packages: `numpy`, `librosa`, `pandas`, `joblib`, `scikit-learn`, `streamlit`, `soundfile`, `pydub`
- Optional packages (enable extra features): `pyloudnorm`, `openl3`, `essentia`, `madmom`, `spleeter`
- FFmpeg installed on your PATH (recommended for pydub MP3 export).

You can install the basics with:

```bash
pip install numpy librosa pandas joblib scikit-learn streamlit soundfile pydub mutagen
```

Add any of the optional libraries above if you plan to use their specific capabilities.

## Quick Start
1. **Clone and enter the project**
   ```bash
   git clone <repo-url>
   cd AI_DJ_with_model
   ```
2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install dependencies** using the `pip install` command shown above (add optional packages as needed).
4. **Analyse a track** to produce a JSON report (saved next to the audio by default):
   ```bash
   python feauture_extraction.py path/to/song.mp3
   ```
   Add `--output -` if you want the JSON printed to the terminal instead of written to disk.
5. **Train the recommender** on all JSON reports:
   ```bash
   python train_recommender.py --reports reports --model-out models/recommender.joblib
   ```
6. **Launch the Streamlit interface**:
   ```bash
   streamlit run dj_interface.py
   ```
   The app will open in your browser, allowing you to load two decks, configure the mix, and generate a transition.

## Detailed Usage

### 1. Audio Feature Extraction
`feauture_extraction.py` can process individual files or entire folders. Useful flags:

- `--output <path>`: write the analysis to a specific location. Omit the flag to save alongside the audio, or pass `--output -` to print to stdout.
- `--batch-folder <folder>`: analyse every file matching `--batch-pattern` (defaults to `*.mp3`).
- `--batch-pattern "**/*.mp3"`: recursively scan subfolders when running in batch mode.
- `--enable-stems`: run Spleeter stem separation (requires Spleeter and FFmpeg).
- `--embeddings`: compute OpenL3 embeddings if the library is installed.
- `--disable-essentia` or `--disable-madmom`: skip those heavy optional dependencies even if available.
- `--overwrite`: recompute analysis even when the JSON already exists next to the file.

#### Single-file extraction

Generate or refresh analysis for one track (JSON saved alongside the audio):

```bash
python feauture_extraction.py /mnt/nvme/Genie_lib/Music_Data/song.mp3
```

To force a refresh, append `--overwrite`. To keep the output in the console, use `--output -`.

Example batch run that mirrors the defaults used by the Streamlit app:

```bash
python feauture_extraction.py --batch-folder /mnt/nvme/Genie_lib --batch-pattern "**/*.mp3"
```

JSON files will be written next to their source audio. Provide `--batch-out reports` to collect them elsewhere, and add `--overwrite` to refresh existing analyses.
- The quoted `"**/*.mp3"` pattern recursively scans every subfolder beneath the chosen library root so you can point to a top-level music directory.

Each JSON file records metadata, rhythm, tonal, dynamics, spatial, mood, instrumentation, and optional embeddings/stem statistics. The tests in `tests/test_feauture_extraction.py` illustrate the expected structure.

### 2. Training the Recommender
`train_recommender.py` reads every JSON report, flattens the relevant metrics, and trains a bundle with:

- `StandardScaler` for feature normalisation.
- `NearestNeighbors` (cosine distance) for similar-track lookup.
- `KMeans` clustering for organising the library into groups.

Key flags:
- `--reports`: folder containing the JSON files (default `reports`).
- `--model-out`: where to save the joblib bundle (default `models/recommender.joblib`).
- `--neighbors`: how many nearest neighbours to keep.
- `--test-size`: fraction reserved for evaluation metrics (set to `0` or `1` to skip splitting).

Run the script whenever you add new analysis files so the Streamlit app can pick up the updated model.

### 3. Streamlit DJ Interface
Start the UI with `streamlit run dj_interface.py`. Important behaviours:

- The app loads the recommender bundle (default `models/recommender.joblib`) and caches it for quicker reloads.
- It scans the `reports` directory for analysis JSON files and filters them to match `--allowed-folder` values. If no reports are found there, it will search every allowed folder for JSONs stored alongside your audio.
- Deck A and Deck B sit on the left/right columns with quick-pick dropdowns and uploaders, while the mix panel stays centered for a clean single-screen workflow.
- Tracks without a matching JSON analysis are ignored, so make sure you have run the feature extractor on any music you want available in the interface.
- Deck search boxes match on file stem or path; the app prevents using the same track on both decks.
- When `auto_recommend` is enabled (controlled via `instructions.json`), the UI lists nearby tracks from the training metadata.
- The mix renderer prefers pydub + FFmpeg. If unavailable, it falls back to a numpy/librosa crossfade implementation exporting WAV.

#### Instructions JSON
`instructions.json` lets you preconfigure default searches, mix style, crossfade length, effects, and allowed library folders. Update the file and refresh Streamlit to load the new defaults.

## Running Tests
Unit tests cover the most critical feature-extraction utilities. Run them with:

```bash
python -m pytest tests
```

## Tips
- Keep audio and report folders organised; large batch runs can create hundreds of JSON files.
- Optional libraries significantly improve analysis quality (e.g., Essentia for advanced tonal descriptors, madmom for downbeat tracking).
- Re-run the training script whenever you add new reports so the model stays in sync.
- Ensure FFmpeg is installed if you want MP3 export from the Streamlit mix renderer.

Enjoy exploring your music library with AI-assisted analysis and quick virtual mixes!
