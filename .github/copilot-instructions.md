## Deep_Learning_Proyect — Copilot instructions

This file gives actionable, project-specific guidance for AI coding agents working on this repo.

High-level overview
- Purpose: a small pipeline to load MP4 videos per-class from `dataset/`, preprocess frames, train a MobileNetV2-based CNN (per-frame) wrapped by an LSTM to classify sequences, and provide realtime inference via `test.py`.
- Key components: `dataset_loader.py` (video read & label encode), `preparacion_datos.py` (split by person, label one-hot), `modelo.py` (MobileNetV2 -> TimeDistributed -> LSTM), `main.py` (orchestrates training), `test.py` (webcam inference), `dataset_analyzer.py` & `renombrar.py` (helpers).

Essential developer workflows (Windows PowerShell)
- Create venv and install deps:
  ```powershell
  python -m venv venv
  venv\Scripts\Activate.ps1  # or venv\Scripts\activate
  pip install -r requirements.txt
  ```
- Train: run `main.py`. It expects `dataset/` to contain one directory per class with `.mp4` files. `main.py` writes `modelo_cnn2d_lstm_30_64.keras` and `label_encoder.pkl`.
- Real-time test: run `test.py` after training. It loads `modelo_cnn2d_lstm_30_64.keras` and `label_encoder.pkl` from repo root.

Project-specific conventions and gotchas
- Filename / person extraction: `preparacion_datos.dividir_por_persona` uses `filename.split("_")[1]` to extract the person token. Ensure video filenames follow the underscore pattern (e.g. `VID_20250101_MATEO_1.mp4` or the `renombrar.py` output like `HOLA_MATEO_1.mp4`). If filenames differ, update that split logic.
- Frames & resolution mismatches: `dataset_loader.cargar_dataset` default is `frames_totales=60, video_size=96` but `main.py` calls `cargar_dataset(..., frames_totales=30, video_size=64)`. Check both `main.py`, `dataset_analyzer.py` and `dataset_loader.py` if you change frame count or size — `dataset_analyzer.py` also expects 60/96 by default.
- Label encoder persistence: `cargar_dataset` writes `label_encoder.pkl`. Other scripts (e.g. `test.py`) load that file — don’t change its name unless updating all references.
- Model files: `main.py` saves model as `modelo_cnn2d_lstm_30_64.keras` and `test.py` expects that exact filename. Keep names in sync or make the path configurable.

Architecture / integration notes
- Input shape: model expects (frames, height, width, 3). `main.py` sets `input_shape = (frames_totales, video_size, video_size, 3)`.
- Backbone: MobileNetV2 is used as frozen feature extractor inside a `TimeDistributed` wrapper; LSTM follows. MobileNetV2 is created with `weights='imagenet'` and `trainable=False` in `modelo.py`.
- Training: `entrenar_modelo` defaults to `epochs=4, batch_size=8` in `modelo.py` — tune for real experiments.

Quick examples for common edits
- Change frame size used across project: update `frames_totales` and `video_size` in `main.py` and adjust `cargar_dataset` defaults if you want a global change. Also update `dataset_analyzer.py` expected values.
- Make model filename configurable: replace hard-coded strings in `main.py` and `test.py` with a single constant or CLI argument.

Files to inspect for examples
- Data load & encoder: `dataset_loader.py`
- Dataset split & label preparation: `preparacion_datos.py`
- Model definition & training: `modelo.py`
- Training orchestration: `main.py`
- Real-time inference demo: `test.py`
- Helpers for dataset sanity & renaming: `dataset_analyzer.py`, `renombrar.py`

If something is unclear or you need more examples (e.g., converting hard-coded names to CLI flags, adding unit tests, or standardizing frame-size constants), ask and I will update this file with concrete diffs.
