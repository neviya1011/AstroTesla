# AstroTesla / DustSentinel

DustSentinel is a Space Instrumentation hackathon project built to monitor dust accumulation on a solar panel and decide when cleaning is needed. The app combines a trained DuraMAT-style forecasting model with live or replayed panel telemetry and presents a clean decision dashboard.

## Project goal

The dashboard answers one practical question:

> Is the panel still healthy enough to keep operating, or is it now dirty enough that a cleaning action is required?

The app tracks a normalized panel health metric, `T`, where higher values indicate cleaner conditions. It then forecasts whether `T` is approaching the maintenance threshold and shows a status such as `OK`, `WARN`, or `CLEAN_NOW`.

## Repo structure

```text
AstroTesla/
├── README.md
├── requirements.txt
├── data/
│   └── fake_run.csv
├── dashboard/
│   ├── app.py
│   ├── backend.py
│   ├── clean_duramat.py
│   ├── duramat_clean.csv
│   ├── duramat_forecasts.csv
│   ├── duramat_model.pkl
│   ├── fake_csv.py
│   ├── inspect_duramat.py
│   ├── plot_duramat.py
│   ├── simple_forest.py
│   └── train_duramat.py
└── figures/
```

## How the app works

1. A raw telemetry series is loaded from either:
   - CSV replay (`data/fake_run.csv`), or
   - a connected Arduino/USB serial device.
2. The series is smoothed and converted into a consistent `T` signal.
3. The dashboard loads the trained regression model from `dashboard/duramat_model.pkl`.
4. The model estimates how long the panel can operate before it reaches the configured threshold.
5. The status logic converts that forecast into an operational decision.

## Install and run

From the project root (`AstroTesla`):

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Then run the dashboard:

```bash
streamlit run dashboard/app.py
```

The app is designed to be launched from the repo root so imports resolve correctly.

## Dashboard modes

### 1) CSV Replay

The dashboard can replay a recorded dataset from `data/fake_run.csv`.

Use this mode to validate the pipeline, model outputs, and dashboard layout without hardware.

### 2) Live Arduino

The live mode opens a USB serial port automatically when an Arduino-like device is available. It supports both:

- comma-separated values such as `A0,A1`
- direct telemetry lines such as `T=86.1% state=1`

The parser extracts the normalized panel value and updates the dashboard in near real time.

## Model training

The training pipeline is in `dashboard/train_duramat.py`.

This project intentionally avoids `scikit-learn` and `scipy` when they are unavailable in the environment. Instead, it uses a NumPy-based forecasting stack with a pure Python fallback model and a `simple_forest.py` implementation.

To retrain the model:

```bash
python dashboard/train_duramat.py
```

This produces artifacts such as:

- `dashboard/duramat_clean.csv`
- `dashboard/duramat_forecasts.csv`
- `dashboard/duramat_model.pkl`

## Data flow summary

```text
Sensor / CSV data
    ↓
Dashboard app
    ↓
backend.py
    ↓
filter + model inference
    ↓
status decision
    ↓
UI cards + forecast chart
```

## Notes

- The app uses a robust `backend.py` compatibility layer so the dashboard keeps working even when teammate modules are not present yet.
- The default decision threshold is effectively centered on a target `T` value near `0.85`.
- The app is intentionally simple and demo-friendly for a hackathon environment.

## Team and context

This project was developed for the SedHacks 2026 Space Instrumentation track by the AstroTesla team. It focuses on dust mitigation and maintenance planning for lunar or orbital solar panel systems.

## License

This project is intended for educational and hackathon use.
