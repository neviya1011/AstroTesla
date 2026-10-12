"""
backend.py: the ONE place where the dashboard talks to teammates' code.

For each function the dashboard needs, we try to import the teammate's real
version. If it is not in the repo yet (or fails to import), we use a simple
stand-in so the dashboard always runs.

If a teammate's function name or arguments differ, fix it HERE only.
app.py never needs to change.
"""
import sys
import pathlib

# let us import sim/, processing/ ... which sit next to the dashboard/ folder
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib

USING_REAL = {}   # shows in the sidebar which parts are real vs stand-in

# ---- 1. simulator (dataset teammate): simulate_day(dust_rate, noise) -> DataFrame[hour, true_T, raw_T]
try:
    from sim.simulator import simulate_day
    USING_REAL["simulator"] = True
except ImportError:
    USING_REAL["simulator"] = False

    def simulate_day(dust_rate=1.0, noise=0.02, hours=354, seed=42):
        rng = np.random.default_rng(seed)
        h = np.arange(hours)
        true_T = np.exp(-0.0007 * dust_rate * h)
        raw_T = true_T + rng.normal(0, noise, hours)
        glitch = rng.random(hours) < 0.01
        raw_T[glitch] = rng.uniform(0.5, 1.1, glitch.sum())
        return pd.DataFrame({"hour": h, "true_T": true_T, "raw_T": raw_T})

# ---- 2. filter (filter teammate): filter_series(raw_T_series) -> filtered T series
try:
    from processing.filters import filter_series
    USING_REAL["filter"] = True
except ImportError:
    USING_REAL["filter"] = False

    def filter_series(raw):
        return raw.rolling(6, min_periods=1).median()

# ---- 3. predictor: use the trained DuraMAT artifact, then the teammate model, then a stand-in
MODEL_PATH = pathlib.Path(__file__).resolve().parent / "duramat_model.pkl"
try:
    _DURAMAT_ARTIFACT = joblib.load(MODEL_PATH)
    _DURAMAT_MODEL = _DURAMAT_ARTIFACT["model"]
    _DURAMAT_FEATURES = _DURAMAT_ARTIFACT["features"]
    if _DURAMAT_ARTIFACT.get("target_mode") == "relative":
        MODEL_TARGET_TEXT = f"model target (drop {float(_DURAMAT_ARTIFACT['delta']):.2f})"
    else:
        MODEL_TARGET_TEXT = f"model target (T < {float(_DURAMAT_ARTIFACT.get('thresh', 0.95)):.2f})"
    USING_REAL["duramat_model"] = True
except Exception:
    _DURAMAT_ARTIFACT = None
    _DURAMAT_MODEL = None
    _DURAMAT_FEATURES = []
    MODEL_TARGET_TEXT = "limit"
    USING_REAL["duramat_model"] = False


def _past_only_model_features(t):
    """Match train_duramat.py using one reading as one demo hour."""
    values = pd.Series(t, dtype=float).reset_index(drop=True).to_numpy()
    days = np.arange(len(values), dtype=float)
    start_t = values[0]
    rows = []
    for index in range(len(values)):
        current_days = days[: index + 1]
        current_values = values[: index + 1]
        def slope(window):
            x = current_days[-window:]
            y = current_values[-window:]
            if len(y) < 2:
                return 0.0
            return float(np.polyfit(x, y, 1)[0])
        rows.append({
            "T_smooth_3": float(np.median(current_values[-3:])),
            "T": values[index],
            "day": days[index],
            "drop_so_far": start_t - values[index],
            "rate_per_day": (start_t - values[index]) / (days[index] + 1),
            "recent_change_3": values[index] - values[index - 3] if index >= 3 else 0.0,
            "slope_7": slope(7),
            "slope_14": slope(14),
            "recent_change_5": values[index] - values[index - 5] if index >= 5 else 0.0,
            "recent_change_10": values[index] - values[index - 10] if index >= 10 else 0.0,
            "recent_change": values[index] - values[index - 5] if index >= 5 else 0.0,
            "days_since_slope_start": days[index],
            "T_start": start_t,
        })
    return pd.DataFrame(rows)


if _DURAMAT_MODEL is not None:
    USING_REAL["predictor"] = True

    def predict_hours(t):
        """Predict remaining demo hours with the saved DuraMAT model."""
        features = _past_only_model_features(t)
        prediction = _DURAMAT_MODEL.predict(features[_DURAMAT_FEATURES].to_numpy())[-1]
        return float(np.clip(prediction, 0, 120))
else:
    try:
        from sim.predictor import predict_hours
        USING_REAL["predictor"] = True
    except ImportError:
        USING_REAL["predictor"] = False

        def predict_hours(t):
            """Stand-in: straight-line extrapolation of the last 20 points."""
            y = t.iloc[-20:].to_numpy()
            slope = np.polyfit(np.arange(len(y)), y, 1)[0]
            if slope >= 0:
                return 72.0
            return float(np.clip((y[-1] - 0.85) / -slope, 0, 72))

# ---- 4. scheduler (filter teammate): decide(T_now, hours_left) -> "OK" | "WARN" | "CLEAN_NOW"
try:
    from processing.scheduler import decide
    USING_REAL["scheduler"] = True
except ImportError:
    USING_REAL["scheduler"] = False

    def decide(t_now, hours_left):
        if t_now < 0.85:
            return "CLEAN_NOW"
        return "WARN" if hours_left < 12 else "OK"