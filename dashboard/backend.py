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

# ---- 3. predictor (random forest teammate): predict_hours(filtered_T_series) -> hours until T < 0.85
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