"""
Makes FAKE rig data so you can test CSV replay before the real rig exists.
Run from the repo root (the AstroTesla folder):
    python dashboard/make_fake_csv.py
Writes data/fake_run.csv with columns a0 (dusty cell) and a1 (reference cell).
Do NOT present this as real data. The real recording comes from the hardware teammate.
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
n = 300
ref = 600 + rng.normal(0, 3, n)                  # reference cell, roughly constant
true_T = np.linspace(1.0, 0.78, n)               # dust slowly builds up
dusty = ref * true_T + rng.normal(0, 4, n)       # dusty cell
glitch = rng.random(n) < 0.02                    # a few random sensor glitches
dusty[glitch] = rng.uniform(300, 650, glitch.sum())

os.makedirs("data", exist_ok=True)
pd.DataFrame({"a0": dusty.round(1), "a1": ref.round(1)}).to_csv("data/fake_run.csv", index=False)
print("Saved data/fake_run.csv")