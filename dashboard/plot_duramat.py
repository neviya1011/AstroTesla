import os

import matplotlib.pyplot as plt
import pandas as pd

folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw_data"))

fig, axes = plt.subplots(3, 1, figsize=(10, 9))
for ax, name in zip(axes, ["data - 2.csv", "data - 3.csv", "data - 4.csv"]):
    df = pd.read_csv(os.path.join(folder, name), skiprows=1)
    df["Date"] = pd.to_datetime(df["Date"])
    df["SR"] = pd.to_numeric(df["SR"], errors="coerce")
    rainy = df[pd.to_numeric(df["Rain"], errors="coerce") > 0]
    ax.plot(df["Date"], df["SR"], ".", markersize=3)
    ax.vlines(rainy["Date"], df["SR"].min(), df["SR"].max(), alpha=0.15, color="blue")
    ax.set_title(name)
    ax.set_ylabel("SR (= T)")
plt.tight_layout()
plt.show()