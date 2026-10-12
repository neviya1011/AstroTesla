import glob
import os

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "raw_data"))
print("Looking in:", folder)

for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
    print("=" * 80)
    print("FILE:", os.path.basename(path))
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    print("ROWS, COLUMNS:", df.shape)
    print("COLUMNS AND TYPES:")
    print(df.dtypes)
    print("FIRST 3 ROWS:")
    print(df.head(3))
    print("LAST 2 ROWS:")
    print(df.tail(2))
    print("MISSING VALUES PER COLUMN:")
    print(df.isna().sum())