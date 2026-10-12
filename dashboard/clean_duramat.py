"""Turn the three raw DuraMAT files into slope segments without inventing data."""

import os

import pandas as pd

# The default output uses the most inclusive documented setting.
JUMP_MIN = 0.005
MIN_POINTS = 5
REQUIRE_RAIN = False

JUMP_OPTIONS = [0.005, 0.01]
MIN_POINT_OPTIONS = [5, 8, 10]
RAIN_OPTIONS = [True, False]

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.abspath(os.path.join(HERE, "..", "..", "raw_data"))
OUT = os.path.join(HERE, "duramat_clean.csv")

FILES = {
    "data - 2.csv": "Calipatria_CA",
    "data - 3.csv": "Mendota_CA",
    "data - 4.csv": "Hyder_AZ",
}


def read_raw_files():
    pieces = []
    for filename, site in FILES.items():
        path = os.path.join(RAW, filename)
        try:
            frame = pd.read_csv(path, skiprows=1)
        except FileNotFoundError:
            print(f"Missing raw file: {path}")
            continue
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["SR"] = pd.to_numeric(frame["SR"], errors="coerce")
        frame["Rain"] = pd.to_numeric(frame["Rain"], errors="coerce").fillna(0)
        frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        frame["RainCum"] = frame["Rain"].cumsum()
        pieces.append((site, frame.dropna(subset=["SR"]).reset_index(drop=True)))
    return pieces


def make_clean(pieces, jump_min, min_points, require_rain):
    output = []
    for site, frame in pieces:
        segment_number = 0
        segment_ids = []
        for index in range(len(frame)):
            starts_after_jump = False
            if index > 0:
                jump = frame.loc[index, "SR"] - frame.loc[index - 1, "SR"]
                starts_after_jump = jump >= jump_min
                if require_rain:
                    starts_after_jump = starts_after_jump and (
                        frame.loc[index, "RainCum"] - frame.loc[index - 1, "RainCum"] > 0
                    )
            if starts_after_jump:
                segment_number += 1
            segment_ids.append(segment_number)

        frame = frame.copy()
        frame["seg"] = segment_ids
        for number, slope in frame.groupby("seg", sort=True):
            if len(slope) < min_points:
                continue
            slope = slope.copy()
            slope["site"] = site
            slope["segment_id"] = f"{site}_{number}"
            slope["day"] = (slope["Date"] - slope["Date"].iloc[0]).dt.days
            slope["T"] = slope["SR"].clip(upper=1.0)
            output.append(slope[["site", "segment_id", "Date", "day", "T"]])
    if not output:
        return pd.DataFrame(columns=["site", "segment_id", "Date", "day", "T"])
    return pd.concat(output, ignore_index=True)


def print_setting_summary(clean, jump_min, min_points, require_rain):
    label = f"jump={jump_min:.3f}, min_points={min_points}, rain_required={require_rain}"
    if clean.empty:
        print(f"{label}: slopes=0, rows=0")
        return
    counts = clean.groupby("site")["segment_id"].nunique()
    rows = clean.groupby("site").size()
    details = "; ".join(
        f"{site}: {counts.get(site, 0)} slopes/{rows.get(site, 0)} rows"
        for site in FILES.values()
    )
    print(f"{label}: total={clean['segment_id'].nunique()} slopes/{len(clean)} rows; {details}")


def main():
    pieces = read_raw_files()
    if len(pieces) != len(FILES):
        print("One or more raw files are missing; no output was written.")
        return

    print("Segmentation settings (no missing SR values are filled or interpolated):")
    for jump_min in JUMP_OPTIONS:
        for min_points in MIN_POINT_OPTIONS:
            for require_rain in RAIN_OPTIONS:
                clean = make_clean(pieces, jump_min, min_points, require_rain)
                print_setting_summary(clean, jump_min, min_points, require_rain)

    clean = make_clean(pieces, JUMP_MIN, MIN_POINTS, REQUIRE_RAIN)
    if clean.empty:
        print("No slopes found. Lower JUMP_MIN or MIN_POINTS and run again.")
        return

    clean.to_csv(OUT, index=False)
    print(f"\nSaved default output: {OUT}")
    print(f"Default rows: {len(clean)} | slopes: {clean['segment_id'].nunique()}")
    print(clean.groupby("site")["segment_id"].nunique().to_string())


if __name__ == "__main__":
    main()
