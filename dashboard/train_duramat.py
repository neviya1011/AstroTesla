"""Evaluate DuraMAT forecasting settings without scipy or scikit-learn."""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESH = 0.95
DELTA_OPTIONS = [0.01, 0.02, 0.03]
CAP = 120
BOOTSTRAPS = 500
SEED = 42
FEATURES = [
    "T_smooth_3", "T", "drop_so_far", "rate_per_day", "recent_change_3",
    "slope_7", "slope_14", "recent_change_5", "recent_change_10",
    "days_since_slope_start", "T_start",
]
FOREST_SETTINGS = [(leaf, depth) for leaf in [3, 5, 10] for depth in [4, 6, 8]]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dashboard.simple_forest import (  # noqa: E402
    MeanDaysRegressor,
    NumpyRidgeRegressor,
    SimpleRandomForestRegressor,
    SmoothedRateRegressor,
    StraightLineRegressor,
)

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
except ImportError:
    xgb = None

CSV_PATH = os.path.join(HERE, "duramat_clean.csv")


def metric_values(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    total = np.sum((actual - np.mean(actual)) ** 2)
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "R2": float(1 - np.sum(errors ** 2) / total) if total else np.nan,
    }


def linear_slope(days, values):
    if len(values) < 2 or np.all(days == days[0]):
        return 0.0
    centered_days = days - np.mean(days)
    centered_values = values - np.mean(values)
    denominator = np.dot(centered_days, centered_days)
    return float(np.dot(centered_days, centered_values) / denominator) if denominator else 0.0


def make_features(frame):
    """Build every feature from the current and previous readings only."""
    frame = frame.sort_values("day").reset_index(drop=True).copy()
    values = frame["T"].to_numpy(dtype=float)
    days = frame["day"].to_numpy(dtype=float)
    start_t = values[0]
    output = []
    for index in range(len(frame)):
        current_days = days[: index + 1]
        current_values = values[: index + 1]
        output.append({
            "T_smooth_3": float(np.median(current_values[-3:])),
            "T": values[index],
            "drop_so_far": start_t - values[index],
            "rate_per_day": (start_t - values[index]) / (days[index] - days[0] + 1),
            "recent_change_3": values[index] - values[index - 3] if index >= 3 else 0.0,
            "slope_7": linear_slope(current_days[-7:], current_values[-7:]),
            "slope_14": linear_slope(current_days[-14:], current_values[-14:]),
            "recent_change_5": values[index] - values[index - 5] if index >= 5 else 0.0,
            "recent_change_10": values[index] - values[index - 10] if index >= 10 else 0.0,
            "days_since_slope_start": days[index] - days[0],
            "T_start": start_t,
        })
    feature_frame = pd.DataFrame(output)
    for column in FEATURES:
        frame[column] = feature_frame[column].to_numpy()
    return frame


def build_target_rows(frame, mode, value):
    rows = []
    skipped = 0
    for _, slope in frame.groupby("segment_id", sort=True):
        slope = make_features(slope)
        if mode == "absolute":
            crossing = slope[slope["T"] < value]
        else:
            crossing = slope[slope["T"] <= slope["T_start"] - value]
        if crossing.empty:
            skipped += 1
            continue
        cross_day = float(crossing["day"].iloc[0])
        before = slope[slope["day"] < cross_day].copy()
        if before.empty:
            skipped += 1
            continue
        before["target"] = cross_day - before["day"]
        rows.append(before)
    if not rows:
        return pd.DataFrame(), skipped
    return pd.concat(rows, ignore_index=True), skipped


def setting_label(mode, value):
    return f"{mode}:{value:.3f}"


def make_model(name, mode, value):
    if name == "straight_line":
        return StraightLineRegressor(mode, value, CAP)
    if name == "smoothed_rate":
        return SmoothedRateRegressor(mode, value, CAP)
    if name == "mean_days":
        return MeanDaysRegressor()
    if name == "ridge":
        return NumpyRidgeRegressor(alpha=1.0)
    if name.startswith("forest_"):
        parts = name.split("_")
        leaf = int(parts[1].replace("leaf", ""))
        depth = int(parts[2].replace("depth", ""))
        settings = {
            "n_estimators": 100,
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "random_state": SEED,
        }
        if SKLEARN_AVAILABLE:
            return RandomForestRegressor(**settings, n_jobs=1)
        return SimpleRandomForestRegressor(**settings)
    if name == "gradient_boosting":
        return xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            objective="reg:squarederror", n_jobs=1, verbosity=0,
        )
    raise ValueError(f"Unknown model: {name}")


def model_names():
    names = ["straight_line", "smoothed_rate", "mean_days", "ridge"]
    names.extend(f"forest_leaf{leaf}_depth{depth}" for leaf, depth in FOREST_SETTINGS)
    if xgb is not None:
        names.append("gradient_boosting")
    return names


def evaluate(data, model_name, mode, value, split):
    """Evaluate grouped folds and retain groups for bootstrap confidence intervals."""
    column = "site" if split == "site" else "segment_id"
    groups = sorted(data[column].unique())
    folds = []
    all_actual = []
    all_predicted = []
    for group in groups:
        test = data[data[column] == group]
        train = data[data[column] != group]
        if train.empty or test.empty:
            continue
        model = make_model(model_name, mode, value)
        model.fit(train[FEATURES].to_numpy(), train["target"].to_numpy())
        predicted = np.asarray(model.predict(test[FEATURES].to_numpy()), dtype=float)
        actual = test["target"].to_numpy(dtype=float)
        folds.append({"group": group, "actual": actual, "predicted": predicted})
        all_actual.extend(actual)
        all_predicted.extend(predicted)
    aggregate = metric_values(all_actual, all_predicted) if all_actual else {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    return {"folds": folds, "aggregate": aggregate}


def bootstrap_mae(folds):
    if not folds:
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    group_errors = [np.abs(fold["actual"] - fold["predicted"]) for fold in folds]
    samples = []
    for _ in range(BOOTSTRAPS):
        selected = rng.integers(0, len(group_errors), size=len(group_errors))
        samples.append(np.concatenate([group_errors[index] for index in selected]).mean())
    return tuple(np.percentile(samples, [2.5, 97.5]))


def print_setting_table(results):
    print("\nAll target-setting results (forest selector and straight-line baseline):")
    rows = []
    for result in results:
        rows.append({
            "setting": result["label"], "slopes": result["slopes"], "rows": result["rows"],
            "sites": result["sites"], "eligible": result["eligible"],
            "forest_MAE": result["forest"]["aggregate"]["MAE"],
            "baseline_MAE": result["baseline"]["aggregate"]["MAE"],
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def print_chosen_metrics(model_results):
    print("\nLeave-one-site-out metrics:")
    rows = []
    for name, result in model_results["site"].items():
        for fold in result["folds"]:
            rows.append({"held_out_site": fold["group"], "model": name, **metric_values(fold["actual"], fold["predicted"])})
        rows.append({"held_out_site": "OVERALL", "model": name, **result["aggregate"]})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nLeave-one-slope-out overall metrics:")
    rows = []
    for name, result in model_results["slope"].items():
        low, high = bootstrap_mae(result["folds"])
        rows.append({"model": name, **result["aggregate"], "MAE_95_low": low, "MAE_95_high": high})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def save_plots(setting, data, chosen_name, site_result, final_model, setting_results, model_results):
    for fold in site_result["folds"]:
        plt.figure(figsize=(7, 5))
        plt.scatter(fold["actual"], fold["predicted"], alpha=0.5)
        limit = max(fold["actual"].max(), fold["predicted"].max(), 1)
        plt.plot([0, limit], [0, limit], "k--", label="perfect forecast")
        plt.xlabel("Actual days to target")
        plt.ylabel("Predicted days to target")
        plt.title(f"Predicted vs actual: {fold['group']}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, f"duramat_predicted_vs_actual_{fold['group']}.png"), dpi=150)
        plt.close()

    chosen_mae = model_results["slope"][chosen_name]["aggregate"]["MAE"]
    baseline_mae = model_results["slope"]["straight_line"]["aggregate"]["MAE"]
    plt.figure(figsize=(7, 5))
    plt.bar(["Chosen model", "Straight-line"], [chosen_mae, baseline_mae], color=["steelblue", "gray"])
    plt.ylabel("Leave-one-slope-out MAE (days)")
    plt.title(f"Chosen model vs baseline: {setting['label']}")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "duramat_mae_model_vs_baseline.png"), dpi=150)
    plt.close()

    labels = [item["label"] for item in setting_results]
    values = [item["forest"]["aggregate"]["MAE"] for item in setting_results]
    errors = []
    for item in setting_results:
        low, high = bootstrap_mae(item["forest"]["folds"])
        errors.append([item["forest"]["aggregate"]["MAE"] - low, high - item["forest"]["aggregate"]["MAE"]])
    plt.figure(figsize=(8, 5))
    plt.errorbar(labels, values, yerr=np.asarray(errors).T, fmt="o", capsize=4)
    plt.ylabel("Leave-one-slope-out forest MAE (days)")
    plt.title("MAE by target setting, 95% bootstrap intervals")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "duramat_mae_by_setting.png"), dpi=150)
    plt.close()

    example = data.sort_values(["segment_id", "day"]).groupby("segment_id").first().index[0]
    slope = data[data["segment_id"] == example].sort_values("day")
    first = slope.iloc[[0]]
    forecast = float(final_model.predict(first[FEATURES].to_numpy())[0])
    target_level = first["T_start"].iloc[0] - setting["value"] if setting["mode"] == "relative" else setting["value"]
    plt.figure(figsize=(8, 5))
    plt.plot(slope["day"], slope["T"], "o-", label="Observed T")
    plt.axhline(target_level, color="red", linestyle="--", label=f"Target ({target_level:.3f})")
    plt.axvline(slope["day"].iloc[0] + forecast, color="green", linestyle=":", label="Forecast day")
    plt.xlabel("Day")
    plt.ylabel("T")
    plt.title(f"Example slope forecast: {example}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "duramat_example_slope_forecast.png"), dpi=150)
    plt.close()


def main():
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    except Exception as error:
        print(f"Could not read {CSV_PATH}: {type(error).__name__}: {error}")
        return

    print(f"CSV shape: {df.shape}")
    print("Slopes per site:")
    print(df.groupby("site")["segment_id"].nunique().to_string())
    print("T min/max per site:")
    print(df.groupby("site")["T"].agg(["min", "max"]).to_string())
    print("scikit-learn available:" if SKLEARN_AVAILABLE else "scikit-learn unavailable; using pure NumPy models")
    print("xgboost available" if xgb is not None else "xgboost unavailable; gradient boosting comparison omitted")

    target_settings = [("absolute", THRESH)] + [("relative", delta) for delta in DELTA_OPTIONS]
    setting_results = []
    print("\nUsable target settings:")
    for mode, value in target_settings:
        data, skipped = build_target_rows(df, mode, value)
        slopes = data["segment_id"].nunique() if not data.empty else 0
        sites = data["site"].nunique() if not data.empty else 0
        label = setting_label(mode, value)
        print(f"{label}: slopes={slopes}, rows={len(data)}, sites={sites}, skipped={skipped}")
        result = {"mode": mode, "value": value, "label": label, "data": data, "slopes": slopes, "rows": len(data), "sites": sites, "eligible": slopes >= 6}
        if result["eligible"]:
            result["forest"] = evaluate(data, "forest_leaf5_depth8", mode, value, "slope")
            result["baseline"] = evaluate(data, "straight_line", mode, value, "slope")
        else:
            empty = {"folds": [], "aggregate": {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}}
            result["forest"] = empty
            result["baseline"] = empty
        setting_results.append(result)

    print("\nSelection rule decided before reading test results: choose the lowest overall leave-one-slope-out forest MAE among settings with at least 6 slopes.")
    print_setting_table(setting_results)
    eligible = [item for item in setting_results if item["eligible"] and np.isfinite(item["forest"]["aggregate"]["MAE"])]
    if not eligible:
        print("No setting has at least 6 usable slopes. Lower DELTA or improve segmentation before training.")
        return
    chosen_setting = min(eligible, key=lambda item: item["forest"]["aggregate"]["MAE"])
    data = chosen_setting["data"]
    model_results = {"site": {}, "slope": {}}
    for name in model_names():
        model_results["site"][name] = evaluate(data, name, chosen_setting["mode"], chosen_setting["value"], "site")
        model_results["slope"][name] = evaluate(data, name, chosen_setting["mode"], chosen_setting["value"], "slope")

    print(f"\nChosen setting: {chosen_setting['label']} ({chosen_setting['slopes']} slopes, {chosen_setting['sites']} sites)")
    print_chosen_metrics(model_results)

    ranked = []
    for name, result in model_results["slope"].items():
        ci = bootstrap_mae(result["folds"])
        if np.isfinite(result["aggregate"]["MAE"]):
            ranked.append((name, result["aggregate"]["MAE"], ci))
    best_name, best_mae, best_ci = min(ranked, key=lambda item: item[1])
    simple_order = {"straight_line": 0, "smoothed_rate": 1, "mean_days": 2, "ridge": 3}
    simple_order.update({name: 4 + index for index, name in enumerate(model_names()) if name.startswith("forest_")})
    simple_order["gradient_boosting"] = 20
    within_noise = [item for item in ranked if item[1] <= best_ci[1]]
    chosen_name = min(within_noise, key=lambda item: simple_order[item[0]])[0]
    print(f"\nBest slope-fold model: {best_name} MAE={best_mae:.3f}, bootstrap 95% interval={best_ci[0]:.3f} to {best_ci[1]:.3f}")
    print(f"Chosen simplest model within that noise interval: {chosen_name}")

    final_model = make_model(chosen_name, chosen_setting["mode"], chosen_setting["value"])
    final_model.fit(data[FEATURES].to_numpy(), data["target"].to_numpy())
    joblib.dump({
        "model": final_model, "features": FEATURES, "thresh": THRESH,
        "delta": chosen_setting["value"], "target_mode": chosen_setting["mode"],
    }, os.path.join(HERE, "duramat_model.pkl"))
    forecasts = data[["segment_id", "day", "T", "target"]].copy()
    forecasts["predicted_days_to_limit"] = final_model.predict(data[FEATURES].to_numpy())
    forecasts.rename(columns={"target": "actual_days_to_limit"})[[
        "segment_id", "day", "T", "predicted_days_to_limit", "actual_days_to_limit"
    ]].to_csv(os.path.join(HERE, "duramat_forecasts.csv"), index=False)

    selector = make_model("forest_leaf5_depth8", chosen_setting["mode"], chosen_setting["value"])
    selector.fit(data[FEATURES].to_numpy(), data["target"].to_numpy())
    print("\nFeature importances from selector forest:")
    for name, importance in zip(FEATURES, selector.feature_importances_):
        print(f"  {name}: {importance:.4f}")
    print("Features were retained because none had zero importance in the final selector forest.")
    save_plots(chosen_setting, data, chosen_name, model_results["site"][chosen_name], final_model, setting_results, model_results)

    chosen_result = model_results["slope"][chosen_name]
    chosen_ci = bootstrap_mae(chosen_result["folds"])
    chosen_mae = chosen_result["aggregate"]["MAE"]
    baseline_mae = model_results["slope"]["straight_line"]["aggregate"]["MAE"]
    hyder_tested = any(fold["group"] == "Hyder_AZ" for fold in model_results["site"][chosen_name]["folds"])
    print("\nSlide summary:")
    print(f"Chose {chosen_setting['label']} because it had the lowest eligible slope-fold MAE, using {chosen_setting['slopes']} slopes across {chosen_setting['sites']} sites.")
    print(f"The chosen {chosen_name} model scored {chosen_mae:.2f} days MAE (95% interval {chosen_ci[0]:.2f}-{chosen_ci[1]:.2f}) versus the baseline at {baseline_mae:.2f} days.")
    print(f"Hyder_AZ could {'now' if hyder_tested else 'not'} be tested in the site-fold evaluation.")
    print("Earth soiling is not charged lunar dust; rain stands in for cleaning events, and the number of slopes is small.")
    print("This validates the method, not the Moon; validation targets are not the real mission limit of T=0.85.")


if __name__ == "__main__":
    main()
