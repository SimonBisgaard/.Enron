from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _infer_meteo_columns(test_df: pd.DataFrame) -> list[str]:
    non_meteo = {
        "id",
        "market",
        "delivery_start",
        "delivery_end",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
    }
    return [c for c in test_df.columns if c not in non_meteo]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new submission from a baseline submission by adding "
            "N * std(target_pred) per (market,hour,dow) on rows with missing meteo."
        )
    )
    parser.add_argument("--baseline-submission", default="submissio.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--std-multiplier", type=float, default=2.0)
    parser.add_argument("--out-path", default="submission_missing_plus2std.csv")
    args = parser.parse_args()

    baseline_path = Path(args.baseline_submission)
    test_path = Path(args.test_path)
    out_path = Path(args.out_path)

    sub = pd.read_csv(baseline_path)
    test = pd.read_csv(test_path)

    if not {"id", "target"}.issubset(sub.columns):
        raise ValueError("Baseline submission must have columns: id,target")
    if not {"id", "market", "delivery_start"}.issubset(test.columns):
        raise ValueError("Test file must have columns: id,market,delivery_start")

    sub = sub.copy()
    test = test.copy()
    sub["id"] = pd.to_numeric(sub["id"], errors="coerce")
    sub["target"] = pd.to_numeric(sub["target"], errors="coerce")
    test["id"] = pd.to_numeric(test["id"], errors="coerce")
    sub = sub.dropna(subset=["id", "target"]).copy()
    test = test.dropna(subset=["id"]).copy()
    sub["id"] = sub["id"].astype(int)
    test["id"] = test["id"].astype(int)

    merged = (
        test.merge(sub[["id", "target"]], on="id", how="left")
        .sort_values("id")
        .copy()
    )
    if merged["target"].isna().any():
        missing_ids = merged.loc[merged["target"].isna(), "id"].head(10).tolist()
        raise ValueError(f"Missing predictions for some test IDs, sample: {missing_ids}")

    start = pd.to_datetime(merged["delivery_start"], errors="coerce")
    merged["hour"] = start.dt.hour
    merged["dow"] = start.dt.dayofweek

    meteo_cols = _infer_meteo_columns(test)
    merged["meteo_missing_any"] = merged[meteo_cols].isna().any(axis=1)

    # Primary std at (market, hour, dow), with robust fallbacks for tiny groups.
    std_mhd = (
        merged.groupby(["market", "hour", "dow"], dropna=False)["target"]
        .std()
        .rename("std_mhd")
        .reset_index()
    )
    std_mh = (
        merged.groupby(["market", "hour"], dropna=False)["target"]
        .std()
        .rename("std_mh")
        .reset_index()
    )
    std_m = (
        merged.groupby(["market"], dropna=False)["target"]
        .std()
        .rename("std_m")
        .reset_index()
    )
    global_std = float(merged["target"].std())

    merged = merged.merge(std_mhd, on=["market", "hour", "dow"], how="left")
    merged = merged.merge(std_mh, on=["market", "hour"], how="left")
    merged = merged.merge(std_m, on=["market"], how="left")
    merged["std_used"] = (
        merged["std_mhd"]
        .fillna(merged["std_mh"])
        .fillna(merged["std_m"])
        .fillna(global_std)
    )

    out = sub.copy()
    adjust = merged["meteo_missing_any"].to_numpy()
    delta = float(args.std_multiplier) * merged["std_used"].to_numpy()
    out_map = pd.Series(delta, index=merged["id"])
    out.loc[out["id"].isin(merged.loc[adjust, "id"]), "target"] = (
        out.loc[out["id"].isin(merged.loc[adjust, "id"]), "target"].to_numpy()
        + out.loc[out["id"].isin(merged.loc[adjust, "id"]), "id"].map(out_map).to_numpy()
    )

    out = out.sort_values("id")
    out.to_csv(out_path, index=False)

    n_missing = int(merged["meteo_missing_any"].sum())
    n_total = len(merged)
    print(f"Saved: {out_path}")
    print(f"Adjusted rows (meteo missing any): {n_missing}/{n_total} ({n_missing / n_total:.4%})")
    print(f"Std multiplier: {args.std_multiplier}")
    print(
        "Mean added delta on adjusted rows: "
        f"{(float(delta[adjust].mean()) if n_missing else 0.0):.6f}"
    )


if __name__ == "__main__":
    main()
