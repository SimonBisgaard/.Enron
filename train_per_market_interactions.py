from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def _to_datetime_col(df: pd.DataFrame, col: str) -> pd.Series:
    out = pd.to_datetime(df[col], errors="coerce")
    if out.isna().any():
        raise ValueError(f"Failed parsing some timestamps in column '{col}'")
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    start = _to_datetime_col(out, "delivery_start")
    out["hour"] = start.dt.hour
    out["dow"] = start.dt.dayofweek
    out["month"] = start.dt.month
    out["weekofyear"] = start.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    # Cyclical encodings help trees with periodic patterns.
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dow"] / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    out["is_morning_peak"] = out["hour"].isin([7, 8, 9]).astype(int)
    out["is_evening_peak"] = out["hour"].isin([17, 18, 19, 20]).astype(int)
    out["is_winter"] = out["month"].isin([11, 12, 1, 2, 3]).astype(int)
    return out


def add_forecast_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["residual_load"] = out["load_forecast"] - out["wind_forecast"] - out["solar_forecast"]
    out["is_18_19"] = out["hour"].isin([18, 19]).astype(int)
    out["is_peak_17_20"] = out["hour"].isin([17, 18, 19, 20]).astype(int)
    out["is_night"] = (out["solar_forecast"] <= 0.0).astype(int)
    out["stress_ratio"] = out["load_forecast"] / (out["wind_forecast"] + 1000.0)
    out["stress_ratio_1819"] = out["stress_ratio"] * out["is_18_19"]
    out["residual_load_1819"] = out["residual_load"] * out["is_18_19"]
    out["net_load_share"] = out["residual_load"] / (out["load_forecast"] + 1e-6)
    out["wind_share"] = out["wind_forecast"] / (out["load_forecast"] + 1e-6)
    out["solar_share"] = out["solar_forecast"] / (out["load_forecast"] + 1e-6)
    out["stress_ratio_sq"] = out["stress_ratio"] ** 2
    out["residual_load_sq"] = out["residual_load"] ** 2
    return out


def add_forecast_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["market", "delivery_start"]).reset_index(drop=True)
    lag_cols = ["load_forecast", "wind_forecast", "solar_forecast", "residual_load"]
    lag_steps = [1, 2, 6, 24, 48, 168]
    roll_windows = [6, 24, 72]

    for c in lag_cols:
        g = out.groupby("market")[c]
        for k in lag_steps:
            out[f"{c}_lag_{k}"] = g.shift(k)
        out[f"{c}_diff_1"] = out[c] - g.shift(1)
        out[f"{c}_diff_6"] = out[c] - g.shift(6)
        out[f"{c}_diff_24"] = out[c] - g.shift(24)
        shifted = g.shift(1)
        for w in roll_windows:
            out[f"{c}_roll_mean_{w}"] = shifted.groupby(out["market"]).transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            out[f"{c}_roll_std_{w}"] = shifted.groupby(out["market"]).transform(
                lambda s: s.rolling(w, min_periods=2).std()
            )

    return out


def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"wind_gust_speed_10m", "wind_speed_10m"}.issubset(out.columns):
        out["gustiness_10m"] = out["wind_gust_speed_10m"] - out["wind_speed_10m"]
        out["gust_ratio_10m"] = out["wind_gust_speed_10m"] / (out["wind_speed_10m"] + 1e-6)
    if {"wind_speed_80m", "wind_speed_10m"}.issubset(out.columns):
        out["ws_ratio_80_10"] = out["wind_speed_80m"] / (out["wind_speed_10m"] + 1e-6)
    if "wind_direction_80m" in out.columns:
        out["wind_dir_sin"] = np.sin(np.deg2rad(out["wind_direction_80m"]))
        out["wind_dir_cos"] = np.cos(np.deg2rad(out["wind_direction_80m"]))

    if {"direct_normal_irradiance", "global_horizontal_irradiance"}.issubset(out.columns):
        out["clear_sky_proxy"] = out["direct_normal_irradiance"] / (
            out["global_horizontal_irradiance"] + 1e-6
        )
    if {"cloud_cover_low", "cloud_cover_total"}.issubset(out.columns):
        out["cloud_low_share"] = out["cloud_cover_low"] / (out["cloud_cover_total"] + 1e-6)

    if {"air_temperature_2m", "dew_point_temperature_2m"}.issubset(out.columns):
        out["temp_dew_spread"] = out["air_temperature_2m"] - out["dew_point_temperature_2m"]
    if {"air_temperature_2m", "apparent_temperature_2m"}.issubset(out.columns):
        out["temp_apparent_gap"] = out["apparent_temperature_2m"] - out["air_temperature_2m"]

    if {"wind_speed_80m", "residual_load"}.issubset(out.columns):
        out["wind80_x_residual_load"] = out["wind_speed_80m"] * out["residual_load"]
    if {"cloud_cover_total", "solar_forecast"}.issubset(out.columns):
        out["cloud_x_solar"] = out["cloud_cover_total"] * out["solar_forecast"]

    return out


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    core_cols = {
        "id",
        "target",
        "market",
        "delivery_start",
        "delivery_end",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
        "_is_train",
    }
    meteo_cols = [c for c in out.columns if c not in core_cols]
    if meteo_cols:
        out["meteo_missing_count"] = out[meteo_cols].isna().sum(axis=1)
        out["meteo_missing_any"] = (out["meteo_missing_count"] > 0).astype(int)
    else:
        out["meteo_missing_count"] = 0
        out["meteo_missing_any"] = 0
    return out


def add_market_categorical_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour_x_market"] = out["hour"].astype(str) + "_" + out["market"].astype(str)
    out["dow_x_market"] = out["dow"].astype(str) + "_" + out["market"].astype(str)
    out["month_x_market"] = out["month"].astype(str) + "_" + out["market"].astype(str)
    return out


def add_cross_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-market interactions at the same timestamp.
    For each row, we compute market-vs-global snapshot differences.
    """
    out = df.copy()
    cross_cols = [
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
        "residual_load",
        "wind_speed_80m",
        "air_temperature_2m",
        "cloud_cover_total",
        "precipitation_amount",
    ]
    present = [c for c in cross_cols if c in out.columns]
    if not present:
        return out

    grouped = out.groupby("delivery_start", dropna=False)
    for c in present:
        mean_col = f"{c}_xmk_mean"
        std_col = f"{c}_xmk_std"
        out[mean_col] = grouped[c].transform("mean")
        out[std_col] = grouped[c].transform("std")
        out[f"{c}_xmk_diff"] = out[c] - out[mean_col]
        out[f"{c}_xmk_z"] = out[f"{c}_xmk_diff"] / (out[std_col] + 1e-6)
    return out


def add_market_profile_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tailored features per market from train target profile.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    by_mhd = (
        train_out.groupby(["market", "hour", "dow"], dropna=False)["target"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "target_profile_mhd_mean", "std": "target_profile_mhd_std"})
    )
    by_mh = (
        train_out.groupby(["market", "hour"], dropna=False)["target"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "target_profile_mh_mean", "std": "target_profile_mh_std"})
    )
    by_m = (
        train_out.groupby("market", dropna=False)["target"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "target_profile_m_mean", "std": "target_profile_m_std"})
    )

    train_out = train_out.merge(by_mhd, on=["market", "hour", "dow"], how="left")
    test_out = test_out.merge(by_mhd, on=["market", "hour", "dow"], how="left")
    train_out = train_out.merge(by_mh, on=["market", "hour"], how="left")
    test_out = test_out.merge(by_mh, on=["market", "hour"], how="left")
    train_out = train_out.merge(by_m, on=["market"], how="left")
    test_out = test_out.merge(by_m, on=["market"], how="left")
    return train_out, test_out


def apply_exclude_2023(df: pd.DataFrame, keep_from_month: int = 10) -> pd.DataFrame:
    out = df.copy()
    start = _to_datetime_col(out, "delivery_start")
    year = start.dt.year
    month = start.dt.month
    keep_mask = (year != 2023) | (month >= keep_from_month)
    removed = int((~keep_mask).sum())
    kept_2023 = int(((year == 2023) & keep_mask).sum())
    print(
        "Exclude 2023 mode: "
        f"removed={removed}, kept_from_2023_month_{keep_from_month}={kept_2023}"
    )
    return out.loc[keep_mask].copy()


@dataclass
class TrainArtifacts:
    global_model: CatBoostRegressor
    local_models: dict[str, CatBoostRegressor]
    feature_cols: list[str]
    cat_cols: list[str]


def train_global_and_local_models(train_df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> TrainArtifacts:
    global_model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=18.0,
        bagging_temperature=0.5,
        random_strength=1.0,
        random_seed=42,
        verbose=0,
    )
    global_model.fit(train_df[feature_cols], train_df["target"], cat_features=cat_cols)
    train_df = train_df.copy()
    train_df["global_pred_feature"] = global_model.predict(train_df[feature_cols])

    local_feature_cols = feature_cols + ["global_pred_feature"]
    local_models: dict[str, CatBoostRegressor] = {}
    for market, mdf in train_df.groupby("market", dropna=False):
        model = CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=3000,
            learning_rate=0.025,
            depth=8,
            l2_leaf_reg=20.0,
            bagging_temperature=0.4,
            random_strength=0.9,
            random_seed=42,
            verbose=0,
        )
        model.fit(mdf[local_feature_cols], mdf["target"], cat_features=cat_cols)
        local_models[str(market)] = model
        print(f"Trained local model for {market} ({len(mdf)} rows)")

    return TrainArtifacts(
        global_model=global_model,
        local_models=local_models,
        feature_cols=local_feature_cols,
        cat_cols=cat_cols,
    )


def build_feature_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat([train_df.assign(_is_train=1), test_df.assign(_is_train=0)], axis=0, ignore_index=True)
    all_df = add_time_features(all_df)
    all_df = add_forecast_core_features(all_df)
    all_df = add_forecast_lag_features(all_df)
    all_df = add_meteo_features(all_df)
    all_df = add_cross_market_features(all_df)
    all_df = add_missingness_features(all_df)
    all_df = add_market_categorical_interactions(all_df)

    train_out = all_df.loc[all_df["_is_train"] == 1].drop(columns=["_is_train"]).copy()
    test_out = all_df.loc[all_df["_is_train"] == 0].drop(columns=["_is_train", "target"], errors="ignore").copy()
    train_out, test_out = add_market_profile_features(train_out, test_out)
    return train_out, test_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-market intraday training with tailored and cross-market features.")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="per_market_interactions")
    parser.add_argument("--exclude-2023", action="store_true")
    parser.add_argument("--exclude-2023-keep-from-month", type=int, default=10)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample = pd.read_csv(args.sample_submission)

    if args.exclude_2023:
        train_df = apply_exclude_2023(train_df, keep_from_month=args.exclude_2023_keep_from_month)

    train_feat, test_feat = build_feature_table(train_df, test_df)
    test_with_key = test_feat[["id", "market"]].copy()

    base_drop = {"id", "target", "delivery_start", "delivery_end"}
    candidate_features = [c for c in train_feat.columns if c not in base_drop]
    cat_cols = [
        c
        for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
        if c in candidate_features
    ]

    artifacts = train_global_and_local_models(train_feat, candidate_features, cat_cols)

    # Add global prediction feature to test and run local market experts.
    test_feat = test_feat.copy()
    test_feat["global_pred_feature"] = artifacts.global_model.predict(test_feat[candidate_features])

    pred = np.full(len(test_feat), np.nan, dtype=float)
    for market, idx in test_with_key.groupby("market", dropna=False).groups.items():
        model = artifacts.local_models.get(str(market))
        if model is None:
            # Fallback to global if market-specific model doesn't exist.
            pred[idx] = test_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float)
            continue
        pred[idx] = model.predict(test_feat.loc[idx, artifacts.feature_cols])

    if np.isnan(pred).any():
        raise ValueError("NaNs found in predictions.")

    out_sub = sample[["id"]].copy()
    pred_map = pd.Series(pred, index=test_feat["id"].astype(int))
    out_sub["target"] = out_sub["id"].astype(int).map(pred_map)
    if out_sub["target"].isna().any():
        raise ValueError("Submission has NaN targets after mapping.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / f"{stamp}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sub_path = run_dir / "submission.csv"
    out_sub.to_csv(sub_path, index=False)

    latest_path = Path("submission_per_market_interactions.csv")
    out_sub.to_csv(latest_path, index=False)

    print(f"Saved submission: {sub_path}")
    print(f"Saved latest copy: {latest_path}")
    print(f"Features used: {len(artifacts.feature_cols)}")
    print(f"Categorical features: {artifacts.cat_cols}")
    print(f"Markets modeled: {sorted(artifacts.local_models.keys())}")


if __name__ == "__main__":
    main()
