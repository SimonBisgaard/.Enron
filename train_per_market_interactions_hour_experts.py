from __future__ import annotations

import argparse
import json
import calendar
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from dateutil.easter import easter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer


def _write_params_yaml(path: Path, payload: dict[str, Any]) -> None:
    def _format_scalar(value: Any) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            v = float(value)
            if not np.isfinite(v):
                return json.dumps(str(v))
            return str(v)
        return json.dumps(str(value), ensure_ascii=False)

    lines: list[str] = []

    def _emit(key: str, value: Any, indent: int) -> None:
        pad = "  " * indent
        if isinstance(value, dict):
            if not value:
                lines.append(f"{pad}{key}: {{}}")
                return
            lines.append(f"{pad}{key}:")
            for sub_key, sub_value in value.items():
                _emit(str(sub_key), sub_value, indent + 1)
            return

        if isinstance(value, (list, tuple)):
            if not value:
                lines.append(f"{pad}{key}: []")
                return
            lines.append(f"{pad}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{pad}  -")
                    for sub_key, sub_value in item.items():
                        _emit(str(sub_key), sub_value, indent + 2)
                else:
                    lines.append(f"{pad}  - {_format_scalar(item)}")
            return

        lines.append(f"{pad}{key}: {_format_scalar(value)}")

    for k, v in payload.items():
        _emit(str(k), v, 0)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _last_sunday(year: int, month: int) -> datetime:
    last_day = calendar.monthrange(year, month)[1]
    d = datetime(year, month, last_day)
    return d if d.weekday() == 6 else d - pd.Timedelta(days=d.weekday() + 1)


def add_temporal_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    start = _to_datetime_col(out, "delivery_start")
    day = start.dt.floor("D")

    out["is_month_start"] = start.dt.is_month_start.astype(int)
    out["is_month_end"] = start.dt.is_month_end.astype(int)
    out["is_quarter_start"] = start.dt.is_quarter_start.astype(int)
    out["is_quarter_end"] = start.dt.is_quarter_end.astype(int)
    out["is_year_start"] = start.dt.is_year_start.astype(int)
    out["is_year_end"] = start.dt.is_year_end.astype(int)

    # Public-holiday proxy shared across many EU power markets.
    holiday_dates: set[pd.Timestamp] = set()
    for y in sorted(start.dt.year.unique()):
        y = int(y)
        easter_sunday = pd.Timestamp(easter(y))
        holiday_dates.update(
            {
                pd.Timestamp(year=y, month=1, day=1),   # New Year
                pd.Timestamp(year=y, month=5, day=1),   # Labour day
                pd.Timestamp(year=y, month=12, day=25), # Christmas
                pd.Timestamp(year=y, month=12, day=26), # Boxing/St. Stephen
                easter_sunday - pd.Timedelta(days=2),   # Good Friday
                easter_sunday + pd.Timedelta(days=1),   # Easter Monday
            }
        )
    out["is_holiday"] = day.isin(list(holiday_dates)).astype(int)
    out["is_bridge_day"] = (
        (out["is_holiday"] == 0)
        & (
            (day - pd.Timedelta(days=1)).isin(list(holiday_dates))
            | (day + pd.Timedelta(days=1)).isin(list(holiday_dates))
        )
        & (start.dt.dayofweek < 5)
    ).astype(int)

    # EU DST transition days (last Sunday in March/October).
    dst_start_days = {pd.Timestamp(_last_sunday(int(y), 3)).floor("D") for y in start.dt.year.unique()}
    dst_end_days = {pd.Timestamp(_last_sunday(int(y), 10)).floor("D") for y in start.dt.year.unique()}
    out["is_dst_start_day"] = day.isin(list(dst_start_days)).astype(int)
    out["is_dst_end_day"] = day.isin(list(dst_end_days)).astype(int)
    out["is_dst_transition_day"] = ((out["is_dst_start_day"] + out["is_dst_end_day"]) > 0).astype(int)
    out["dst_transition_window"] = (
        (day - pd.Timedelta(days=1)).isin(list(dst_start_days | dst_end_days))
        | day.isin(list(dst_start_days | dst_end_days))
        | (day + pd.Timedelta(days=1)).isin(list(dst_start_days | dst_end_days))
    ).astype(int)

    # Additional seasonal phase features.
    day_of_year = start.dt.dayofyear
    out["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    return out


def add_forecast_core_features(df: pd.DataFrame, *, use_peak_interactions: bool) -> pd.DataFrame:
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
    if use_peak_interactions:
        out["net_load_share_peak"] = out["net_load_share"] * out["is_peak_17_20"]
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


def add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["market", "delivery_start"]).reset_index(drop=True)
    eps = 1e-6
    core = [
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
        "air_temperature_2m",
        "wind_speed_80m",
    ]
    for c in core:
        if c not in out.columns:
            continue
        g = out.groupby("market")[c]
        shifted = g.shift(1)
        mean24 = shifted.groupby(out["market"]).transform(lambda s: s.rolling(24, min_periods=6).mean())
        mean168 = shifted.groupby(out["market"]).transform(lambda s: s.rolling(168, min_periods=24).mean())
        std168 = shifted.groupby(out["market"]).transform(lambda s: s.rolling(168, min_periods=24).std())

        out[f"{c}_anom_24"] = out[c] - mean24
        out[f"{c}_anom_168"] = out[c] - mean168
        out[f"{c}_z_168"] = (out[c] - mean168) / (std168 + eps)
    return out


def add_volatility_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["market", "delivery_start"]).reset_index(drop=True)
    eps = 1e-6
    windows = {"7d": 168, "28d": 672}

    for c in ["residual_load", "wind_forecast", "load_forecast", "solar_forecast"]:
        if c not in out.columns:
            continue
        g = out.groupby("market")[c]
        diff1 = out[c] - g.shift(1)
        abs_diff1 = diff1.abs()
        out[f"{c}_absdiff_1"] = abs_diff1

        vol7 = (
            abs_diff1.groupby(out["market"])
            .transform(lambda s: s.shift(1).rolling(windows["7d"], min_periods=24).mean())
        )
        vol28 = (
            abs_diff1.groupby(out["market"])
            .transform(lambda s: s.shift(1).rolling(windows["28d"], min_periods=48).mean())
        )
        std28 = (
            abs_diff1.groupby(out["market"])
            .transform(lambda s: s.shift(1).rolling(windows["28d"], min_periods=48).std())
        )

        out[f"{c}_vol_7d"] = vol7
        out[f"{c}_vol_28d"] = vol28
        out[f"{c}_vol_ratio_7d_28d"] = vol7 / (vol28 + eps)
        out[f"{c}_vol_z_7d_28d"] = (vol7 - vol28) / (std28 + eps)
        out[f"{c}_vol_regime_high"] = (out[f"{c}_vol_ratio_7d_28d"] > 1.15).astype(int)
    return out


def add_meteo_features(df: pd.DataFrame, *, use_peak_interactions: bool) -> pd.DataFrame:
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
    if use_peak_interactions and {"wind_speed_80m", "is_peak_17_20"}.issubset(out.columns):
        out["wind_speed_80m_peak"] = out["wind_speed_80m"] * out["is_peak_17_20"]
    if {"cloud_cover_total", "solar_forecast"}.issubset(out.columns):
        out["cloud_x_solar"] = out["cloud_cover_total"] * out["solar_forecast"]
        if use_peak_interactions:
            out["cloud_x_solar_day"] = out["cloud_cover_total"] * out["solar_forecast"] * (
                out["solar_forecast"] > 0.0
            ).astype(int)

    if "wind_speed_80m" in out.columns:
        ws80 = out["wind_speed_80m"]
        out["ws80_sq"] = ws80**2
        out["ws80_cu"] = ws80**3

        # Piecewise turbine-style behavior (cut-in / rated / cut-out proxies).
        out["ws80_bin_low"] = (ws80 < 3.0).astype(int)
        out["ws80_bin_medium"] = ((ws80 >= 3.0) & (ws80 < 12.0)).astype(int)
        out["ws80_bin_high"] = ((ws80 >= 12.0) & (ws80 < 25.0)).astype(int)
        out["ws80_bin_cutout"] = (ws80 >= 25.0).astype(int)
        out["ws80_above_cutin"] = np.maximum(ws80 - 3.0, 0.0)
        out["ws80_above_rated"] = np.maximum(ws80 - 12.0, 0.0)
        out["ws80_below_cutin"] = np.maximum(3.0 - ws80, 0.0)
        out["ws80_medium_cubic"] = out["ws80_bin_medium"] * (ws80**3)
        out["ws80_high_flat"] = out["ws80_bin_high"] * 12.0**3

    return out


def add_wind_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "wind_forecast" not in out.columns:
        return out

    proxy_candidates = [
        "market",
        "hour",
        "dow",
        "month",
        "wind_speed_80m",
        "wind_speed_10m",
        "wind_gust_speed_10m",
        "wind_direction_80m",
        "ws80_sq",
        "ws80_cu",
        "ws80_bin_low",
        "ws80_bin_medium",
        "ws80_bin_high",
        "ws80_bin_cutout",
        "ws80_above_cutin",
        "ws80_above_rated",
        "ws80_medium_cubic",
        "ws80_high_flat",
        "gust_ratio_10m",
        "wind_dir_sin",
        "wind_dir_cos",
        "air_temperature_2m",
        "surface_pressure",
        "relative_humidity_2m",
    ]
    proxy_features = [c for c in proxy_candidates if c in out.columns]
    if not proxy_features:
        return out

    fit_mask = out["wind_forecast"].notna()
    if fit_mask.sum() < 100:
        return out

    cat_cols = [c for c in ["market"] if c in proxy_features]
    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=8.0,
        random_seed=42,
        verbose=0,
    )
    model.fit(out.loc[fit_mask, proxy_features], out.loc[fit_mask, "wind_forecast"], cat_features=cat_cols)

    out["wind_prod_est"] = model.predict(out[proxy_features])
    out["residual_load_est"] = out["load_forecast"] - out["solar_forecast"] - out["wind_prod_est"]
    out["wind_prod_est_ratio"] = out["wind_prod_est"] / (out["load_forecast"] + 1e-6)
    out["residual_load_est_share"] = out["residual_load_est"] / (out["load_forecast"] + 1e-6)
    out["wind_forecast_proxy_gap"] = out["wind_forecast"] - out["wind_prod_est"]
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


def add_cross_market_features(df: pd.DataFrame, *, use_rank_features: bool) -> pd.DataFrame:
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

    # Relative rank across markets at each timestamp.
    if use_rank_features:
        rank_cols = [
            c
            for c in ["load_forecast", "wind_forecast", "solar_forecast", "residual_load"]
            if c in out.columns
        ]
        for c in rank_cols:
            out[f"{c}_xmk_rank_pct"] = grouped[c].rank(method="average", pct=True)
    return out


def _add_dynamic_profile_for_keys(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    keys: list[str],
    prefix: str,
    *,
    target_col: str = "target",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()

    # Train features are strictly past-only per group (no future leakage).
    work = train_out[keys + ["delivery_start", target_col]].copy()
    work["_orig_idx"] = work.index
    work = work.sort_values(keys + ["delivery_start", "_orig_idx"]).reset_index(drop=True)
    grp = work.groupby(keys, dropna=False, sort=False)

    work[f"{prefix}_dyn_count"] = grp.cumcount().astype(float)
    work["_target_sq"] = work[target_col] ** 2
    work["_sum_prev"] = grp[target_col].cumsum() - work[target_col]
    work["_sum_sq_prev"] = grp["_target_sq"].cumsum() - work["_target_sq"]
    denom = work[f"{prefix}_dyn_count"].replace(0.0, np.nan)

    work[f"{prefix}_dyn_mean"] = work["_sum_prev"] / denom
    work[f"{prefix}_dyn_sq_mean"] = work["_sum_sq_prev"] / denom
    var_prev = work[f"{prefix}_dyn_sq_mean"] - work[f"{prefix}_dyn_mean"] ** 2
    work[f"{prefix}_dyn_std"] = np.sqrt(np.maximum(var_prev, 0.0))
    work[f"{prefix}_dyn_rms"] = np.sqrt(np.maximum(work[f"{prefix}_dyn_sq_mean"], 0.0))

    dyn_cols = [
        f"{prefix}_dyn_count",
        f"{prefix}_dyn_mean",
        f"{prefix}_dyn_sq_mean",
        f"{prefix}_dyn_std",
        f"{prefix}_dyn_rms",
    ]
    train_out = train_out.join(work.set_index("_orig_idx")[dyn_cols], how="left")

    # Test rows get latest profile from full train history.
    agg_in = train_out[keys + [target_col]].copy()
    agg_in["_target_sq"] = agg_in[target_col] ** 2
    agg = (
        agg_in.groupby(keys, dropna=False)
        .agg(
            **{
                f"{prefix}_dyn_count": (target_col, "count"),
                f"{prefix}_dyn_mean": (target_col, "mean"),
                f"{prefix}_dyn_std": (target_col, "std"),
                f"{prefix}_dyn_sq_mean": ("_target_sq", "mean"),
            }
        )
        .reset_index()
    )
    agg[f"{prefix}_dyn_count"] = agg[f"{prefix}_dyn_count"].astype(float)
    agg[f"{prefix}_dyn_rms"] = np.sqrt(np.maximum(agg[f"{prefix}_dyn_sq_mean"], 0.0))
    test_out = test_out.merge(agg, on=keys, how="left")
    return train_out, test_out


def add_dynamic_market_profile_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str = "target",
    shrink_alpha_mhd: float = 24.0,
    shrink_alpha_m: float = 24.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train_df.copy()
    test_out = test_df.copy()

    scopes = [
        (["market", "hour", "dow"], "target_profile_mhd"),
        (["market", "hour"], "target_profile_mh"),
        (["market"], "target_profile_m"),
    ]
    for keys, prefix in scopes:
        train_out, test_out = _add_dynamic_profile_for_keys(
            train_out,
            test_out,
            keys,
            prefix,
            target_col=target_col,
        )

    # Fall back to static profile values when dynamic history is missing.
    for prefix in ["target_profile_mhd", "target_profile_mh", "target_profile_m"]:
        mean_col = f"{prefix}_dyn_mean"
        sq_col = f"{prefix}_dyn_sq_mean"
        std_col = f"{prefix}_dyn_std"
        rms_col = f"{prefix}_dyn_rms"
        count_col = f"{prefix}_dyn_count"
        base_mean = f"{prefix}_mean"
        base_std = f"{prefix}_std"

        if base_mean in train_out.columns:
            train_out[mean_col] = train_out[mean_col].fillna(train_out[base_mean])
            test_out[mean_col] = test_out[mean_col].fillna(test_out[base_mean])
        if base_std in train_out.columns:
            train_out[std_col] = train_out[std_col].fillna(train_out[base_std])
            test_out[std_col] = test_out[std_col].fillna(test_out[base_std])

        train_out[sq_col] = train_out[sq_col].fillna(train_out[mean_col] ** 2)
        test_out[sq_col] = test_out[sq_col].fillna(test_out[mean_col] ** 2)
        train_out[rms_col] = train_out[rms_col].fillna(np.sqrt(np.maximum(train_out[sq_col], 0.0)))
        test_out[rms_col] = test_out[rms_col].fillna(np.sqrt(np.maximum(test_out[sq_col], 0.0)))
        train_out[count_col] = train_out[count_col].fillna(0.0)
        test_out[count_col] = test_out[count_col].fillna(0.0)

    global_mean = float(train_out[target_col].mean())
    global_std = float(train_out[target_col].std())

    # Cross-scope backoff for unseen combinations in test.
    for df in [train_out, test_out]:
        df["target_profile_m_dyn_mean"] = (
            df["target_profile_m_dyn_mean"]
            .fillna(df["target_profile_m_mean"])
            .fillna(global_mean)
        )
        df["target_profile_m_dyn_std"] = (
            df["target_profile_m_dyn_std"]
            .fillna(df["target_profile_m_std"])
            .fillna(global_std)
        )

        df["target_profile_mh_dyn_mean"] = (
            df["target_profile_mh_dyn_mean"]
            .fillna(df["target_profile_mh_mean"])
            .fillna(df["target_profile_m_dyn_mean"])
        )
        df["target_profile_mh_dyn_std"] = (
            df["target_profile_mh_dyn_std"]
            .fillna(df["target_profile_mh_std"])
            .fillna(df["target_profile_m_dyn_std"])
        )

        df["target_profile_mhd_dyn_mean"] = (
            df["target_profile_mhd_dyn_mean"]
            .fillna(df["target_profile_mhd_mean"])
            .fillna(df["target_profile_mh_dyn_mean"])
            .fillna(df["target_profile_m_dyn_mean"])
        )
        df["target_profile_mhd_dyn_std"] = (
            df["target_profile_mhd_dyn_std"]
            .fillna(df["target_profile_mhd_std"])
            .fillna(df["target_profile_mh_dyn_std"])
            .fillna(df["target_profile_m_dyn_std"])
        )

        # Count-aware shrinkage to market-hour level prevents tiny-group spikes.
        c_mhd = pd.to_numeric(df["target_profile_mhd_dyn_count"], errors="coerce").fillna(0.0)
        c_mh = pd.to_numeric(df["target_profile_mh_dyn_count"], errors="coerce").fillna(0.0)
        a_mhd = float(max(shrink_alpha_mhd, 1e-6))
        a_m = float(max(shrink_alpha_m, 1e-6))
        w_mhd = c_mhd / (c_mhd + a_mhd)
        w_mh = c_mh / (c_mh + a_m)

        df["target_profile_mhd_dyn_mean"] = (
            w_mhd * df["target_profile_mhd_dyn_mean"]
            + (1.0 - w_mhd) * df["target_profile_mh_dyn_mean"]
        )
        df["target_profile_mhd_dyn_std"] = np.maximum(
            0.0,
            w_mhd * df["target_profile_mhd_dyn_std"]
            + (1.0 - w_mhd) * df["target_profile_mh_dyn_std"],
        )

        df["target_profile_m_dyn_mean"] = (
            (1.0 - w_mh) * df["target_profile_m_dyn_mean"]
            + w_mh * df["target_profile_mh_dyn_mean"]
        )
        df["target_profile_m_dyn_std"] = np.maximum(
            0.0,
            (1.0 - w_mh) * df["target_profile_m_dyn_std"]
            + w_mh * df["target_profile_mh_dyn_std"],
        )

        for prefix in ["target_profile_m", "target_profile_mh", "target_profile_mhd"]:
            mean_col = f"{prefix}_dyn_mean"
            sq_col = f"{prefix}_dyn_sq_mean"
            std_col = f"{prefix}_dyn_std"
            rms_col = f"{prefix}_dyn_rms"
            count_col = f"{prefix}_dyn_count"
            df[sq_col] = np.maximum(df[mean_col] ** 2 + np.maximum(df[std_col], 0.0) ** 2, 0.0)
            df[rms_col] = np.sqrt(np.maximum(df[sq_col], 0.0))
            df[count_col] = df[count_col].fillna(0.0)
    return train_out, test_out


def add_market_profile_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    use_dynamic_profiles: bool,
    use_robust_profiles: bool = True,
    robust_winsor_lower_q: float = 0.01,
    robust_winsor_upper_q: float = 0.99,
    robust_winsor_min_rows: int = 24,
    profile_shrink_alpha_mhd: float = 24.0,
    profile_shrink_alpha_m: float = 24.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Tailored features per market from train target profile.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    if not (0.0 < robust_winsor_lower_q < robust_winsor_upper_q < 1.0):
        raise ValueError("Robust profile quantiles must satisfy 0 < lower < upper < 1.")
    if robust_winsor_min_rows < 1:
        raise ValueError("robust_winsor_min_rows must be >= 1.")

    profile_target_col = "target"
    if use_robust_profiles:
        base = train_out[["market", "hour", "target"]].copy()
        g_mh = base.groupby(["market", "hour"], dropna=False)["target"]
        by_mh_q = pd.concat(
            [
                g_mh.quantile(robust_winsor_lower_q).rename("_q_lo_mh"),
                g_mh.quantile(robust_winsor_upper_q).rename("_q_hi_mh"),
                g_mh.size().rename("_n_mh"),
            ],
            axis=1,
        ).reset_index()

        g_m = base.groupby("market", dropna=False)["target"]
        by_m_q = pd.concat(
            [
                g_m.quantile(robust_winsor_lower_q).rename("_q_lo_m"),
                g_m.quantile(robust_winsor_upper_q).rename("_q_hi_m"),
            ],
            axis=1,
        ).reset_index()

        q_lo_g = float(base["target"].quantile(robust_winsor_lower_q))
        q_hi_g = float(base["target"].quantile(robust_winsor_upper_q))
        tmp = base.merge(by_mh_q, on=["market", "hour"], how="left").merge(by_m_q, on=["market"], how="left")
        use_mh = pd.to_numeric(tmp["_n_mh"], errors="coerce").fillna(0.0) >= float(robust_winsor_min_rows)
        lo = np.where(use_mh, tmp["_q_lo_mh"], tmp["_q_lo_m"])
        hi = np.where(use_mh, tmp["_q_hi_mh"], tmp["_q_hi_m"])
        lo = pd.Series(lo, index=tmp.index).fillna(q_lo_g).to_numpy(dtype=float)
        hi = pd.Series(hi, index=tmp.index).fillna(q_hi_g).to_numpy(dtype=float)
        hi = np.maximum(hi, lo)

        train_out["_target_profile_robust"] = np.clip(train_out["target"].to_numpy(dtype=float), lo, hi)
        profile_target_col = "_target_profile_robust"

    by_mhd = (
        train_out.groupby(["market", "hour", "dow"], dropna=False)[profile_target_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "target_profile_mhd_mean",
                "std": "target_profile_mhd_std",
                "count": "target_profile_mhd_count",
            }
        )
    )
    by_mh = (
        train_out.groupby(["market", "hour"], dropna=False)[profile_target_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "target_profile_mh_mean",
                "std": "target_profile_mh_std",
                "count": "target_profile_mh_count",
            }
        )
    )
    by_m = (
        train_out.groupby("market", dropna=False)[profile_target_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "target_profile_m_mean",
                "std": "target_profile_m_std",
                "count": "target_profile_m_count",
            }
        )
    )

    train_out = train_out.merge(by_mhd, on=["market", "hour", "dow"], how="left")
    test_out = test_out.merge(by_mhd, on=["market", "hour", "dow"], how="left")
    train_out = train_out.merge(by_mh, on=["market", "hour"], how="left")
    test_out = test_out.merge(by_mh, on=["market", "hour"], how="left")
    train_out = train_out.merge(by_m, on=["market"], how="left")
    test_out = test_out.merge(by_m, on=["market"], how="left")

    global_mean = float(train_out[profile_target_col].mean())
    global_std = float(train_out[profile_target_col].std())
    if not np.isfinite(global_std):
        global_std = 0.0

    for df in [train_out, test_out]:
        df["target_profile_m_mean"] = df["target_profile_m_mean"].fillna(global_mean)
        df["target_profile_m_std"] = df["target_profile_m_std"].fillna(global_std)

        df["target_profile_mh_mean"] = df["target_profile_mh_mean"].fillna(df["target_profile_m_mean"])
        df["target_profile_mh_std"] = df["target_profile_mh_std"].fillna(df["target_profile_m_std"])

        df["target_profile_mhd_mean"] = df["target_profile_mhd_mean"].fillna(df["target_profile_mh_mean"])
        df["target_profile_mhd_std"] = df["target_profile_mhd_std"].fillna(df["target_profile_mh_std"])

        for c in ["target_profile_m_count", "target_profile_mh_count", "target_profile_mhd_count"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # Count-aware shrinkage toward market-hour baselines.
        a_mhd = float(max(profile_shrink_alpha_mhd, 1e-6))
        a_m = float(max(profile_shrink_alpha_m, 1e-6))

        w_mhd = df["target_profile_mhd_count"] / (df["target_profile_mhd_count"] + a_mhd)
        w_mh = df["target_profile_mh_count"] / (df["target_profile_mh_count"] + a_m)

        df["target_profile_mhd_mean"] = (
            w_mhd * df["target_profile_mhd_mean"]
            + (1.0 - w_mhd) * df["target_profile_mh_mean"]
        )
        df["target_profile_mhd_std"] = np.maximum(
            0.0,
            w_mhd * df["target_profile_mhd_std"]
            + (1.0 - w_mhd) * df["target_profile_mh_std"],
        )

        df["target_profile_m_mean"] = (
            (1.0 - w_mh) * df["target_profile_m_mean"]
            + w_mh * df["target_profile_mh_mean"]
        )
        df["target_profile_m_std"] = np.maximum(
            0.0,
            (1.0 - w_mh) * df["target_profile_m_std"]
            + w_mh * df["target_profile_mh_std"],
        )

    if use_dynamic_profiles:
        train_out, test_out = add_dynamic_market_profile_features(
            train_out,
            test_out,
            target_col=profile_target_col,
            shrink_alpha_mhd=profile_shrink_alpha_mhd,
            shrink_alpha_m=profile_shrink_alpha_m,
        )

    train_out = train_out.drop(columns=["_target_profile_robust"], errors="ignore")
    test_out = test_out.drop(columns=["_target_profile_robust"], errors="ignore")
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
class PredictionCapper:
    by_market_hour: pd.DataFrame
    by_market: pd.DataFrame
    global_lower: float
    global_upper: float
    softness: float


def _ensure_hour_column(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" in df.columns:
        out = df.copy()
        out["hour"] = pd.to_numeric(out["hour"], errors="coerce").fillna(-1).astype(int)
        return out
    if "delivery_start" not in df.columns:
        raise ValueError("Need either 'hour' or 'delivery_start' to build market-hour caps.")
    out = df.copy()
    out["hour"] = _to_datetime_col(out, "delivery_start").dt.hour.astype(int)
    return out


def fit_market_hour_prediction_capper(
    train_df: pd.DataFrame,
    *,
    lower_q: float,
    upper_q: float,
    min_rows: int,
    softness: float,
) -> PredictionCapper:
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("Prediction cap quantiles must satisfy 0 <= lower < upper <= 1.")
    if min_rows < 1:
        raise ValueError("Prediction cap min_rows must be >= 1.")
    if not (0.0 <= softness <= 1.0):
        raise ValueError("Prediction cap softness must be in [0,1].")

    base = _ensure_hour_column(train_df)[["market", "hour", "target"]].copy()

    g_mh = base.groupby(["market", "hour"], dropna=False)["target"]
    by_mh = pd.concat(
        [
            g_mh.quantile(lower_q).rename("lower"),
            g_mh.quantile(upper_q).rename("upper"),
            g_mh.size().rename("n"),
        ],
        axis=1,
    ).reset_index()
    by_mh = by_mh.loc[by_mh["n"] >= int(min_rows)].copy()

    g_m = base.groupby("market", dropna=False)["target"]
    by_m = pd.concat(
        [
            g_m.quantile(lower_q).rename("lower"),
            g_m.quantile(upper_q).rename("upper"),
        ],
        axis=1,
    ).reset_index()

    global_lower = float(base["target"].quantile(lower_q))
    global_upper = float(base["target"].quantile(upper_q))
    if global_upper < global_lower:
        global_upper = global_lower

    return PredictionCapper(
        by_market_hour=by_mh,
        by_market=by_m,
        global_lower=global_lower,
        global_upper=global_upper,
        softness=float(softness),
    )


def apply_market_hour_prediction_caps(
    pred: np.ndarray,
    rows: pd.DataFrame,
    capper: PredictionCapper,
) -> tuple[np.ndarray, int]:
    if len(pred) == 0:
        return pred.astype(float, copy=True), 0

    row_df = _ensure_hour_column(rows)[["market", "hour"]].copy()
    row_df["_row"] = np.arange(len(row_df), dtype=int)

    tmp = row_df.merge(capper.by_market_hour, on=["market", "hour"], how="left")
    by_m = capper.by_market.rename(columns={"lower": "lower_m", "upper": "upper_m"})
    tmp = tmp.merge(by_m, on=["market"], how="left")

    lower = tmp["lower"].fillna(tmp["lower_m"]).fillna(capper.global_lower).to_numpy(dtype=float)
    upper = tmp["upper"].fillna(tmp["upper_m"]).fillna(capper.global_upper).to_numpy(dtype=float)
    upper = np.maximum(upper, lower)

    out = pred.astype(float, copy=True)
    softness = float(capper.softness)
    high = out > upper
    low = out < lower
    out[high] = upper[high] + softness * (out[high] - upper[high])
    out[low] = lower[low] + softness * (out[low] - lower[low])
    changed = int(np.count_nonzero(np.abs(out - pred) > 1e-12))
    return out, changed


@dataclass
class TrainArtifacts:
    global_model: CatBoostRegressor
    local_models: dict[str, CatBoostRegressor]
    local_tail_models: dict[str, CatBoostRegressor]
    local_gate_models: dict[str, CatBoostClassifier]
    local_hour_models: dict[str, dict[str, CatBoostRegressor]]
    local_hour_model_counts: dict[str, dict[str, int]]
    feature_cols: list[str]
    cat_cols: list[str]
    local_target_is_residual: bool
    local_uses_global_pred_feature: bool
    target_transform_state: TargetTransformState
    use_tail_experts: bool
    tail_label_mode: str
    tail_quantile_upper: float
    tail_quantile_lower: float
    tail_gate_threshold: float
    tail_blend_mode: str
    tail_delta_clip: float | None
    tail_model_mode: str
    tail_weight: float
    tail_min_rows: int
    use_hour_experts: bool
    hour_expert_mode: str
    hour_expert_min_rows: int
    hour_expert_weight: float


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass
class TargetTransformState:
    method: str
    log_shift: float = 0.0
    yeo_transformer: PowerTransformer | None = None

    def to_metadata(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method}
        if self.method == "log_shift":
            out["log_shift"] = float(self.log_shift)
        elif self.method == "yeo_johnson" and self.yeo_transformer is not None:
            out["yeo_lambda"] = float(self.yeo_transformer.lambdas_[0])
        return out


def _fit_target_transform(y: np.ndarray, method: str) -> tuple[np.ndarray, TargetTransformState]:
    y = y.astype(float)
    if method == "none":
        return y.copy(), TargetTransformState(method="none")
    if method == "signed_log":
        y_t = np.sign(y) * np.log1p(np.abs(y))
        return y_t, TargetTransformState(method="signed_log")
    if method == "log_shift":
        min_y = float(np.nanmin(y))
        # Keep argument strictly positive for log.
        shift = 1.0 - min_y if min_y <= 0.0 else 0.0
        y_t = np.log(y + shift)
        return y_t, TargetTransformState(method="log_shift", log_shift=shift)
    if method == "yeo_johnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        y_t = pt.fit_transform(y.reshape(-1, 1)).reshape(-1)
        return y_t, TargetTransformState(method="yeo_johnson", yeo_transformer=pt)
    raise ValueError(f"Unknown target transform method: {method}")


def _apply_target_inverse(y_pred: np.ndarray, state: TargetTransformState) -> np.ndarray:
    y_pred = y_pred.astype(float)
    if state.method == "none":
        return y_pred
    if state.method == "signed_log":
        return np.sign(y_pred) * np.expm1(np.abs(y_pred))
    if state.method == "log_shift":
        return np.exp(y_pred) - state.log_shift
    if state.method == "yeo_johnson":
        if state.yeo_transformer is None:
            raise ValueError("Missing Yeo-Johnson transformer for inverse transform.")
        return state.yeo_transformer.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    raise ValueError(f"Unknown target transform method: {state.method}")


def _predict_point(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    """
    Return point prediction for both standard RMSE and RMSEWithUncertainty models.
    RMSEWithUncertainty returns two columns; first is mean prediction.
    """
    pred = np.asarray(model.predict(X))
    if pred.ndim == 2:
        return pred[:, 0].astype(float)
    return pred.astype(float)


def _predict_tail_probability(model: CatBoostClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X))
    if proba.ndim == 2 and proba.shape[1] >= 2:
        return proba[:, 1].astype(float)
    if proba.ndim == 2 and proba.shape[1] == 1:
        return proba[:, 0].astype(float)
    return proba.astype(float)


def _compute_tail_mask_per_market(
    y_true: pd.Series,
    market_col: pd.Series,
    *,
    label_mode: str,
    quantile_upper: float,
    quantile_lower: float,
) -> np.ndarray:
    tmp = pd.DataFrame({"y": y_true.to_numpy(dtype=float), "market": market_col.astype(str).to_numpy()})
    mask = np.zeros(len(tmp), dtype=bool)
    for _, idx in tmp.groupby("market", dropna=False, sort=False).groups.items():
        pos = np.asarray(idx, dtype=int)
        vals = tmp.iloc[pos]["y"]
        q_hi = float(vals.quantile(quantile_upper))
        if label_mode == "upper":
            local = vals >= q_hi
        elif label_mode == "two_sided":
            q_lo = float(vals.quantile(quantile_lower))
            local = (vals >= q_hi) | (vals <= q_lo)
        else:
            raise ValueError(f"Unsupported tail label mode: {label_mode}")
        mask[pos] = local.to_numpy()
    return mask


def _blend_tail_expert_prediction(
    *,
    normal_pred: np.ndarray,
    tail_pred: np.ndarray,
    tail_prob: np.ndarray,
    gate_threshold: float,
    blend_mode: str,
    delta_clip: float | None,
) -> np.ndarray:
    if blend_mode == "hard":
        weight = (tail_prob >= gate_threshold).astype(float)
    elif blend_mode == "soft":
        denom = max(1.0 - gate_threshold, 1e-6)
        weight = np.clip((tail_prob - gate_threshold) / denom, 0.0, 1.0)
    else:
        raise ValueError(f"Unsupported tail blend mode: {blend_mode}")

    delta = tail_pred - normal_pred
    if delta_clip is not None and delta_clip > 0:
        delta = np.clip(delta, -delta_clip, delta_clip)
    return normal_pred + weight * delta


def _hour_regime_from_hour(hour: pd.Series, mode: str) -> pd.Series:
    h = pd.to_numeric(hour, errors="coerce").fillna(-1).astype(int)
    if mode == "hour":
        return h.map(lambda x: f"h{x:02d}" if 0 <= x <= 23 else "h_na")
    if mode == "block5":
        out = pd.Series(index=h.index, dtype=object)
        out.loc[(h >= 0) & (h <= 5)] = "night_00_05"
        out.loc[(h >= 6) & (h <= 10)] = "morning_06_10"
        out.loc[(h >= 11) & (h <= 16)] = "day_11_16"
        out.loc[(h >= 17) & (h <= 20)] = "peak_17_20"
        out.loc[(h >= 21) & (h <= 23)] = "late_21_23"
        out = out.fillna("h_na")
        return out
    raise ValueError(f"Unsupported hour expert mode: {mode}")


def _predict_local_expert(
    artifacts: TrainArtifacts,
    market: str,
    X: pd.DataFrame,
) -> np.ndarray:
    normal_model = artifacts.local_models.get(str(market))
    if normal_model is None:
        raise KeyError(f"Missing normal local model for market={market}")
    local_pred = _predict_point(normal_model, X[artifacts.feature_cols])

    if artifacts.use_tail_experts:
        tail_model = artifacts.local_tail_models.get(str(market))
        gate_model = artifacts.local_gate_models.get(str(market))
        if tail_model is not None and gate_model is not None:
            tail_pred = _predict_point(tail_model, X[artifacts.feature_cols])
            tail_prob = _predict_tail_probability(gate_model, X[artifacts.feature_cols])
            local_pred = _blend_tail_expert_prediction(
                normal_pred=local_pred,
                tail_pred=tail_pred,
                tail_prob=tail_prob,
                gate_threshold=artifacts.tail_gate_threshold,
                blend_mode=artifacts.tail_blend_mode,
                delta_clip=artifacts.tail_delta_clip,
            )
    if not artifacts.use_hour_experts:
        return local_pred

    hour_models = artifacts.local_hour_models.get(str(market), {})
    if not hour_models:
        return local_pred
    hour_counts = artifacts.local_hour_model_counts.get(str(market), {})
    regime = _hour_regime_from_hour(X["hour"], artifacts.hour_expert_mode).reset_index(drop=True)

    hour_adj = np.zeros(len(X), dtype=float)
    for r, pos_idx in regime.groupby(regime, dropna=False).groups.items():
        model = hour_models.get(str(r))
        if model is None:
            continue
        pos = np.asarray(pos_idx, dtype=int)
        sub = X.iloc[pos]
        raw_adj = _predict_point(model, sub[artifacts.feature_cols])
        n = float(hour_counts.get(str(r), 0))
        shrink = n / (n + float(max(artifacts.hour_expert_min_rows, 1)))
        hour_adj[pos] = float(artifacts.hour_expert_weight) * shrink * raw_adj
    return local_pred + hour_adj


def make_time_series_folds(
    train_df: pd.DataFrame,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = _to_datetime_col(train_df, "delivery_start")
    days = pd.Series(start.dt.floor("D").unique()).sort_values().reset_index(drop=True)
    if days.empty:
        return []

    max_day = days.iloc[-1]
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(n_folds):
        val_end = max_day - pd.Timedelta(days=step_days * i)
        val_start = val_end - pd.Timedelta(days=val_days)
        train_start = days.iloc[0]
        min_train_end = train_start + pd.Timedelta(days=min_train_days)
        if val_start <= min_train_end:
            break
        folds.append((val_start, val_end))
    folds.reverse()
    return folds


def run_time_series_cv(
    train_df_raw: pd.DataFrame,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
    use_dynamic_target_profiles: bool,
    use_robust_target_profiles: bool,
    profile_winsor_lower_q: float,
    profile_winsor_upper_q: float,
    profile_winsor_min_rows: int,
    profile_shrink_alpha_mhd: float,
    profile_shrink_alpha_m: float,
    use_temporal_regime: bool,
    use_volatility_regime: bool,
    use_wind_proxy: bool,
    use_anomaly_features: bool,
    use_cross_market_rank_features: bool,
    use_peak_interactions: bool,
    local_residual_modeling: bool,
    include_global_pred_in_local: bool,
    loss_function: str,
    target_transform: str,
    use_tail_experts: bool,
    tail_label_mode: str,
    tail_quantile_upper: float,
    tail_quantile_lower: float,
    tail_model_mode: str,
    tail_weight: float,
    tail_min_rows: int,
    tail_gate_threshold: float,
    tail_blend_mode: str,
    tail_delta_clip: float | None,
    use_hour_experts: bool,
    hour_expert_mode: str,
    hour_expert_min_rows: int,
    hour_expert_weight: float,
    use_market_hour_pred_caps: bool,
    pred_cap_lower_q: float,
    pred_cap_upper_q: float,
    pred_cap_min_rows: int,
    pred_cap_softness: float,
) -> tuple[float | None, pd.DataFrame, pd.DataFrame]:
    folds = make_time_series_folds(
        train_df=train_df_raw,
        n_folds=n_folds,
        val_days=val_days,
        step_days=step_days,
        min_train_days=min_train_days,
    )
    if not folds:
        print("CV skipped: not enough history for requested folds.")
        return None, pd.DataFrame(), pd.DataFrame()

    start_all = _to_datetime_col(train_df_raw, "delivery_start")
    fold_rows: list[dict[str, object]] = []
    oof_rows: list[pd.DataFrame] = []
    oof = pd.Series(index=train_df_raw.index, dtype=float)

    print(
        f"CV start: folds={len(folds)}, val_days={val_days}, step_days={step_days}, "
        f"min_train_days={min_train_days}"
    )
    for fold_idx, (val_start, val_end) in enumerate(folds, start=1):
        tr_mask = start_all < val_start
        va_mask = (start_all >= val_start) & (start_all < val_end)
        tr = train_df_raw.loc[tr_mask].copy()
        va = train_df_raw.loc[va_mask].copy()
        if tr.empty or va.empty:
            print(f"CV fold {fold_idx}: skipped (empty train/val)")
            continue

        print(
            f"CV fold {fold_idx}/{len(folds)} | "
            f"train={len(tr)} val={len(va)} | "
            f"val_range=[{val_start.date()} -> {val_end.date()})"
        )

        tr_feat, va_feat = build_feature_table(
            tr,
            va.drop(columns=["target"]).copy(),
            use_dynamic_target_profiles=use_dynamic_target_profiles,
            use_robust_target_profiles=use_robust_target_profiles,
            profile_winsor_lower_q=profile_winsor_lower_q,
            profile_winsor_upper_q=profile_winsor_upper_q,
            profile_winsor_min_rows=profile_winsor_min_rows,
            profile_shrink_alpha_mhd=profile_shrink_alpha_mhd,
            profile_shrink_alpha_m=profile_shrink_alpha_m,
            use_temporal_regime=use_temporal_regime,
            use_volatility_regime=use_volatility_regime,
            use_wind_proxy=use_wind_proxy,
            use_anomaly_features=use_anomaly_features,
            use_cross_market_rank_features=use_cross_market_rank_features,
            use_peak_interactions=use_peak_interactions,
        )

        base_drop = {"id", "target", "delivery_start", "delivery_end"}
        feat_cols = [c for c in tr_feat.columns if c not in base_drop]
        cat_cols = [
            c
            for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
            if c in feat_cols
        ]

        artifacts = train_global_and_local_models(
            tr_feat,
            feat_cols,
            cat_cols,
            local_residual_modeling=local_residual_modeling,
            include_global_pred_in_local=include_global_pred_in_local,
            loss_function=loss_function,
            target_transform=target_transform,
            use_tail_experts=use_tail_experts,
            tail_label_mode=tail_label_mode,
            tail_quantile_upper=tail_quantile_upper,
            tail_quantile_lower=tail_quantile_lower,
            tail_model_mode=tail_model_mode,
            tail_weight=tail_weight,
            tail_min_rows=tail_min_rows,
            tail_gate_threshold=tail_gate_threshold,
            tail_blend_mode=tail_blend_mode,
            tail_delta_clip=tail_delta_clip,
            use_hour_experts=use_hour_experts,
            hour_expert_mode=hour_expert_mode,
            hour_expert_min_rows=hour_expert_min_rows,
            hour_expert_weight=hour_expert_weight,
        )
        va_feat = va_feat.copy()
        va_feat["global_pred_feature"] = _predict_point(artifacts.global_model, va_feat[feat_cols])

        pred = np.full(len(va_feat), np.nan, dtype=float)
        key = va_feat[["market"]].copy()
        for market, idx in key.groupby("market", dropna=False).groups.items():
            normal_model = artifacts.local_models.get(str(market))
            if normal_model is None:
                pred[idx] = va_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float)
            else:
                local_pred = _predict_local_expert(
                    artifacts=artifacts,
                    market=str(market),
                    X=va_feat.loc[idx],
                )
                if artifacts.local_target_is_residual:
                    pred[idx] = va_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float) + local_pred
                else:
                    pred[idx] = local_pred
        if np.isnan(pred).any():
            raise ValueError(f"NaNs in CV predictions for fold {fold_idx}")

        pred_original = _apply_target_inverse(pred, artifacts.target_transform_state)
        if use_market_hour_pred_caps:
            capper = fit_market_hour_prediction_capper(
                tr_feat,
                lower_q=pred_cap_lower_q,
                upper_q=pred_cap_upper_q,
                min_rows=pred_cap_min_rows,
                softness=pred_cap_softness,
            )
            pred_original, n_capped = apply_market_hour_prediction_caps(
                pred_original,
                va_feat[["market", "hour"]],
                capper,
            )
            print(
                f"  - fold {fold_idx} market-hour caps: adjusted_rows={n_capped} "
                f"({(100.0 * n_capped / max(len(pred_original), 1)):.2f}%)"
            )

        fold_rmse = _rmse(va["target"].to_numpy(dtype=float), pred_original)
        print(f"CV fold {fold_idx} RMSE={fold_rmse:.6f}")
        fold_rows.append(
            {
                "fold": fold_idx,
                "val_start": str(val_start.date()),
                "val_end": str(val_end.date()),
                "rmse": fold_rmse,
                "train_rows": len(tr),
                "val_rows": len(va),
            }
        )

        tmp = pd.DataFrame(
            {
                "idx": va.index.to_numpy(),
                "id": va["id"].to_numpy(dtype=int),
                "delivery_start": va["delivery_start"].to_numpy(),
                "market": va["market"].to_numpy(),
                "y_true": va["target"].to_numpy(dtype=float),
                "y_pred": pred_original,
                "fold": fold_idx,
            }
        )
        for market, sdf in tmp.groupby("market"):
            m_rmse = _rmse(sdf["y_true"].to_numpy(), sdf["y_pred"].to_numpy())
            print(f"  - {market}: RMSE={m_rmse:.6f} (n={len(sdf)})")
        oof_rows.append(
            tmp.rename(columns={"y_true": "target", "y_pred": "pred"})[
                ["id", "delivery_start", "market", "target", "pred", "fold"]
            ].copy()
        )
        oof.loc[va.index] = pred_original

    coverage = float(oof.notna().mean())
    overall = None
    if coverage > 0.0:
        valid = oof.notna()
        overall = _rmse(
            train_df_raw.loc[valid, "target"].to_numpy(dtype=float),
            oof.loc[valid].to_numpy(dtype=float),
        )
    cv_oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    print(f"CV OOF coverage: {coverage:.4%}")
    print(f"CV OOF RMSE: {overall if overall is not None else 'None'}")
    return overall, pd.DataFrame(fold_rows), cv_oof


def train_global_and_local_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    *,
    local_residual_modeling: bool,
    include_global_pred_in_local: bool,
    loss_function: str = "RMSE",
    target_transform: str = "none",
    use_tail_experts: bool = False,
    tail_label_mode: str = "upper",
    tail_quantile_upper: float = 0.99,
    tail_quantile_lower: float = 0.01,
    tail_model_mode: str = "tail_only",
    tail_weight: float = 6.0,
    tail_min_rows: int = 120,
    tail_gate_threshold: float = 0.95,
    tail_blend_mode: str = "soft",
    tail_delta_clip: float | None = 250.0,
    use_hour_experts: bool = False,
    hour_expert_mode: str = "block5",
    hour_expert_min_rows: int = 240,
    hour_expert_weight: float = 0.6,
) -> TrainArtifacts:
    if not (0.0 < tail_quantile_upper < 1.0):
        raise ValueError("--tail-quantile-upper must be in (0,1)")
    if not (0.0 < tail_quantile_lower < 1.0):
        raise ValueError("--tail-quantile-lower must be in (0,1)")
    if tail_quantile_lower >= tail_quantile_upper:
        raise ValueError("--tail-quantile-lower must be < --tail-quantile-upper")
    if not (0.0 <= tail_gate_threshold < 1.0):
        raise ValueError("--tail-gate-threshold must be in [0,1)")
    if hour_expert_min_rows < 1:
        raise ValueError("--hour-expert-min-rows must be >= 1")

    transformed_target, t_state = _fit_target_transform(
        train_df["target"].to_numpy(dtype=float),
        method=target_transform,
    )

    global_model = CatBoostRegressor(
        loss_function=loss_function,
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
    global_model.fit(train_df[feature_cols], transformed_target, cat_features=cat_cols)
    train_df = train_df.copy()
    train_df["target_transformed"] = transformed_target
    train_df["global_pred_feature"] = _predict_point(global_model, train_df[feature_cols])

    local_feature_cols = list(feature_cols)
    if include_global_pred_in_local:
        local_feature_cols = local_feature_cols + ["global_pred_feature"]
    local_models: dict[str, CatBoostRegressor] = {}
    local_tail_models: dict[str, CatBoostRegressor] = {}
    local_gate_models: dict[str, CatBoostClassifier] = {}
    local_hour_models: dict[str, dict[str, CatBoostRegressor]] = {}
    local_hour_model_counts: dict[str, dict[str, int]] = {}
    for market, mdf in train_df.groupby("market", dropna=False):
        normal_model = CatBoostRegressor(
            loss_function=loss_function,
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
        local_target = (
            mdf["target_transformed"] - mdf["global_pred_feature"]
            if local_residual_modeling
            else mdf["target_transformed"]
        )
        local_target_np = local_target.to_numpy(dtype=float)
        local_cat_cols = [c for c in cat_cols if c in local_feature_cols]
        normal_model.fit(mdf[local_feature_cols], local_target_np, cat_features=local_cat_cols)
        local_models[str(market)] = normal_model
        print(f"Trained local model for {market} ({len(mdf)} rows)")

        tail_model_for_market: CatBoostRegressor | None = None
        gate_model_for_market: CatBoostClassifier | None = None
        if use_tail_experts:
            tail_mask = _compute_tail_mask_per_market(
                mdf["target"],
                mdf["market"],
                label_mode=tail_label_mode,
                quantile_upper=tail_quantile_upper,
                quantile_lower=tail_quantile_lower,
            )
            n_tail = int(tail_mask.sum())
            if n_tail < tail_min_rows or n_tail >= len(mdf):
                print(
                    f"Skipped tail expert for {market}: "
                    f"tail_rows={n_tail}, required_min={tail_min_rows}"
                )
            else:
                gate_labels = tail_mask.astype(int)
                if gate_labels.min() == gate_labels.max():
                    print(f"Skipped tail gate for {market}: single-class labels")
                else:
                    gate_model = CatBoostClassifier(
                        loss_function="Logloss",
                        eval_metric="AUC",
                        iterations=900,
                        learning_rate=0.04,
                        depth=6,
                        l2_leaf_reg=12.0,
                        auto_class_weights="Balanced",
                        random_seed=42,
                        verbose=0,
                    )
                    gate_model.fit(mdf[local_feature_cols], gate_labels, cat_features=local_cat_cols)
                    local_gate_models[str(market)] = gate_model

                    tail_model = CatBoostRegressor(
                        loss_function=loss_function,
                        eval_metric="RMSE",
                        iterations=2200,
                        learning_rate=0.03,
                        depth=7,
                        l2_leaf_reg=24.0,
                        bagging_temperature=0.6,
                        random_strength=1.1,
                        random_seed=42,
                        verbose=0,
                    )
                    if tail_model_mode == "tail_only":
                        tail_model.fit(
                            mdf.loc[tail_mask, local_feature_cols],
                            local_target_np[tail_mask],
                            cat_features=local_cat_cols,
                        )
                    elif tail_model_mode == "weighted_all":
                        sample_weight = np.ones(len(mdf), dtype=float)
                        sample_weight[tail_mask] = float(tail_weight)
                        tail_model.fit(
                            mdf[local_feature_cols],
                            local_target_np,
                            cat_features=local_cat_cols,
                            sample_weight=sample_weight,
                        )
                    else:
                        raise ValueError(f"Unsupported tail model mode: {tail_model_mode}")
                    local_tail_models[str(market)] = tail_model
                    tail_model_for_market = tail_model
                    gate_model_for_market = gate_model
                    print(
                        f"Trained tail expert for {market} "
                        f"(tail_rows={n_tail}, mode={tail_model_mode})"
                    )

        # Optional hour-regime residual specialist on top of local prediction.
        if use_hour_experts:
            # Recompute local base prediction (normal + optional tail blend).
            base_local_pred = _predict_point(normal_model, mdf[local_feature_cols])
            if tail_model_for_market is not None and gate_model_for_market is not None:
                tail_pred_for_market = _predict_point(tail_model_for_market, mdf[local_feature_cols])
                tail_prob_for_market = _predict_tail_probability(gate_model_for_market, mdf[local_feature_cols])
                base_local_pred = _blend_tail_expert_prediction(
                    normal_pred=base_local_pred,
                    tail_pred=tail_pred_for_market,
                    tail_prob=tail_prob_for_market,
                    gate_threshold=tail_gate_threshold,
                    blend_mode=tail_blend_mode,
                    delta_clip=tail_delta_clip,
                )

            if local_residual_modeling:
                base_total_pred = mdf["global_pred_feature"].to_numpy(dtype=float) + base_local_pred
            else:
                base_total_pred = base_local_pred
            hour_target = mdf["target_transformed"].to_numpy(dtype=float) - base_total_pred

            regime = _hour_regime_from_hour(mdf["hour"], hour_expert_mode).reset_index(drop=True)
            hour_models_for_market: dict[str, CatBoostRegressor] = {}
            hour_counts_for_market: dict[str, int] = {}
            for reg, pos_idx in regime.groupby(regime, dropna=False).groups.items():
                pos = np.asarray(pos_idx, dtype=int)
                n_reg = int(len(pos))
                if n_reg < hour_expert_min_rows:
                    continue
                reg_model = CatBoostRegressor(
                    loss_function=loss_function,
                    eval_metric="RMSE",
                    iterations=1200,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=26.0,
                    bagging_temperature=0.7,
                    random_strength=1.1,
                    random_seed=42,
                    verbose=0,
                )
                reg_model.fit(
                    mdf.iloc[pos][local_feature_cols],
                    hour_target[pos],
                    cat_features=local_cat_cols,
                )
                hour_models_for_market[str(reg)] = reg_model
                hour_counts_for_market[str(reg)] = n_reg

            if hour_models_for_market:
                local_hour_models[str(market)] = hour_models_for_market
                local_hour_model_counts[str(market)] = hour_counts_for_market
                print(
                    f"Trained hour experts for {market}: "
                    f"{sorted(hour_models_for_market.keys())}"
                )

    return TrainArtifacts(
        global_model=global_model,
        local_models=local_models,
        local_tail_models=local_tail_models,
        local_gate_models=local_gate_models,
        local_hour_models=local_hour_models,
        local_hour_model_counts=local_hour_model_counts,
        feature_cols=local_feature_cols,
        cat_cols=cat_cols,
        local_target_is_residual=local_residual_modeling,
        local_uses_global_pred_feature=include_global_pred_in_local,
        target_transform_state=t_state,
        use_tail_experts=use_tail_experts,
        tail_label_mode=tail_label_mode,
        tail_quantile_upper=tail_quantile_upper,
        tail_quantile_lower=tail_quantile_lower,
        tail_gate_threshold=tail_gate_threshold,
        tail_blend_mode=tail_blend_mode,
        tail_delta_clip=tail_delta_clip,
        tail_model_mode=tail_model_mode,
        tail_weight=tail_weight,
        tail_min_rows=tail_min_rows,
        use_hour_experts=use_hour_experts,
        hour_expert_mode=hour_expert_mode,
        hour_expert_min_rows=hour_expert_min_rows,
        hour_expert_weight=hour_expert_weight,
    )


def build_feature_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    use_dynamic_target_profiles: bool = True,
    use_robust_target_profiles: bool = True,
    profile_winsor_lower_q: float = 0.01,
    profile_winsor_upper_q: float = 0.99,
    profile_winsor_min_rows: int = 24,
    profile_shrink_alpha_mhd: float = 24.0,
    profile_shrink_alpha_m: float = 24.0,
    use_temporal_regime: bool = False,
    use_volatility_regime: bool = False,
    use_wind_proxy: bool = False,
    use_anomaly_features: bool = True,
    use_cross_market_rank_features: bool = True,
    use_peak_interactions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat([train_df.assign(_is_train=1), test_df.assign(_is_train=0)], axis=0, ignore_index=True)
    all_df = add_time_features(all_df)
    if use_temporal_regime:
        all_df = add_temporal_regime_features(all_df)
    all_df = add_forecast_core_features(all_df, use_peak_interactions=use_peak_interactions)
    all_df = add_forecast_lag_features(all_df)
    if use_anomaly_features:
        all_df = add_anomaly_features(all_df)
    if use_volatility_regime:
        all_df = add_volatility_regime_features(all_df)
    all_df = add_meteo_features(all_df, use_peak_interactions=use_peak_interactions)
    if use_wind_proxy:
        all_df = add_wind_proxy_features(all_df)
    all_df = add_cross_market_features(all_df, use_rank_features=use_cross_market_rank_features)
    all_df = add_missingness_features(all_df)
    all_df = add_market_categorical_interactions(all_df)

    train_out = all_df.loc[all_df["_is_train"] == 1].drop(columns=["_is_train"]).copy()
    test_out = all_df.loc[all_df["_is_train"] == 0].drop(columns=["_is_train", "target"], errors="ignore").copy()
    train_out, test_out = add_market_profile_features(
        train_out,
        test_out,
        use_dynamic_profiles=use_dynamic_target_profiles,
        use_robust_profiles=use_robust_target_profiles,
        robust_winsor_lower_q=profile_winsor_lower_q,
        robust_winsor_upper_q=profile_winsor_upper_q,
        robust_winsor_min_rows=profile_winsor_min_rows,
        profile_shrink_alpha_mhd=profile_shrink_alpha_mhd,
        profile_shrink_alpha_m=profile_shrink_alpha_m,
    )
    return train_out, test_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-market intraday training with tailored and cross-market features plus optional hour experts."
    )
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="per_market_interactions_hour_experts")
    parser.add_argument("--exclude-2023", action="store_true")
    parser.add_argument("--exclude-2023-keep-from-month", type=int, default=10)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-val-days", type=int, default=14)
    parser.add_argument("--cv-step-days", type=int, default=14)
    parser.add_argument("--cv-min-train-days", type=int, default=90)
    parser.add_argument(
        "--use-dynamic-target-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable lag-safe dynamic target profiles including dynamic squared-mean and RMS features (default: enabled).",
    )
    parser.add_argument(
        "--use-robust-target-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use winsorized targets + market-hour shrinkage in target_profile_* features (default: enabled).",
    )
    parser.add_argument(
        "--profile-winsor-lower-q",
        type=float,
        default=0.01,
        help="Lower quantile for robust target-profile winsorization (default: 0.01).",
    )
    parser.add_argument(
        "--profile-winsor-upper-q",
        type=float,
        default=0.99,
        help="Upper quantile for robust target-profile winsorization (default: 0.99).",
    )
    parser.add_argument(
        "--profile-winsor-min-rows",
        type=int,
        default=24,
        help="Minimum market-hour rows before using market-hour winsor bounds (default: 24).",
    )
    parser.add_argument(
        "--profile-shrink-alpha-mhd",
        type=float,
        default=24.0,
        help="Shrinkage alpha for market-hour-dow profiles toward market-hour baseline (default: 24).",
    )
    parser.add_argument(
        "--profile-shrink-alpha-m",
        type=float,
        default=24.0,
        help="Shrinkage alpha for market profile toward market-hour baseline (default: 24).",
    )
    parser.add_argument(
        "--use-temporal-regime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable holiday/month-end/DST/day-of-year regime features (default: disabled).",
    )
    parser.add_argument(
        "--use-volatility-regime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable 7d/28d volatility regime features (default: disabled).",
    )
    parser.add_argument(
        "--use-wind-proxy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable wind proxy model-derived features (default: disabled).",
    )
    parser.add_argument(
        "--use-anomaly-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable 24h/168h anomaly and z-score features (default: enabled).",
    )
    parser.add_argument(
        "--use-cross-market-rank-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable same-timestamp cross-market percentile rank features (default: enabled).",
    )
    parser.add_argument(
        "--use-peak-interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable peak/daytime interaction features (default: enabled).",
    )
    parser.add_argument(
        "--local-residual-modeling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train local market models on residuals (target-global_pred) and add back global prediction (default: enabled).",
    )
    parser.add_argument(
        "--include-global-pred-in-local",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include global_pred_feature as local-model input feature (default: disabled).",
    )
    parser.add_argument(
        "--loss-function",
        choices=["RMSE", "RMSEWithUncertainty"],
        default="RMSE",
        help="CatBoost training loss for global and local models (default: RMSE).",
    )
    parser.add_argument(
        "--target-transform",
        choices=["none", "signed_log", "log_shift", "yeo_johnson"],
        default="none",
        help="Transform target before training and invert predictions before evaluation/submission (default: none).",
    )
    parser.add_argument(
        "--use-tail-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable local mixture-of-experts with a tail gate and tail regressor (default: disabled).",
    )
    parser.add_argument(
        "--tail-label-mode",
        choices=["upper", "two_sided"],
        default="upper",
        help="Tail event definition per market for the gate model.",
    )
    parser.add_argument(
        "--tail-quantile-upper",
        type=float,
        default=0.99,
        help="Upper quantile per market used to define tail events (default: 0.99).",
    )
    parser.add_argument(
        "--tail-quantile-lower",
        type=float,
        default=0.01,
        help="Lower quantile per market used when --tail-label-mode=two_sided (default: 0.01).",
    )
    parser.add_argument(
        "--tail-model-mode",
        choices=["tail_only", "weighted_all"],
        default="tail_only",
        help="How to train the tail expert: on tail subset only or all rows with tail upweights.",
    )
    parser.add_argument(
        "--tail-weight",
        type=float,
        default=6.0,
        help="Tail sample weight when --tail-model-mode=weighted_all (default: 6.0).",
    )
    parser.add_argument(
        "--tail-min-rows",
        type=int,
        default=120,
        help="Minimum tail rows in a market to train tail gate/expert (default: 120).",
    )
    parser.add_argument(
        "--tail-gate-threshold",
        type=float,
        default=0.95,
        help="High-precision gate threshold for tail activation (default: 0.95).",
    )
    parser.add_argument(
        "--tail-blend-mode",
        choices=["soft", "hard"],
        default="soft",
        help="Tail blending rule: probability-weighted soft blend or hard switch.",
    )
    parser.add_argument(
        "--tail-delta-clip",
        type=float,
        default=250.0,
        help="Optional cap on tail minus normal adjustment in transformed space (<=0 disables clip).",
    )
    parser.add_argument(
        "--use-hour-experts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable additional per-market hour-of-day residual experts (default: disabled).",
    )
    parser.add_argument(
        "--hour-expert-mode",
        choices=["block5", "hour"],
        default="block5",
        help="Hour expert grouping: block5 regimes or full hour buckets.",
    )
    parser.add_argument(
        "--hour-expert-min-rows",
        type=int,
        default=240,
        help="Minimum rows required to train a market-hour expert (default: 240).",
    )
    parser.add_argument(
        "--hour-expert-weight",
        type=float,
        default=0.6,
        help="Global scaling applied to hour-expert correction (default: 0.6).",
    )
    parser.add_argument(
        "--use-market-hour-pred-caps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply market-hour soft prediction caps from train quantiles (default: enabled).",
    )
    parser.add_argument(
        "--pred-cap-lower-q",
        type=float,
        default=0.01,
        help="Lower quantile for market-hour prediction cap fitting (default: 0.01).",
    )
    parser.add_argument(
        "--pred-cap-upper-q",
        type=float,
        default=0.99,
        help="Upper quantile for market-hour prediction cap fitting (default: 0.99).",
    )
    parser.add_argument(
        "--pred-cap-min-rows",
        type=int,
        default=24,
        help="Minimum rows for market-hour cap; otherwise fallback to market/global caps (default: 24).",
    )
    parser.add_argument(
        "--pred-cap-softness",
        type=float,
        default=0.10,
        help="Soft-cap slope in [0,1]; 0 is hard clip, 1 keeps extremes unchanged (default: 0.10).",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save fitted global/local CatBoost models and metadata in the run directory (default: enabled).",
    )
    parser.add_argument(
        "--save-shap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and save SHAP outputs after training (default: enabled).",
    )
    parser.add_argument(
        "--shap-global-sample-size",
        type=int,
        default=1500,
        help="Sample size for global SHAP outputs (default: 1500).",
    )
    parser.add_argument(
        "--shap-per-market-sample-size",
        type=int,
        default=400,
        help="Sample size per market for local SHAP outputs (default: 400).",
    )
    args = parser.parse_args()
    tail_delta_clip = args.tail_delta_clip if args.tail_delta_clip > 0 else None

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample = pd.read_csv(args.sample_submission)

    if args.exclude_2023:
        train_df = apply_exclude_2023(train_df, keep_from_month=args.exclude_2023_keep_from_month)

    cv_rmse = None
    cv_details = pd.DataFrame()
    cv_oof = pd.DataFrame()
    if args.cv:
        cv_rmse, cv_details, cv_oof = run_time_series_cv(
            train_df_raw=train_df,
            n_folds=args.cv_folds,
            val_days=args.cv_val_days,
            step_days=args.cv_step_days,
            min_train_days=args.cv_min_train_days,
            use_dynamic_target_profiles=args.use_dynamic_target_profiles,
            use_robust_target_profiles=args.use_robust_target_profiles,
            profile_winsor_lower_q=args.profile_winsor_lower_q,
            profile_winsor_upper_q=args.profile_winsor_upper_q,
            profile_winsor_min_rows=args.profile_winsor_min_rows,
            profile_shrink_alpha_mhd=args.profile_shrink_alpha_mhd,
            profile_shrink_alpha_m=args.profile_shrink_alpha_m,
            use_temporal_regime=args.use_temporal_regime,
            use_volatility_regime=args.use_volatility_regime,
            use_wind_proxy=args.use_wind_proxy,
            use_anomaly_features=args.use_anomaly_features,
            use_cross_market_rank_features=args.use_cross_market_rank_features,
            use_peak_interactions=args.use_peak_interactions,
            local_residual_modeling=args.local_residual_modeling,
            include_global_pred_in_local=args.include_global_pred_in_local,
            loss_function=args.loss_function,
            target_transform=args.target_transform,
            use_tail_experts=args.use_tail_experts,
            tail_label_mode=args.tail_label_mode,
            tail_quantile_upper=args.tail_quantile_upper,
            tail_quantile_lower=args.tail_quantile_lower,
            tail_model_mode=args.tail_model_mode,
            tail_weight=args.tail_weight,
            tail_min_rows=args.tail_min_rows,
            tail_gate_threshold=args.tail_gate_threshold,
            tail_blend_mode=args.tail_blend_mode,
            tail_delta_clip=tail_delta_clip,
            use_hour_experts=args.use_hour_experts,
            hour_expert_mode=args.hour_expert_mode,
            hour_expert_min_rows=args.hour_expert_min_rows,
            hour_expert_weight=args.hour_expert_weight,
            use_market_hour_pred_caps=args.use_market_hour_pred_caps,
            pred_cap_lower_q=args.pred_cap_lower_q,
            pred_cap_upper_q=args.pred_cap_upper_q,
            pred_cap_min_rows=args.pred_cap_min_rows,
            pred_cap_softness=args.pred_cap_softness,
        )

    train_feat, test_feat = build_feature_table(
        train_df,
        test_df,
        use_dynamic_target_profiles=args.use_dynamic_target_profiles,
        use_robust_target_profiles=args.use_robust_target_profiles,
        profile_winsor_lower_q=args.profile_winsor_lower_q,
        profile_winsor_upper_q=args.profile_winsor_upper_q,
        profile_winsor_min_rows=args.profile_winsor_min_rows,
        profile_shrink_alpha_mhd=args.profile_shrink_alpha_mhd,
        profile_shrink_alpha_m=args.profile_shrink_alpha_m,
        use_temporal_regime=args.use_temporal_regime,
        use_volatility_regime=args.use_volatility_regime,
        use_wind_proxy=args.use_wind_proxy,
        use_anomaly_features=args.use_anomaly_features,
        use_cross_market_rank_features=args.use_cross_market_rank_features,
        use_peak_interactions=args.use_peak_interactions,
    )
    test_with_key = test_feat[["id", "market"]].copy()

    base_drop = {"id", "target", "delivery_start", "delivery_end"}
    candidate_features = [c for c in train_feat.columns if c not in base_drop]
    cat_cols = [
        c
        for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
        if c in candidate_features
    ]

    artifacts = train_global_and_local_models(
        train_feat,
        candidate_features,
        cat_cols,
        local_residual_modeling=args.local_residual_modeling,
        include_global_pred_in_local=args.include_global_pred_in_local,
        loss_function=args.loss_function,
        target_transform=args.target_transform,
        use_tail_experts=args.use_tail_experts,
        tail_label_mode=args.tail_label_mode,
        tail_quantile_upper=args.tail_quantile_upper,
        tail_quantile_lower=args.tail_quantile_lower,
        tail_model_mode=args.tail_model_mode,
        tail_weight=args.tail_weight,
        tail_min_rows=args.tail_min_rows,
        tail_gate_threshold=args.tail_gate_threshold,
        tail_blend_mode=args.tail_blend_mode,
        tail_delta_clip=tail_delta_clip,
        use_hour_experts=args.use_hour_experts,
        hour_expert_mode=args.hour_expert_mode,
        hour_expert_min_rows=args.hour_expert_min_rows,
        hour_expert_weight=args.hour_expert_weight,
    )

    # Add global prediction feature to test and run local market experts.
    test_feat = test_feat.copy()
    test_feat["global_pred_feature"] = _predict_point(artifacts.global_model, test_feat[candidate_features])

    pred = np.full(len(test_feat), np.nan, dtype=float)
    for market, idx in test_with_key.groupby("market", dropna=False).groups.items():
        normal_model = artifacts.local_models.get(str(market))
        if normal_model is None:
            # Fallback to global if market-specific model doesn't exist.
            pred[idx] = test_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float)
            continue
        local_pred = _predict_local_expert(
            artifacts=artifacts,
            market=str(market),
            X=test_feat.loc[idx],
        )
        if artifacts.local_target_is_residual:
            pred[idx] = test_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float) + local_pred
        else:
            pred[idx] = local_pred

    if np.isnan(pred).any():
        raise ValueError("NaNs found in predictions.")

    pred = _apply_target_inverse(pred, artifacts.target_transform_state)
    if args.use_market_hour_pred_caps:
        capper = fit_market_hour_prediction_capper(
            train_feat,
            lower_q=args.pred_cap_lower_q,
            upper_q=args.pred_cap_upper_q,
            min_rows=args.pred_cap_min_rows,
            softness=args.pred_cap_softness,
        )
        pred, n_capped = apply_market_hour_prediction_caps(
            pred,
            test_feat[["market", "hour"]],
            capper,
        )
        print(
            "Applied market-hour prediction caps: "
            f"adjusted_rows={n_capped} ({(100.0 * n_capped / max(len(pred), 1)):.2f}%), "
            f"q=({args.pred_cap_lower_q:.3f},{args.pred_cap_upper_q:.3f}), "
            f"min_rows={args.pred_cap_min_rows}, softness={args.pred_cap_softness:.3f}"
        )

    out_sub = sample[["id"]].copy()
    pred_map = pd.Series(pred, index=test_feat["id"].astype(int))
    out_sub["target"] = out_sub["id"].astype(int).map(pred_map)
    if out_sub["target"].isna().any():
        raise ValueError("Submission has NaN targets after mapping.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / f"{stamp}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    params_path = run_dir / "params.yaml"
    _write_params_yaml(
        params_path,
        {
            "run_id": run_dir.name,
            "name": args.name,
            "generated_at": datetime.now().isoformat(),
            "cv_rmse": None if cv_rmse is None else float(cv_rmse),
            "target_transform": artifacts.target_transform_state.to_metadata(),
            "args": vars(args),
        },
    )

    sub_path = run_dir / "submission.csv"
    out_sub.to_csv(sub_path, index=False)

    latest_path = Path("submission_per_market_interactions.csv")
    out_sub.to_csv(latest_path, index=False)

    print(f"Saved submission: {sub_path}")
    print(f"Saved latest copy: {latest_path}")
    print(f"Saved params: {params_path}")
    print(f"Features used: {len(artifacts.feature_cols)}")
    print(f"Categorical features: {artifacts.cat_cols}")
    print(f"Markets modeled: {sorted(artifacts.local_models.keys())}")
    print(f"Local target is residual: {artifacts.local_target_is_residual}")
    print(f"Local uses global_pred_feature: {artifacts.local_uses_global_pred_feature}")
    print(f"Tail experts enabled: {artifacts.use_tail_experts}")
    if artifacts.use_tail_experts:
        print(
            "Tail expert settings: "
            f"label_mode={artifacts.tail_label_mode}, "
            f"q_upper={artifacts.tail_quantile_upper}, "
            f"q_lower={artifacts.tail_quantile_lower}, "
            f"gate_tau={artifacts.tail_gate_threshold}, "
            f"blend={artifacts.tail_blend_mode}, "
            f"delta_clip={artifacts.tail_delta_clip}, "
            f"mode={artifacts.tail_model_mode}, "
            f"tail_weight={artifacts.tail_weight}, "
            f"tail_min_rows={artifacts.tail_min_rows}"
        )
        print(f"Tail models trained: {len(artifacts.local_tail_models)}")
        print(f"Gate models trained: {len(artifacts.local_gate_models)}")
    print(f"Hour experts enabled: {artifacts.use_hour_experts}")
    if artifacts.use_hour_experts:
        print(
            "Hour expert settings: "
            f"mode={artifacts.hour_expert_mode}, "
            f"min_rows={artifacts.hour_expert_min_rows}, "
            f"weight={artifacts.hour_expert_weight}"
        )
        print(f"Hour expert markets trained: {len(artifacts.local_hour_models)}")
    print(f"Robust target profiles enabled: {args.use_robust_target_profiles}")
    if args.use_robust_target_profiles:
        print(
            "Robust target profile settings: "
            f"winsor_q=({args.profile_winsor_lower_q},{args.profile_winsor_upper_q}), "
            f"winsor_min_rows={args.profile_winsor_min_rows}, "
            f"alpha_mhd={args.profile_shrink_alpha_mhd}, "
            f"alpha_m={args.profile_shrink_alpha_m}"
        )
    print(f"Market-hour prediction caps enabled: {args.use_market_hour_pred_caps}")
    print(f"Target transform: {artifacts.target_transform_state.to_metadata()}")
    print(f"CV RMSE: {cv_rmse}")
    if not cv_details.empty:
        cv_path = run_dir / "cv_results.csv"
        cv_details.to_csv(cv_path, index=False)
        print(f"Saved CV details: {cv_path}")
    if not cv_oof.empty:
        cv_oof_path = run_dir / "cv_oof.csv"
        cv_oof.to_csv(cv_oof_path, index=False)
        print(f"Saved CV OOF rows: {cv_oof_path}")

    if args.save_models:
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        global_model_path = models_dir / "global_model.cbm"
        artifacts.global_model.save_model(global_model_path)

        local_model_paths: dict[str, str] = {}
        for market, model in artifacts.local_models.items():
            safe_market = str(market).replace(" ", "_")
            local_path = models_dir / f"local_model_{safe_market}.cbm"
            model.save_model(local_path)
            local_model_paths[str(market)] = str(local_path.name)

        local_tail_model_paths: dict[str, str] = {}
        for market, model in artifacts.local_tail_models.items():
            safe_market = str(market).replace(" ", "_")
            local_path = models_dir / f"local_tail_model_{safe_market}.cbm"
            model.save_model(local_path)
            local_tail_model_paths[str(market)] = str(local_path.name)

        local_gate_model_paths: dict[str, str] = {}
        for market, model in artifacts.local_gate_models.items():
            safe_market = str(market).replace(" ", "_")
            local_path = models_dir / f"local_gate_model_{safe_market}.cbm"
            model.save_model(local_path)
            local_gate_model_paths[str(market)] = str(local_path.name)

        local_hour_model_paths: dict[str, dict[str, str]] = {}
        for market, models_by_regime in artifacts.local_hour_models.items():
            safe_market = str(market).replace(" ", "_")
            local_hour_model_paths[str(market)] = {}
            for regime, model in models_by_regime.items():
                safe_regime = (
                    str(regime)
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("-", "_")
                )
                local_path = models_dir / f"local_hour_model_{safe_market}_{safe_regime}.cbm"
                model.save_model(local_path)
                local_hour_model_paths[str(market)][str(regime)] = str(local_path.name)

        model_meta = {
            "global_model": str(global_model_path.name),
            "local_models": local_model_paths,
            "local_tail_models": local_tail_model_paths,
            "local_gate_models": local_gate_model_paths,
            "local_hour_models": local_hour_model_paths,
            "local_hour_model_counts": artifacts.local_hour_model_counts,
            "feature_cols": artifacts.feature_cols,
            "cat_cols": artifacts.cat_cols,
            "local_target_is_residual": artifacts.local_target_is_residual,
            "local_uses_global_pred_feature": artifacts.local_uses_global_pred_feature,
            "tail_expert_config": {
                "use_tail_experts": artifacts.use_tail_experts,
                "tail_label_mode": artifacts.tail_label_mode,
                "tail_quantile_upper": artifacts.tail_quantile_upper,
                "tail_quantile_lower": artifacts.tail_quantile_lower,
                "tail_gate_threshold": artifacts.tail_gate_threshold,
                "tail_blend_mode": artifacts.tail_blend_mode,
                "tail_delta_clip": artifacts.tail_delta_clip,
                "tail_model_mode": artifacts.tail_model_mode,
                "tail_weight": artifacts.tail_weight,
                "tail_min_rows": artifacts.tail_min_rows,
            },
            "hour_expert_config": {
                "use_hour_experts": artifacts.use_hour_experts,
                "hour_expert_mode": artifacts.hour_expert_mode,
                "hour_expert_min_rows": artifacts.hour_expert_min_rows,
                "hour_expert_weight": artifacts.hour_expert_weight,
            },
            "target_transform": artifacts.target_transform_state.to_metadata(),
            "candidate_features_before_global_pred": candidate_features,
            "train_args": vars(args),
        }
        model_meta_path = run_dir / "model_metadata.json"
        model_meta_path.write_text(json.dumps(model_meta, indent=2))

        print(f"Saved models dir: {models_dir}")
        print(f"Saved model metadata: {model_meta_path}")

        if args.save_shap:
            if artifacts.use_tail_experts or artifacts.use_hour_experts:
                print(
                    "Note: SHAP export uses saved normal experts for local attributions; "
                    "gate/tail/hour blending logic is not decomposed in the SHAP output."
                )
            shap_cmd = [
                Path(__file__).with_name("shap_from_saved_models.py").as_posix(),
                "--run-dir",
                str(run_dir),
                "--global-sample-size",
                str(args.shap_global_sample_size),
                "--per-market-sample-size",
                str(args.shap_per_market_sample_size),
            ]
            print(
                "Saving SHAP outputs "
                f"(global_sample={args.shap_global_sample_size}, "
                f"per_market_sample={args.shap_per_market_sample_size})..."
            )
            subprocess.run(
                [Path(__file__).resolve().parent.joinpath(".venv/bin/python").as_posix(), *shap_cmd]
                if Path(__file__).resolve().parent.joinpath(".venv/bin/python").exists()
                else ["python3", *shap_cmd],
                check=True,
            )
    elif args.save_shap:
        print("Skipping SHAP: --save-shap requires saved models. Use --save-models.")


if __name__ == "__main__":
    main()
