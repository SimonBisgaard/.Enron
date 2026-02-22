from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error


REDUNDANT_FEATURE_DROP_LIST = [
    # Forecast/residual x-market means duplicate base forecasts in this dataset.
    "load_forecast_xmk_mean",
    "wind_forecast_xmk_mean",
    "solar_forecast_xmk_mean",
    "residual_load_xmk_mean",
    # These x-market std features are constant zero.
    "load_forecast_xmk_std",
    "wind_forecast_xmk_std",
    "solar_forecast_xmk_std",
    "residual_load_xmk_std",
    # These collapse to near-zero noise-only values for global forecasts.
    "load_forecast_xmk_diff",
    "wind_forecast_xmk_diff",
    "solar_forecast_xmk_diff",
    "residual_load_xmk_diff",
    "load_forecast_xmk_z",
    "wind_forecast_xmk_z",
    "solar_forecast_xmk_z",
    "residual_load_xmk_z",
    # Exact duplicate indicator.
    "is_evening_peak",
]

CORE_FORECAST_FEATURES = {
    "load_forecast",
    "wind_forecast",
    "solar_forecast",
}

KEY_METEO_FEATURES = {
    "wind_speed_80m",
    "wind_speed_10m",
    "wind_gust_speed_10m",
    "wind_direction_80m",
    "air_temperature_2m",
    "dew_point_temperature_2m",
    "apparent_temperature_2m",
    "relative_humidity_2m",
    "cloud_cover_total",
    "cloud_cover_low",
    "precipitation_amount",
    "global_horizontal_irradiance",
    "direct_normal_irradiance",
    "diffuse_horizontal_irradiance",
    "convective_available_potential_energy",
    "convective_inhibition",
    "freezing_level_height",
    "gustiness_10m",
    "gust_ratio_10m",
    "ws_ratio_80_10",
    "wind_dir_sin",
    "wind_dir_cos",
    "clear_sky_proxy",
    "cloud_low_share",
    "temp_dew_spread",
    "temp_apparent_gap",
}

ROBUST_POSITIVE_FEATURES = {
    "solar_forecast_diff_1",
    "wind_forecast_diff_1",
    "wind_forecast_diff_6",
    "wind_forecast_lag_6",
    "solar_forecast_roll_mean_6",
    "wind_forecast_roll_mean_24",
    "temp_apparent_gap",
    "convective_available_potential_energy",
}


DEFAULT_2C02EB6_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "global_model": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 18.0,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "random_seed": 42,
        "verbose": 0,
    },
    "local_model": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3000,
        "learning_rate": 0.025,
        "depth": 8,
        "l2_leaf_reg": 20.0,
        "bagging_temperature": 0.4,
        "random_strength": 0.9,
        "random_seed": 42,
        "verbose": 0,
    },
}

TUNED_2C02EB6_TRIAL1_MODEL_PARAM_OVERRIDES: dict[str, dict[str, Any]] = {
    "global_model": {
        "iterations": 2700,
        "learning_rate": 0.071417442434829,
        "depth": 9,
        "l2_leaf_reg": 24.03989887352124,
        "bagging_temperature": 0.3900466011060913,
        "random_strength": 0.4743868488068863,
    },
    "local_model": {
        "iterations": 1800,
        "learning_rate": 0.05878494724216357,
        "depth": 9,
        "l2_leaf_reg": 39.07535901228298,
        "bagging_temperature": 0.05146123573950612,
        "random_strength": 2.427783645188786,
    },
}


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


def add_temperature_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "air_temperature_2m" not in out.columns:
        return out

    temp = out["air_temperature_2m"]
    out["hdd_18"] = (18.0 - temp).clip(lower=0.0)
    out["cdd_22"] = (temp - 22.0).clip(lower=0.0)
    out["temp_extreme_mag"] = out["hdd_18"] + out["cdd_22"]
    out["temp_extreme_flag"] = ((temp <= 2.0) | (temp >= 28.0)).astype(int)
    return out


def add_physics_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "wind_speed_80m" in out.columns:
        out["wind_speed_80m_cubed"] = out["wind_speed_80m"] ** 3
        if {"market", "delivery_start"}.issubset(out.columns):
            tmp = out[["market", "delivery_start", "wind_speed_80m"]].copy()
            tmp["_row_id"] = np.arange(len(tmp))
            tmp = tmp.sort_values(["market", "delivery_start", "_row_id"])
            g_ws = tmp.groupby("market")["wind_speed_80m"]
            tmp["wind_ramp_proxy"] = tmp["wind_speed_80m"] - g_ws.shift(1)
            tmp = tmp.sort_values("_row_id")
            out["wind_ramp_proxy"] = tmp["wind_ramp_proxy"].to_numpy(dtype=float)

    if "residual_load_diff_6" in out.columns:
        out["residual_load_ramp_abs_6"] = out["residual_load_diff_6"].abs()
    if "residual_load_diff_24" in out.columns:
        out["residual_load_ramp_abs_24"] = out["residual_load_diff_24"].abs()

    if {"diffuse_horizontal_irradiance", "direct_normal_irradiance"}.issubset(out.columns):
        out["cloud_regime_ratio"] = out["diffuse_horizontal_irradiance"] / (
            out["direct_normal_irradiance"] + 1e-6
        )
    if {
        "global_horizontal_irradiance",
        "direct_normal_irradiance",
        "diffuse_horizontal_irradiance",
    }.issubset(out.columns):
        out["clear_sky_error_proxy"] = out["global_horizontal_irradiance"] - (
            out["direct_normal_irradiance"] + out["diffuse_horizontal_irradiance"]
        )

    if "air_temperature_2m" in out.columns:
        out["temp_gap_18"] = (out["air_temperature_2m"] - 18.0).abs()

    if {"cape", "convective_inhibition"}.issubset(out.columns):
        out["storm_score"] = out["cape"] / (1.0 + out["convective_inhibition"].abs())

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


def apply_train_start_cutoff(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    out = df.copy()
    start = _to_datetime_col(out, "delivery_start")
    cutoff = pd.Timestamp(start_date)
    keep_mask = start >= cutoff
    removed = int((~keep_mask).sum())
    kept = int(keep_mask.sum())
    print(
        "Train start cutoff mode: "
        f"start_date={cutoff.date()}, removed={removed}, kept={kept}"
    )
    return out.loc[keep_mask].copy()


@dataclass
class TrainArtifacts:
    global_model: CatBoostRegressor
    local_models: dict[str, CatBoostRegressor]
    feature_cols: list[str]
    cat_cols: list[str]
    local_target_is_residual: bool


@dataclass
class PermutationEvalOutputs:
    perm_importance_by_fold: pd.DataFrame
    perm_importance_summary: pd.DataFrame
    corr_groups: pd.DataFrame
    perm_importance_groups_by_fold: pd.DataFrame
    perm_importance_groups_summary: pd.DataFrame
    perm_importance_families: pd.DataFrame
    perm_importance_families_summary: pd.DataFrame
    runtime_seconds: float
    n_features_evaluated: int
    n_groups_evaluated: int
    n_family_evaluations: int


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "?"
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_bar(completed: int, total: int, width: int = 24) -> str:
    safe_total = max(total, 1)
    ratio = min(max(completed / safe_total, 0.0), 1.0)
    filled = int(round(ratio * width))
    filled = min(max(filled, 0), width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


@dataclass
class LoopProgress:
    label: str
    total: int
    min_interval_seconds: float = 1.0
    start_time: float = field(default_factory=time.perf_counter)
    last_emit_time: float = 0.0

    def update(self, completed: int, *, force: bool = False) -> None:
        done = min(max(completed, 0), max(self.total, 0))
        now = time.perf_counter()
        should_emit = force or done >= self.total or (now - self.last_emit_time) >= self.min_interval_seconds
        if not should_emit:
            return

        elapsed = now - self.start_time
        pct = (100.0 * done / self.total) if self.total > 0 else 100.0
        rate = (done / elapsed) if elapsed > 0 else 0.0
        remaining = max(self.total - done, 0)
        eta = (remaining / rate) if rate > 1e-12 else float("nan")
        msg = (
            f"{self.label} {_progress_bar(done, self.total)} "
            f"{done}/{self.total} ({pct:5.1f}%) "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}"
        )

        if sys.stdout.isatty() and not force and done < self.total:
            print(msg, end="\r", flush=True)
        else:
            print(msg)
        self.last_emit_time = now

    def finish(self) -> None:
        self.update(self.total, force=True)


def _stable_hash_int(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _permutation_seed(base_seed: int, fold_index: int, name: str) -> int:
    return int(base_seed + fold_index * 1000 + (_stable_hash_int(name) % 1_000_000_000))


def _predict_with_global_local_pipeline(
    feat_df: pd.DataFrame,
    *,
    artifacts: TrainArtifacts,
    global_feature_cols: list[str],
) -> np.ndarray:
    out = feat_df.copy()
    out["global_pred_feature"] = artifacts.global_model.predict(out[global_feature_cols])

    pred = np.full(len(out), np.nan, dtype=float)
    key = out[["market"]].copy()
    for market, idx in key.groupby("market", dropna=False).groups.items():
        model = artifacts.local_models.get(str(market))
        if model is None:
            pred[idx] = out.loc[idx, "global_pred_feature"].to_numpy(dtype=float)
            continue

        local_pred = model.predict(out.loc[idx, artifacts.feature_cols])
        if artifacts.local_target_is_residual:
            pred[idx] = out.loc[idx, "global_pred_feature"].to_numpy(dtype=float) + local_pred
        else:
            pred[idx] = local_pred
    if np.isnan(pred).any():
        raise ValueError("NaNs found in global/local inference pipeline.")
    return pred


def _permute_columns_with_index(
    feat_df: pd.DataFrame,
    columns: list[str],
    perm_idx: np.ndarray,
) -> pd.DataFrame:
    out = feat_df.copy()
    for col in columns:
        out[col] = out[col].to_numpy()[perm_idx]
    return out


def _build_correlation_groups(
    train_feat: pd.DataFrame,
    *,
    feature_cols: list[str],
    cat_cols: list[str],
    corr_threshold: float,
) -> tuple[list[tuple[str, list[str]]], list[dict[str, object]]]:
    numeric_cols = [
        c
        for c in feature_cols
        if c not in cat_cols and pd.api.types.is_numeric_dtype(train_feat[c])
    ]
    if not numeric_cols:
        return [], []

    corr = train_feat[numeric_cols].corr(method="pearson").abs().fillna(0.0)
    visited: set[str] = set()
    groups: list[tuple[str, list[str]]] = []
    corr_group_rows: list[dict[str, object]] = []

    for feature in numeric_cols:
        if feature in visited:
            continue
        stack = [feature]
        component: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            neighbors = corr.columns[(corr.loc[cur] >= corr_threshold) & (corr.columns != cur)].tolist()
            for nxt in neighbors:
                if nxt not in visited:
                    stack.append(nxt)

        component_sorted = sorted(component)
        key = "|".join(component_sorted)
        group_id = f"corr_{_stable_hash_int(key) % 10_000_000_000:010d}"
        groups.append((group_id, component_sorted))

        group_size = int(len(component_sorted))
        for col in component_sorted:
            corr_group_rows.append(
                {
                    "group_id": group_id,
                    "feature": col,
                    "group_size": group_size,
                    "corr_threshold": float(corr_threshold),
                }
            )
    return groups, corr_group_rows


def _feature_family_map(feature_cols: list[str]) -> dict[str, list[str]]:
    patterns = {
        "lags": re.compile(r"_lag_|_diff_|_roll_", flags=re.IGNORECASE),
        "xmk": re.compile(r"_xmk_", flags=re.IGNORECASE),
        "ratios_shares": re.compile(r"ratio|share", flags=re.IGNORECASE),
        "meteo": re.compile(
            r"wind|solar|irradiance|cloud|temperature|dew|precip|snow|pressure|humidity|cape|convective|gust|apparent",
            flags=re.IGNORECASE,
        ),
        "profiles": re.compile(r"target_profile_", flags=re.IGNORECASE),
    }
    out: dict[str, list[str]] = {}
    for family_name, pattern in patterns.items():
        out[family_name] = [c for c in feature_cols if pattern.search(c)]
    return out


# Interpretation guide for saved deltas:
# - Positive mean_delta: robust OOS signal.
# - Negative mean_delta: likely noise-fit/overfit behavior.
# - Large std_delta: unstable effect across folds.
# - Single-feature delta near 0 but group delta > 0: correlation masking/redundancy.
# - Group delta < 0: the correlated information cluster is harmful out-of-sample.
def _summarize_single_feature_importance(perm_rows: pd.DataFrame) -> pd.DataFrame:
    out_cols = [
        "feature",
        "mean_delta",
        "median_delta",
        "std_delta",
        "positive_fold_count",
        "n_folds",
    ]
    if perm_rows.empty:
        return pd.DataFrame(columns=out_cols)

    grouped = perm_rows.groupby("feature", dropna=False)
    out = grouped["delta"].agg(["mean", "median", "std", "count"]).reset_index()
    out = out.rename(
        columns={
            "mean": "mean_delta",
            "median": "median_delta",
            "std": "std_delta",
            "count": "n_folds",
        }
    )
    out["positive_fold_count"] = grouped["delta"].apply(lambda s: int((s > 0).sum())).to_numpy()
    out["std_delta"] = out["std_delta"].fillna(0.0)
    out["positive_fold_count"] = out["positive_fold_count"].astype(int)
    out["n_folds"] = out["n_folds"].astype(int)
    return out[out_cols].sort_values("mean_delta", ascending=False).reset_index(drop=True)


def _summarize_group_importance(perm_group_rows: pd.DataFrame) -> pd.DataFrame:
    out_cols = [
        "group_id",
        "group_size",
        "mean_delta",
        "median_delta",
        "std_delta",
        "positive_fold_count",
    ]
    if perm_group_rows.empty:
        return pd.DataFrame(columns=out_cols)

    grouped = perm_group_rows.groupby("group_id", dropna=False)
    out = grouped.agg(
        group_size=("group_size", "max"),
        mean_delta=("delta", "mean"),
        median_delta=("delta", "median"),
        std_delta=("delta", "std"),
        positive_fold_count=("delta", lambda s: int((s > 0).sum())),
    ).reset_index()
    out["std_delta"] = out["std_delta"].fillna(0.0)
    out["group_size"] = out["group_size"].astype(int)
    out["positive_fold_count"] = out["positive_fold_count"].astype(int)
    return out[out_cols].sort_values("mean_delta", ascending=False).reset_index(drop=True)


def _summarize_family_importance(perm_family_rows: pd.DataFrame) -> pd.DataFrame:
    out_cols = [
        "family_name",
        "mean_delta",
        "median_delta",
        "std_delta",
        "positive_fold_count",
        "n_folds",
    ]
    if perm_family_rows.empty:
        return pd.DataFrame(columns=out_cols)

    grouped = perm_family_rows.groupby("family_name", dropna=False)
    out = grouped["delta"].agg(["mean", "median", "std", "count"]).reset_index()
    out = out.rename(
        columns={
            "mean": "mean_delta",
            "median": "median_delta",
            "std": "std_delta",
            "count": "n_folds",
        }
    )
    out["positive_fold_count"] = grouped["delta"].apply(lambda s: int((s > 0).sum())).to_numpy()
    out["std_delta"] = out["std_delta"].fillna(0.0)
    out["positive_fold_count"] = out["positive_fold_count"].astype(int)
    out["n_folds"] = out["n_folds"].astype(int)
    return out[out_cols].sort_values("mean_delta", ascending=False).reset_index(drop=True)


def load_2c02eb6_model_params(params_in: str | None) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = copy.deepcopy(DEFAULT_2C02EB6_MODEL_PARAMS)
    if not params_in:
        return params
    path = Path(params_in)
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter JSON file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Hyperparameter JSON must be a top-level object.")
    for section in ["global_model", "local_model"]:
        override = payload.get(section, {})
        if isinstance(override, dict):
            params[section].update(override)
    return params


def _build_tuning_defaults_from_params(model_params: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        "global_iterations": int(model_params["global_model"]["iterations"]),
        "global_learning_rate": float(model_params["global_model"]["learning_rate"]),
        "global_depth": int(model_params["global_model"]["depth"]),
        "global_l2_leaf_reg": float(model_params["global_model"]["l2_leaf_reg"]),
        "global_bagging_temperature": float(model_params["global_model"]["bagging_temperature"]),
        "global_random_strength": float(model_params["global_model"]["random_strength"]),
        "local_iterations": int(model_params["local_model"]["iterations"]),
        "local_learning_rate": float(model_params["local_model"]["learning_rate"]),
        "local_depth": int(model_params["local_model"]["depth"]),
        "local_l2_leaf_reg": float(model_params["local_model"]["l2_leaf_reg"]),
        "local_bagging_temperature": float(model_params["local_model"]["bagging_temperature"]),
        "local_random_strength": float(model_params["local_model"]["random_strength"]),
    }


def _model_params_from_trial(trial: Any, base_model_params: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    params = copy.deepcopy(base_model_params)
    params["global_model"].update(
        {
            "iterations": int(trial.suggest_int("global_iterations", 1400, 5000, step=100)),
            "learning_rate": float(trial.suggest_float("global_learning_rate", 0.008, 0.08, log=True)),
            "depth": int(trial.suggest_int("global_depth", 6, 10)),
            "l2_leaf_reg": float(trial.suggest_float("global_l2_leaf_reg", 4.0, 80.0, log=True)),
            "bagging_temperature": float(trial.suggest_float("global_bagging_temperature", 0.0, 2.5)),
            "random_strength": float(trial.suggest_float("global_random_strength", 0.1, 2.5)),
            "verbose": 0,
        }
    )
    params["local_model"].update(
        {
            "iterations": int(trial.suggest_int("local_iterations", 1600, 5600, step=100)),
            "learning_rate": float(trial.suggest_float("local_learning_rate", 0.008, 0.08, log=True)),
            "depth": int(trial.suggest_int("local_depth", 6, 10)),
            "l2_leaf_reg": float(trial.suggest_float("local_l2_leaf_reg", 4.0, 100.0, log=True)),
            "bagging_temperature": float(trial.suggest_float("local_bagging_temperature", 0.0, 2.5)),
            "random_strength": float(trial.suggest_float("local_random_strength", 0.1, 2.5)),
            "verbose": 0,
        }
    )
    return params


def _make_global_model(model_params: dict[str, Any] | None = None) -> CatBoostRegressor:
    params = dict(DEFAULT_2C02EB6_MODEL_PARAMS["global_model"])
    if model_params:
        params.update(model_params)
    return CatBoostRegressor(**params)


def _make_local_model(model_params: dict[str, Any] | None = None) -> CatBoostRegressor:
    params = dict(DEFAULT_2C02EB6_MODEL_PARAMS["local_model"])
    if model_params:
        params.update(model_params)
    return CatBoostRegressor(**params)


def tune_2c02eb6_model_params(
    *,
    train_df: pd.DataFrame,
    cv_folds: int,
    cv_val_days: int,
    cv_step_days: int,
    cv_min_train_days: int,
    base_model_params: dict[str, dict[str, Any]],
    tune_trials: int,
    tune_timeout_minutes: float,
    use_residual_stacking: bool,
    residual_oof_folds: int,
    residual_oof_val_days: int,
    residual_oof_step_days: int,
    residual_oof_min_train_days: int,
    add_temperature_demand: bool,
    add_physics_regime: bool,
    drop_redundant_features: bool,
    use_permutation_pruned_feature_set: bool,
    perm_base_seed: int,
    perm_corr_threshold: float,
    perm_eval_families: bool,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for --tune-hparams. Install it first (for example: `uv add optuna`)."
        ) from exc

    if tune_trials <= 0:
        raise ValueError("--tune-trials must be > 0.")
    if tune_timeout_minutes <= 0.0:
        raise ValueError("--tune-time-budget-minutes must be > 0.")

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.enqueue_trial(_build_tuning_defaults_from_params(base_model_params))

    trial_logs: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        t0 = time.perf_counter()
        model_params_trial = _model_params_from_trial(trial, base_model_params)
        try:
            cv_rmse, _, _, _ = run_time_series_cv(
                train_df_raw=train_df,
                n_folds=cv_folds,
                val_days=cv_val_days,
                step_days=cv_step_days,
                min_train_days=cv_min_train_days,
                use_residual_stacking=use_residual_stacking,
                residual_oof_folds=residual_oof_folds,
                residual_oof_val_days=residual_oof_val_days,
                residual_oof_step_days=residual_oof_step_days,
                residual_oof_min_train_days=residual_oof_min_train_days,
                add_temperature_demand=add_temperature_demand,
                add_physics_regime=add_physics_regime,
                drop_redundant_features=drop_redundant_features,
                use_permutation_pruned_feature_set=use_permutation_pruned_feature_set,
                permutation_eval_enabled=False,
                permutation_eval_base_seed=perm_base_seed,
                permutation_corr_threshold=perm_corr_threshold,
                permutation_eval_families=perm_eval_families,
                on_fold_complete=None,
                model_params=model_params_trial,
            )
            score = float(cv_rmse) if cv_rmse is not None and np.isfinite(cv_rmse) else float("inf")
        except Exception as exc:
            score = float("inf")
            trial.set_user_attr("exception", str(exc))

        elapsed_s = float(time.perf_counter() - t0)
        trial.set_user_attr("elapsed_seconds", elapsed_s)
        trial_logs.append(
            {
                "trial": int(trial.number),
                "cv_rmse": float(score),
                "elapsed_seconds": elapsed_s,
            }
        )
        print(f"[TUNE] trial={trial.number} cv_rmse={score:.6f} elapsed={elapsed_s/60.0:.2f}m")
        return score

    total_timeout_seconds = float(tune_timeout_minutes) * 60.0
    t_start = time.perf_counter()
    study.optimize(
        objective,
        n_trials=int(tune_trials),
        timeout=total_timeout_seconds,
        show_progress_bar=False,
        gc_after_trial=True,
    )
    elapsed_seconds = float(time.perf_counter() - t_start)

    best_model_params = _model_params_from_trial(study.best_trial, base_model_params)
    report = {
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_params": dict(study.best_trial.params),
        "best_trial_user_attrs": dict(study.best_trial.user_attrs),
        "n_trials_completed": int(len(study.trials)),
        "elapsed_seconds": elapsed_seconds,
        "trials": trial_logs,
    }
    return best_model_params, report


def maybe_drop_redundant_features(
    feature_cols: list[str],
    *,
    enabled: bool,
) -> tuple[list[str], list[str]]:
    if not enabled:
        return feature_cols, []
    drop_set = set(REDUNDANT_FEATURE_DROP_LIST)
    kept = [c for c in feature_cols if c not in drop_set]
    dropped = [c for c in feature_cols if c in drop_set]
    return kept, dropped


def apply_permutation_pruned_feature_policy(
    feature_cols: list[str],
    *,
    enabled: bool,
) -> tuple[list[str], list[str]]:
    if not enabled:
        return feature_cols, []

    kept: list[str] = []
    dropped: list[str] = []

    for col in feature_cols:
        if col.startswith("target_profile_"):
            dropped.append(col)
            continue
        if "_xmk_" in col:
            dropped.append(col)
            continue

        is_short_lag = re.search(r"_lag_(1|2|6)$", col) is not None
        is_roll_mean_6_24 = re.search(r"_roll_mean_(6|24)$", col) is not None
        should_keep = (
            col in CORE_FORECAST_FEATURES
            or is_short_lag
            or is_roll_mean_6_24
            or col in KEY_METEO_FEATURES
            or col in ROBUST_POSITIVE_FEATURES
        )
        if should_keep:
            kept.append(col)
        else:
            dropped.append(col)

    if not kept:
        raise ValueError("Permutation-pruned feature policy removed all features.")
    return kept, dropped


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if not np.isfinite(x):
            return "null"
        return format(x, ".15g")
    s = str(v)
    if s == "" or any(ch in s for ch in [":", "#", "{", "}", "[", "]", ",", "\n", "\"", "'"]) or s.strip() != s:
        return json.dumps(s)
    return s


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    sp = "  " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{sp}{{}}"]
        lines: list[str] = []
        for k, v in value.items():
            key = str(k)
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.extend(_yaml_lines(v, indent + 1))
            else:
                lines.append(f"{sp}{key}: {_yaml_scalar(v)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{sp}[]"]
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                lines.extend(_yaml_lines(item, indent + 1))
            else:
                lines.append(f"{sp}- {_yaml_scalar(item)}")
        return lines
    return [f"{sp}{_yaml_scalar(value)}"]


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text("\n".join(_yaml_lines(data)) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_cmd(args: list[str]) -> str | None:
    try:
        p = subprocess.run(args, capture_output=True, text=True, check=True)
        return p.stdout.strip()
    except Exception:
        return None


def _cat_indices(feature_cols: list[str], cat_cols: list[str]) -> list[int]:
    cset = set(cat_cols)
    return [i for i, c in enumerate(feature_cols) if c in cset]


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def save_shap_outputs(
    run_dir: Path,
    train_feat: pd.DataFrame,
    artifacts: TrainArtifacts,
    global_feature_cols: list[str],
    *,
    global_sample_size: int,
    per_market_sample_size: int,
    seed: int,
) -> None:
    global_sample = _sample_df(train_feat, global_sample_size, seed)
    global_pool = Pool(
        data=global_sample[global_feature_cols],
        cat_features=_cat_indices(global_feature_cols, artifacts.cat_cols),
    )
    shap_global = artifacts.global_model.get_feature_importance(global_pool, type="ShapValues")
    shap_global_vals = shap_global[:, :-1]
    shap_global_base = shap_global[:, -1]
    global_preds = artifacts.global_model.predict(global_sample[global_feature_cols])

    global_imp = pd.DataFrame(
        {"feature": global_feature_cols, "mean_abs_shap": np.abs(shap_global_vals).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    global_imp.to_csv(run_dir / "global_feature_importance_shap.csv", index=False)

    global_rows = global_sample[["id", "market", "delivery_start", "target"]].reset_index(drop=True)
    global_rows = global_rows.assign(base_value=shap_global_base, prediction=global_preds)
    shap_cols = pd.DataFrame(
        shap_global_vals,
        columns=[f"shap__{feat}" for feat in global_feature_cols],
    )
    global_rows = pd.concat([global_rows, shap_cols], axis=1)
    global_rows.to_csv(run_dir / "global_shap_sample_rows.csv", index=False)

    train_local = train_feat.copy()
    train_local["global_pred_feature"] = artifacts.global_model.predict(train_feat[global_feature_cols])
    local_rows: list[dict[str, object]] = []
    local_cat_idx = _cat_indices(artifacts.feature_cols, artifacts.cat_cols)

    for market, model in artifacts.local_models.items():
        mdf = train_local.loc[train_local["market"].astype(str) == str(market)].copy()
        if mdf.empty:
            continue
        ms = _sample_df(mdf, per_market_sample_size, seed)
        pool = Pool(data=ms[artifacts.feature_cols], cat_features=local_cat_idx)
        shap_local = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        imp = np.abs(shap_local).mean(axis=0)
        for feat, score in zip(artifacts.feature_cols, imp):
            local_rows.append(
                {"market": str(market), "feature": feat, "mean_abs_shap": float(score)}
            )

    pd.DataFrame(local_rows).sort_values(
        ["market", "mean_abs_shap"], ascending=[True, False]
    ).to_csv(run_dir / "local_feature_importance_shap.csv", index=False)

    out_meta = {
        "run_dir": str(run_dir),
        "global_sample_size_used": int(len(global_sample)),
        "per_market_sample_size_requested": int(per_market_sample_size),
        "rows_train": int(len(train_feat)),
        "global_feature_count": int(len(global_feature_cols)),
        "local_feature_count": int(len(artifacts.feature_cols)),
    }
    (run_dir / "shap_metadata.json").write_text(json.dumps(out_meta, indent=2))

    print("Saved SHAP outputs:")
    print(f"- {run_dir / 'global_feature_importance_shap.csv'}")
    print(f"- {run_dir / 'local_feature_importance_shap.csv'}")
    print(f"- {run_dir / 'global_shap_sample_rows.csv'}")
    print(f"- {run_dir / 'shap_metadata.json'}")


def save_repro_artifacts(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    cv_rmse: float | None,
    train_rows: int,
    test_rows: int,
    candidate_features: list[str],
    artifacts: TrainArtifacts,
    model_file_map: dict[str, Any],
    model_params: dict[str, dict[str, Any]],
) -> None:
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_path = Path(args.sample_submission)

    run_config = {
        "script": Path(__file__).name,
        "created_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_args": vars(args),
        "data": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "sample_submission_path": str(sample_path),
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "sample_submission_sha256": _sha256_file(sample_path),
            "train_rows": train_rows,
            "test_rows": test_rows,
        },
        "model_params": model_params,
        "features": {
            "candidate_features_before_global_pred": candidate_features,
            "local_feature_cols": artifacts.feature_cols,
            "cat_cols": artifacts.cat_cols,
        },
        "metrics": {"cv_rmse": cv_rmse},
    }
    _write_yaml(run_dir / "run_config.yaml", run_config)

    git_commit = _run_cmd(["git", "rev-parse", "HEAD"])
    git_branch = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_status = _run_cmd(["git", "status", "--short"])
    repro_context = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "catboost_version": getattr(sys.modules.get("catboost"), "__version__", None),
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "status_short": git_status,
        },
    }
    (run_dir / "repro_context.json").write_text(
        json.dumps(repro_context, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_metadata = {
        "global_model": model_file_map.get("global_model"),
        "local_models": model_file_map.get("local_models", {}),
        "feature_cols": artifacts.feature_cols,
        "cat_cols": artifacts.cat_cols,
        "local_target_is_residual": artifacts.local_target_is_residual,
        "candidate_features_before_global_pred": candidate_features,
        "cv_rmse": cv_rmse,
        "train_args": vars(args),
        "data_hashes": {
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "sample_submission_sha256": _sha256_file(sample_path),
        },
    }
    (run_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved reproducibility config: {run_dir / 'run_config.yaml'}")
    print(f"Saved reproducibility context: {run_dir / 'repro_context.json'}")
    print(f"Saved model metadata: {run_dir / 'model_metadata.json'}")


def save_permutation_eval_outputs(run_dir: Path, outputs: PermutationEvalOutputs) -> None:
    outputs.perm_importance_by_fold.to_csv(run_dir / "perm_importance_by_fold.csv", index=False)
    outputs.perm_importance_summary.to_csv(run_dir / "perm_importance_summary.csv", index=False)
    outputs.corr_groups.to_csv(run_dir / "corr_groups.csv", index=False)
    outputs.perm_importance_groups_by_fold.to_csv(
        run_dir / "perm_importance_groups_by_fold.csv",
        index=False,
    )
    outputs.perm_importance_groups_summary.to_csv(
        run_dir / "perm_importance_groups_summary.csv",
        index=False,
    )
    outputs.perm_importance_families.to_csv(run_dir / "perm_importance_families.csv", index=False)
    outputs.perm_importance_families_summary.to_csv(
        run_dir / "perm_importance_families_summary.csv",
        index=False,
    )

    meta = {
        "runtime_seconds": float(outputs.runtime_seconds),
        "n_features_evaluated": int(outputs.n_features_evaluated),
        "n_groups_evaluated": int(outputs.n_groups_evaluated),
        "n_family_evaluations": int(outputs.n_family_evaluations),
    }
    (run_dir / "perm_importance_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("Saved permutation OOS importance reports:")
    print(f"- {run_dir / 'perm_importance_by_fold.csv'}")
    print(f"- {run_dir / 'perm_importance_summary.csv'}")
    print(f"- {run_dir / 'corr_groups.csv'}")
    print(f"- {run_dir / 'perm_importance_groups_by_fold.csv'}")
    print(f"- {run_dir / 'perm_importance_groups_summary.csv'}")
    print(f"- {run_dir / 'perm_importance_families.csv'}")
    print(f"- {run_dir / 'perm_importance_families_summary.csv'}")
    print(f"- {run_dir / 'perm_importance_metadata.json'}")


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


def compute_global_oof_predictions(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    *,
    global_model_params: dict[str, Any] | None = None,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
) -> pd.Series:
    folds = make_time_series_folds(
        train_df=train_df,
        n_folds=n_folds,
        val_days=val_days,
        step_days=step_days,
        min_train_days=min_train_days,
    )
    if not folds:
        return pd.Series(index=train_df.index, dtype=float)

    start_all = _to_datetime_col(train_df, "delivery_start")
    oof = pd.Series(index=train_df.index, dtype=float)
    for val_start, val_end in folds:
        tr_mask = start_all < val_start
        va_mask = (start_all >= val_start) & (start_all < val_end)
        tr = train_df.loc[tr_mask]
        va = train_df.loc[va_mask]
        if tr.empty or va.empty:
            continue

        model = _make_global_model(global_model_params)
        model.fit(tr[feature_cols], tr["target"], cat_features=cat_cols)
        oof.loc[va.index] = model.predict(va[feature_cols])
    return oof


def run_time_series_cv(
    train_df_raw: pd.DataFrame,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
    *,
    use_residual_stacking: bool,
    residual_oof_folds: int,
    residual_oof_val_days: int,
    residual_oof_step_days: int,
    residual_oof_min_train_days: int,
    add_temperature_demand: bool,
    add_physics_regime: bool,
    drop_redundant_features: bool,
    use_permutation_pruned_feature_set: bool,
    permutation_eval_enabled: bool,
    permutation_eval_base_seed: int,
    permutation_corr_threshold: float,
    permutation_eval_families: bool,
    on_fold_complete: Callable[[int, int], None] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> tuple[float | None, pd.DataFrame, pd.DataFrame, PermutationEvalOutputs | None]:
    model_params = copy.deepcopy(model_params) if model_params is not None else copy.deepcopy(DEFAULT_2C02EB6_MODEL_PARAMS)
    folds = make_time_series_folds(
        train_df=train_df_raw,
        n_folds=n_folds,
        val_days=val_days,
        step_days=step_days,
        min_train_days=min_train_days,
    )
    if not folds:
        print("CV skipped: not enough history for requested folds.")
        return None, pd.DataFrame(), pd.DataFrame(), None

    start_all = _to_datetime_col(train_df_raw, "delivery_start")
    fold_rows: list[dict[str, object]] = []
    oof_rows: list[pd.DataFrame] = []
    oof = pd.Series(index=train_df_raw.index, dtype=float)
    perm_feature_rows: list[dict[str, object]] = []
    perm_group_rows: list[dict[str, object]] = []
    perm_family_rows: list[dict[str, object]] = []
    corr_group_rows_all: list[dict[str, object]] = []
    perm_runtime_total = 0.0
    n_features_evaluated = 0
    n_groups_evaluated = 0
    n_family_evaluations = 0

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
            if on_fold_complete is not None:
                on_fold_complete(fold_idx, len(folds))
            continue

        print(
            f"CV fold {fold_idx}/{len(folds)} | "
            f"train={len(tr)} val={len(va)} | "
            f"val_range=[{val_start.date()} -> {val_end.date()})"
        )

        tr_feat, va_feat = build_feature_table(
            tr,
            va.drop(columns=["target"]).copy(),
            add_temperature_demand=add_temperature_demand,
            add_physics_regime=add_physics_regime,
        )

        base_drop = {"id", "target", "delivery_start", "delivery_end"}
        feat_cols = [c for c in tr_feat.columns if c not in base_drop]
        feat_cols, dropped_cols = maybe_drop_redundant_features(
            feat_cols,
            enabled=drop_redundant_features,
        )
        if dropped_cols:
            print(f"CV fold {fold_idx}: dropped {len(dropped_cols)} redundant features")
        feat_cols, dropped_policy = apply_permutation_pruned_feature_policy(
            feat_cols,
            enabled=use_permutation_pruned_feature_set,
        )
        if dropped_policy:
            print(
                f"CV fold {fold_idx}: permutation-pruned feature policy active | "
                f"kept={len(feat_cols)} dropped={len(dropped_policy)}"
            )
        cat_cols = [
            c
            for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
            if c in feat_cols
        ]

        artifacts = train_global_and_local_models(
            tr_feat,
            feat_cols,
            cat_cols,
            model_params=model_params,
            use_residual_stacking=use_residual_stacking,
            residual_oof_folds=residual_oof_folds,
            residual_oof_val_days=residual_oof_val_days,
            residual_oof_step_days=residual_oof_step_days,
            residual_oof_min_train_days=residual_oof_min_train_days,
        )
        pred = _predict_with_global_local_pipeline(
            va_feat,
            artifacts=artifacts,
            global_feature_cols=feat_cols,
        )
        y_val = va["target"].to_numpy(dtype=float)
        fold_rmse = _rmse(y_val, pred)
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
                "market": va["market"].to_numpy(),
                "y_true": va["target"].to_numpy(dtype=float),
                "y_pred": pred,
            }
        )
        for market, sdf in tmp.groupby("market"):
            m_rmse = _rmse(sdf["y_true"].to_numpy(), sdf["y_pred"].to_numpy())
            print(f"  - {market}: RMSE={m_rmse:.6f} (n={len(sdf)})")
        oof_rows.append(
            pd.DataFrame(
                {
                    "id": va["id"].to_numpy(dtype=int),
                    "delivery_start": va["delivery_start"].to_numpy(),
                    "market": va["market"].to_numpy(),
                    "target": va["target"].to_numpy(dtype=float),
                    "pred": pred,
                    "fold": fold_idx,
                }
            )
        )

        if permutation_eval_enabled:
            fold_eval_start = time.perf_counter()
            baseline_rmse = _rmse(y_val, pred)
            if abs(baseline_rmse - fold_rmse) > 1e-9:
                raise ValueError(
                    f"Baseline RMSE mismatch in fold {fold_idx}: baseline={baseline_rmse}, fold={fold_rmse}"
                )
            n_val_rows = len(va_feat)
            if n_val_rows == 0:
                raise ValueError(f"Cannot run permutation eval for fold {fold_idx} with empty validation set.")

            feature_progress = LoopProgress(
                label=f"Fold {fold_idx} feature permutations",
                total=len(feat_cols),
            )
            for feature_i, feature in enumerate(feat_cols, start=1):
                seed = _permutation_seed(
                    permutation_eval_base_seed,
                    fold_idx,
                    f"feature::{feature}",
                )
                perm_idx = np.random.default_rng(seed).permutation(n_val_rows)
                va_perm = _permute_columns_with_index(va_feat, [feature], perm_idx)
                perm_pred = _predict_with_global_local_pipeline(
                    va_perm,
                    artifacts=artifacts,
                    global_feature_cols=feat_cols,
                )
                rmse_permuted = _rmse(y_val, perm_pred)
                # Positive delta means robust OOS signal; negative delta suggests noisy/overfit behavior.
                delta = rmse_permuted - baseline_rmse
                perm_feature_rows.append(
                    {
                        "fold": fold_idx,
                        "feature": feature,
                        "baseline_rmse": baseline_rmse,
                        "rmse_permuted": rmse_permuted,
                        "delta": delta,
                        "n_val_rows": n_val_rows,
                    }
                )
                feature_progress.update(feature_i)
            feature_progress.finish()
            n_features_evaluated += len(feat_cols)

            corr_groups, corr_rows = _build_correlation_groups(
                tr_feat,
                feature_cols=feat_cols,
                cat_cols=cat_cols,
                corr_threshold=permutation_corr_threshold,
            )
            corr_group_rows_all.extend(corr_rows)
            group_progress = LoopProgress(
                label=f"Fold {fold_idx} correlation-group permutations",
                total=len(corr_groups),
            )
            for group_i, (group_id, group_features) in enumerate(corr_groups, start=1):
                seed = _permutation_seed(
                    permutation_eval_base_seed,
                    fold_idx,
                    f"group::{group_id}",
                )
                perm_idx = np.random.default_rng(seed).permutation(n_val_rows)
                va_perm = _permute_columns_with_index(va_feat, group_features, perm_idx)
                perm_pred = _predict_with_global_local_pipeline(
                    va_perm,
                    artifacts=artifacts,
                    global_feature_cols=feat_cols,
                )
                rmse_permuted = _rmse(y_val, perm_pred)
                delta = rmse_permuted - baseline_rmse
                perm_group_rows.append(
                    {
                        "fold": fold_idx,
                        "group_id": group_id,
                        "group_size": int(len(group_features)),
                        "baseline_rmse": baseline_rmse,
                        "rmse_permuted": rmse_permuted,
                        "delta": delta,
                    }
                )
                group_progress.update(group_i)
            group_progress.finish()
            n_groups_evaluated += len(corr_groups)

            families_evaluated_in_fold = 0
            if permutation_eval_families:
                family_map = _feature_family_map(feat_cols)
                family_items = [(k, v) for k, v in family_map.items() if v]
                family_progress = LoopProgress(
                    label=f"Fold {fold_idx} family permutations",
                    total=len(family_items),
                )
                for family_i, (family_name, family_features) in enumerate(family_items, start=1):
                    seed = _permutation_seed(
                        permutation_eval_base_seed,
                        fold_idx,
                        f"family::{family_name}",
                    )
                    perm_idx = np.random.default_rng(seed).permutation(n_val_rows)
                    va_perm = _permute_columns_with_index(va_feat, family_features, perm_idx)
                    perm_pred = _predict_with_global_local_pipeline(
                        va_perm,
                        artifacts=artifacts,
                        global_feature_cols=feat_cols,
                    )
                    rmse_permuted = _rmse(y_val, perm_pred)
                    delta = rmse_permuted - baseline_rmse
                    perm_family_rows.append(
                        {
                            "fold": fold_idx,
                            "family_name": family_name,
                            "baseline_rmse": baseline_rmse,
                            "rmse_permuted": rmse_permuted,
                            "delta": delta,
                        }
                    )
                    families_evaluated_in_fold += 1
                    family_progress.update(family_i)
                family_progress.finish()
            n_family_evaluations += families_evaluated_in_fold

            fold_eval_runtime = time.perf_counter() - fold_eval_start
            perm_runtime_total += fold_eval_runtime
            print(
                f"CV fold {fold_idx}: permutation OOS eval complete | "
                f"features={len(feat_cols)} groups={len(corr_groups)} "
                f"families={families_evaluated_in_fold} runtime={fold_eval_runtime:.2f}s"
            )
        oof.loc[va.index] = pred
        if on_fold_complete is not None:
            on_fold_complete(fold_idx, len(folds))

    coverage = float(oof.notna().mean())
    overall = None
    if coverage > 0.0:
        valid = oof.notna()
        overall = _rmse(
            train_df_raw.loc[valid, "target"].to_numpy(dtype=float),
            oof.loc[valid].to_numpy(dtype=float),
        )
    print(f"CV OOF coverage: {coverage:.4%}")
    print(f"CV OOF RMSE: {overall if overall is not None else 'None'}")
    cv_oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    permutation_outputs = None
    if permutation_eval_enabled:
        feature_cols_out = [
            "fold",
            "feature",
            "baseline_rmse",
            "rmse_permuted",
            "delta",
            "n_val_rows",
        ]
        group_cols_out = [
            "fold",
            "group_id",
            "group_size",
            "baseline_rmse",
            "rmse_permuted",
            "delta",
        ]
        family_cols_out = [
            "fold",
            "family_name",
            "baseline_rmse",
            "rmse_permuted",
            "delta",
        ]
        corr_group_cols_out = ["group_id", "feature", "group_size", "corr_threshold"]

        perm_importance_by_fold = (
            pd.DataFrame(perm_feature_rows, columns=feature_cols_out)
            .sort_values(["fold", "feature"])
            .reset_index(drop=True)
            if perm_feature_rows
            else pd.DataFrame(columns=feature_cols_out)
        )
        corr_groups = (
            pd.DataFrame(corr_group_rows_all, columns=corr_group_cols_out)
            .drop_duplicates()
            .sort_values(["group_id", "feature"])
            .reset_index(drop=True)
            if corr_group_rows_all
            else pd.DataFrame(columns=corr_group_cols_out)
        )
        perm_importance_groups_by_fold = (
            pd.DataFrame(perm_group_rows, columns=group_cols_out)
            .sort_values(["fold", "group_id"])
            .reset_index(drop=True)
            if perm_group_rows
            else pd.DataFrame(columns=group_cols_out)
        )
        perm_importance_families = (
            pd.DataFrame(perm_family_rows, columns=family_cols_out)
            .sort_values(["fold", "family_name"])
            .reset_index(drop=True)
            if perm_family_rows
            else pd.DataFrame(columns=family_cols_out)
        )

        permutation_outputs = PermutationEvalOutputs(
            perm_importance_by_fold=perm_importance_by_fold,
            perm_importance_summary=_summarize_single_feature_importance(perm_importance_by_fold),
            corr_groups=corr_groups,
            perm_importance_groups_by_fold=perm_importance_groups_by_fold,
            perm_importance_groups_summary=_summarize_group_importance(perm_importance_groups_by_fold),
            perm_importance_families=perm_importance_families,
            perm_importance_families_summary=_summarize_family_importance(perm_importance_families),
            runtime_seconds=perm_runtime_total,
            n_features_evaluated=n_features_evaluated,
            n_groups_evaluated=n_groups_evaluated,
            n_family_evaluations=n_family_evaluations,
        )
        print(
            "Permutation OOS eval totals: "
            f"features={n_features_evaluated}, "
            f"groups={n_groups_evaluated}, "
            f"family_evaluations={n_family_evaluations}, "
            f"runtime={perm_runtime_total:.2f}s"
        )
    return overall, pd.DataFrame(fold_rows), cv_oof, permutation_outputs


def train_global_and_local_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    *,
    model_params: dict[str, dict[str, Any]] | None = None,
    use_residual_stacking: bool,
    residual_oof_folds: int,
    residual_oof_val_days: int,
    residual_oof_step_days: int,
    residual_oof_min_train_days: int,
) -> TrainArtifacts:
    model_params = copy.deepcopy(model_params) if model_params is not None else copy.deepcopy(DEFAULT_2C02EB6_MODEL_PARAMS)
    global_model = _make_global_model(model_params.get("global_model"))
    global_model.fit(train_df[feature_cols], train_df["target"], cat_features=cat_cols)
    train_df = train_df.copy()
    train_df["global_pred_feature"] = global_model.predict(train_df[feature_cols])

    local_feature_cols = feature_cols + ["global_pred_feature"]
    if use_residual_stacking:
        oof_global = compute_global_oof_predictions(
            train_df,
            feature_cols,
            cat_cols,
            global_model_params=model_params.get("global_model"),
            n_folds=residual_oof_folds,
            val_days=residual_oof_val_days,
            step_days=residual_oof_step_days,
            min_train_days=residual_oof_min_train_days,
        )
        coverage = float(oof_global.notna().mean()) if len(oof_global) > 0 else 0.0
        if coverage == 0.0:
            print(
                "Residual stacking: no OOF global predictions generated; "
                "falling back to in-sample global predictions for residual target."
            )
        train_df["global_pred_oof_feature"] = oof_global.fillna(train_df["global_pred_feature"])
        print(f"Residual stacking OOF coverage: {coverage:.4%}")

    local_models: dict[str, CatBoostRegressor] = {}
    for market, mdf in train_df.groupby("market", dropna=False):
        model = _make_local_model(model_params.get("local_model"))
        local_target = (
            mdf["target"] - mdf["global_pred_oof_feature"]
            if use_residual_stacking
            else mdf["target"]
        )
        model.fit(mdf[local_feature_cols], local_target, cat_features=cat_cols)
        local_models[str(market)] = model
        print(f"Trained local model for {market} ({len(mdf)} rows)")

    return TrainArtifacts(
        global_model=global_model,
        local_models=local_models,
        feature_cols=local_feature_cols,
        cat_cols=cat_cols,
        local_target_is_residual=use_residual_stacking,
    )


def build_feature_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    add_temperature_demand: bool = False,
    add_physics_regime: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_df = pd.concat([train_df.assign(_is_train=1), test_df.assign(_is_train=0)], axis=0, ignore_index=True)
    all_df = add_time_features(all_df)
    all_df = add_forecast_core_features(all_df)
    all_df = add_forecast_lag_features(all_df)
    all_df = add_meteo_features(all_df)
    if add_temperature_demand:
        all_df = add_temperature_demand_features(all_df)
    if add_physics_regime:
        all_df = add_physics_regime_features(all_df)
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
    parser.add_argument(
        "--params-in",
        default=None,
        help="Optional JSON overrides for global/local CatBoost params.",
    )
    parser.add_argument(
        "--params-out",
        default=None,
        help="Optional path to write the effective CatBoost params used in the run.",
    )
    parser.add_argument(
        "--use-2c02eb6-trial1-hparams",
        action="store_true",
        help=(
            "Apply fixed trial hyperparameters for global/local CatBoost models "
            "(2c02eb6-compatible). Applied after --params-in."
        ),
    )
    parser.add_argument("--exclude-2023", action="store_true")
    parser.add_argument("--exclude-2023-keep-from-month", type=int, default=10)
    parser.add_argument(
        "--train-start-oct-2023",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict training rows to delivery_start >= 2023-10-01 (default: disabled).",
    )
    parser.add_argument(
        "--add-temperature-demand-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add temperature-demand features (hdd_18, cdd_22, temp_extreme_mag, temp_extreme_flag) (default: disabled).",
    )
    parser.add_argument(
        "--add-physics-regime-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add wind/ramp/cloud/temp/storm regime features (default: disabled).",
    )
    parser.add_argument(
        "--drop-redundant-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Drop conservative duplicate/constant feature set (default: disabled).",
    )
    parser.add_argument(
        "--use-permutation-pruned-feature-set",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply strict permutation-guided feature policy: drop full profiles/xmk families, "
            "then keep only core forecasts, short lags (<=6), roll_mean_{6,24}, "
            "wind_speed_80m + key meteo, and curated robust positives (default: disabled)."
        ),
    )
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-val-days", type=int, default=14)
    parser.add_argument("--cv-step-days", type=int, default=14)
    parser.add_argument("--cv-min-train-days", type=int, default=90)
    parser.add_argument(
        "--tune-hparams",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Optuna tuning for 2c02eb6 global/local CatBoost params before main training.",
    )
    parser.add_argument("--tune-trials", type=int, default=30)
    parser.add_argument("--tune-time-budget-minutes", type=float, default=120.0)
    parser.add_argument("--tune-cv-folds", type=int, default=None)
    parser.add_argument("--tune-cv-val-days", type=int, default=None)
    parser.add_argument("--tune-cv-step-days", type=int, default=None)
    parser.add_argument("--tune-cv-min-train-days", type=int, default=None)
    parser.add_argument(
        "--save-permutation-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OOS permutation importance during CV and save CSV reports (default: enabled).",
    )
    parser.add_argument(
        "--perm-base-seed",
        type=int,
        default=42,
        help="Base seed used for deterministic fold-aware permutations (default: 42).",
    )
    parser.add_argument(
        "--perm-corr-threshold",
        type=float,
        default=0.95,
        help="Absolute Pearson threshold for correlation-group components (default: 0.95).",
    )
    parser.add_argument(
        "--perm-eval-families",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate predefined feature-family group permutations (default: enabled).",
    )
    parser.add_argument(
        "--use-residual-stacking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train local market models on residual target (y - global_pred_oof) and add global prediction at inference (default: disabled).",
    )
    parser.add_argument(
        "--residual-oof-folds",
        type=int,
        default=3,
        help="Inner time-series folds used to build global OOF predictions for residual targets (default: 3).",
    )
    parser.add_argument(
        "--residual-oof-val-days",
        type=int,
        default=14,
        help="Validation window days for residual OOF global predictions (default: 14).",
    )
    parser.add_argument(
        "--residual-oof-step-days",
        type=int,
        default=14,
        help="Step days for residual OOF global predictions (default: 14).",
    )
    parser.add_argument(
        "--residual-oof-min-train-days",
        type=int,
        default=90,
        help="Minimum train days for residual OOF global predictions (default: 90).",
    )
    parser.add_argument(
        "--save-shap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and save SHAP outputs after training (default: enabled).",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save fitted global/local CatBoost models in the run directory (default: enabled).",
    )
    parser.add_argument(
        "--save-repro-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save YAML config + metadata/context files for reproducibility (default: enabled).",
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
    parser.add_argument(
        "--shap-seed",
        type=int,
        default=42,
        help="Sampling seed for SHAP exports (default: 42).",
    )
    args = parser.parse_args()
    if not (0.0 <= args.perm_corr_threshold <= 1.0):
        raise ValueError("--perm-corr-threshold must be between 0 and 1.")
    if args.tune_trials <= 0:
        raise ValueError("--tune-trials must be > 0.")
    if args.tune_time_budget_minutes <= 0.0:
        raise ValueError("--tune-time-budget-minutes must be > 0.")

    tune_cv_folds = int(args.tune_cv_folds) if args.tune_cv_folds is not None else int(args.cv_folds)
    tune_cv_val_days = int(args.tune_cv_val_days) if args.tune_cv_val_days is not None else int(args.cv_val_days)
    tune_cv_step_days = int(args.tune_cv_step_days) if args.tune_cv_step_days is not None else int(args.cv_step_days)
    tune_cv_min_train_days = (
        int(args.tune_cv_min_train_days)
        if args.tune_cv_min_train_days is not None
        else int(args.cv_min_train_days)
    )
    model_params = load_2c02eb6_model_params(args.params_in)
    if args.use_2c02eb6_trial1_hparams:
        for section in ("global_model", "local_model"):
            model_params[section].update(TUNED_2C02EB6_TRIAL1_MODEL_PARAM_OVERRIDES[section])
        print("Applied model-param preset: 2c02eb6_trial1_tuned_hparams")

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample = pd.read_csv(args.sample_submission)

    if args.exclude_2023:
        train_df = apply_exclude_2023(train_df, keep_from_month=args.exclude_2023_keep_from_month)
    if args.train_start_oct_2023:
        train_df = apply_train_start_cutoff(train_df, start_date="2023-10-01")

    cv_fold_count = 0
    if args.cv:
        cv_fold_count = len(
            make_time_series_folds(
                train_df=train_df,
                n_folds=args.cv_folds,
                val_days=args.cv_val_days,
                step_days=args.cv_step_days,
                min_train_days=args.cv_min_train_days,
            )
        )
    overall_total_steps = (
        4  # data prepared + main feature table + main training + submission write
        + cv_fold_count
        + int(args.tune_hparams)
        + int(args.save_models)
        + int(args.save_repro_artifacts)
        + int(args.save_shap)
    )
    overall_progress = LoopProgress(
        label="Overall process",
        total=overall_total_steps,
        min_interval_seconds=0.0,
    )
    overall_done = 0

    def _overall_tick(step_name: str) -> None:
        nonlocal overall_done
        overall_done += 1
        print(f"Overall step complete: {step_name}")
        overall_progress.update(overall_done, force=True)

    _overall_tick("data loaded and filters applied")

    tuning_report: dict[str, Any] | None = None
    if args.tune_hparams:
        print(
            "Tuning 2c02eb6 CatBoost params: "
            f"trials={args.tune_trials}, time_budget_min={args.tune_time_budget_minutes}, "
            f"cv={tune_cv_folds}x{tune_cv_val_days}d"
        )
        model_params, tuning_report = tune_2c02eb6_model_params(
            train_df=train_df,
            cv_folds=tune_cv_folds,
            cv_val_days=tune_cv_val_days,
            cv_step_days=tune_cv_step_days,
            cv_min_train_days=tune_cv_min_train_days,
            base_model_params=model_params,
            tune_trials=args.tune_trials,
            tune_timeout_minutes=args.tune_time_budget_minutes,
            use_residual_stacking=args.use_residual_stacking,
            residual_oof_folds=args.residual_oof_folds,
            residual_oof_val_days=args.residual_oof_val_days,
            residual_oof_step_days=args.residual_oof_step_days,
            residual_oof_min_train_days=args.residual_oof_min_train_days,
            add_temperature_demand=args.add_temperature_demand_features,
            add_physics_regime=args.add_physics_regime_features,
            drop_redundant_features=args.drop_redundant_features,
            use_permutation_pruned_feature_set=args.use_permutation_pruned_feature_set,
            perm_base_seed=args.perm_base_seed,
            perm_corr_threshold=args.perm_corr_threshold,
            perm_eval_families=args.perm_eval_families,
        )
        print(
            "Tuning complete: "
            f"best_cv_rmse={tuning_report['best_value']:.6f} "
            f"best_trial={tuning_report['best_trial_number']}"
        )
        _overall_tick("hyperparameter tuning complete")

    cv_rmse = None
    cv_details = pd.DataFrame()
    cv_oof = pd.DataFrame()
    cv_permutation_outputs: PermutationEvalOutputs | None = None
    if args.cv:
        def _on_cv_fold_complete(fold_index: int, fold_total: int) -> None:
            _overall_tick(f"cv fold {fold_index}/{fold_total}")

        cv_rmse, cv_details, cv_oof, cv_permutation_outputs = run_time_series_cv(
            train_df_raw=train_df,
            n_folds=args.cv_folds,
            val_days=args.cv_val_days,
            step_days=args.cv_step_days,
            min_train_days=args.cv_min_train_days,
            use_residual_stacking=args.use_residual_stacking,
            residual_oof_folds=args.residual_oof_folds,
            residual_oof_val_days=args.residual_oof_val_days,
            residual_oof_step_days=args.residual_oof_step_days,
            residual_oof_min_train_days=args.residual_oof_min_train_days,
            add_temperature_demand=args.add_temperature_demand_features,
            add_physics_regime=args.add_physics_regime_features,
            drop_redundant_features=args.drop_redundant_features,
            use_permutation_pruned_feature_set=args.use_permutation_pruned_feature_set,
            permutation_eval_enabled=args.save_permutation_eval,
            permutation_eval_base_seed=args.perm_base_seed,
            permutation_corr_threshold=args.perm_corr_threshold,
            permutation_eval_families=args.perm_eval_families,
            on_fold_complete=_on_cv_fold_complete,
            model_params=model_params,
        )

    train_feat, test_feat = build_feature_table(
        train_df,
        test_df,
        add_temperature_demand=args.add_temperature_demand_features,
        add_physics_regime=args.add_physics_regime_features,
    )
    _overall_tick("main feature table built")
    test_with_key = test_feat[["id", "market"]].copy()

    base_drop = {"id", "target", "delivery_start", "delivery_end"}
    candidate_features = [c for c in train_feat.columns if c not in base_drop]
    candidate_features, dropped_cols_main = maybe_drop_redundant_features(
        candidate_features,
        enabled=args.drop_redundant_features,
    )
    if dropped_cols_main:
        print(f"Dropped {len(dropped_cols_main)} redundant features in main fit path.")
        print(f"Dropped features: {sorted(dropped_cols_main)}")
    candidate_features, dropped_policy_main = apply_permutation_pruned_feature_policy(
        candidate_features,
        enabled=args.use_permutation_pruned_feature_set,
    )
    if dropped_policy_main:
        print(
            "Permutation-pruned feature policy active in main fit path: "
            f"kept={len(candidate_features)} dropped={len(dropped_policy_main)}"
        )
    cat_cols = [
        c
        for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
        if c in candidate_features
    ]

    artifacts = train_global_and_local_models(
        train_feat,
        candidate_features,
        cat_cols,
        model_params=model_params,
        use_residual_stacking=args.use_residual_stacking,
        residual_oof_folds=args.residual_oof_folds,
        residual_oof_val_days=args.residual_oof_val_days,
        residual_oof_step_days=args.residual_oof_step_days,
        residual_oof_min_train_days=args.residual_oof_min_train_days,
    )
    _overall_tick("main global/local models trained")

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
        local_pred = model.predict(test_feat.loc[idx, artifacts.feature_cols])
        if artifacts.local_target_is_residual:
            pred[idx] = test_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float) + local_pred
        else:
            pred[idx] = local_pred

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

    latest_path = Path("csv/submission_per_market_interactions.csv")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    out_sub.to_csv(latest_path, index=False)
    _overall_tick("submission generated and saved")

    print(f"Saved submission: {sub_path}")
    print(f"Saved latest copy: {latest_path}")
    print(f"Features used: {len(artifacts.feature_cols)}")
    print(f"Categorical features: {artifacts.cat_cols}")
    print(f"Markets modeled: {sorted(artifacts.local_models.keys())}")
    print(f"Local target is residual: {artifacts.local_target_is_residual}")
    print(f"CV RMSE: {cv_rmse}")
    if tuning_report is not None:
        print(
            f"Tuning best CV RMSE: {tuning_report['best_value']:.6f} "
            f"(trial {tuning_report['best_trial_number']})"
        )
    model_params_path = run_dir / "model_params.json"
    model_params_path.write_text(json.dumps(model_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved effective model params: {model_params_path}")
    if args.params_out:
        Path(args.params_out).write_text(json.dumps(model_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved params-out copy: {args.params_out}")
    if tuning_report is not None:
        tuning_json_path = run_dir / "hparam_tuning.json"
        tuning_csv_path = run_dir / "hparam_tuning_trials.csv"
        tuning_json_path.write_text(json.dumps(tuning_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if tuning_report.get("trials"):
            pd.DataFrame(tuning_report["trials"]).to_csv(tuning_csv_path, index=False)
        print(f"Saved tuning report: {tuning_json_path}")
        if tuning_report.get("trials"):
            print(f"Saved tuning trials: {tuning_csv_path}")
    if not cv_details.empty:
        cv_path = run_dir / "cv_results.csv"
        cv_details.to_csv(cv_path, index=False)
        print(f"Saved CV details: {cv_path}")
    if not cv_oof.empty:
        cv_oof_path = run_dir / "cv_oof.csv"
        cv_oof.to_csv(cv_oof_path, index=False)
        print(f"Saved CV OOF rows: {cv_oof_path}")

        market_rows: list[dict[str, object]] = []
        for market, sdf in cv_oof.groupby("market", dropna=False):
            market_rows.append(
                {
                    "market": str(market),
                    "rows": int(len(sdf)),
                    "rmse": _rmse(
                        sdf["target"].to_numpy(dtype=float),
                        sdf["pred"].to_numpy(dtype=float),
                    ),
                }
            )
        cv_market = pd.DataFrame(market_rows).sort_values("market").reset_index(drop=True)
        cv_market_path = run_dir / "cv_market_results.csv"
        cv_market.to_csv(cv_market_path, index=False)
        print(f"Saved CV market details: {cv_market_path}")
    if cv_permutation_outputs is not None:
        save_permutation_eval_outputs(run_dir=run_dir, outputs=cv_permutation_outputs)
    model_file_map: dict[str, Any] = {"global_model": None, "local_models": {}}
    if args.save_models:
        models_dir = run_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        global_model_path = models_dir / "global_model.cbm"
        artifacts.global_model.save_model(global_model_path)
        model_file_map["global_model"] = str(global_model_path.name)

        local_model_paths: dict[str, str] = {}
        for market, model in artifacts.local_models.items():
            safe_market = str(market).replace(" ", "_")
            local_path = models_dir / f"local_model_{safe_market}.cbm"
            model.save_model(local_path)
            local_model_paths[str(market)] = str(local_path.name)
        model_file_map["local_models"] = local_model_paths
        print(f"Saved models dir: {models_dir}")
        _overall_tick("models saved")

    if args.save_repro_artifacts:
        save_repro_artifacts(
            run_dir=run_dir,
            args=args,
            cv_rmse=cv_rmse,
            train_rows=len(train_df),
            test_rows=len(test_df),
            candidate_features=candidate_features,
            artifacts=artifacts,
            model_file_map=model_file_map,
            model_params=model_params,
        )
        _overall_tick("repro artifacts saved")

    if args.save_shap:
        save_shap_outputs(
            run_dir=run_dir,
            train_feat=train_feat,
            artifacts=artifacts,
            global_feature_cols=candidate_features,
            global_sample_size=args.shap_global_sample_size,
            per_market_sample_size=args.shap_per_market_sample_size,
            seed=args.shap_seed,
        )
        _overall_tick("shap outputs saved")


if __name__ == "__main__":
    main()
