from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAG_STEPS = [1, 2, 24, 48, 168]
ROLL_WINDOWS = [24, 48, 168]
EXOG_LAG_STEPS = [1, 2, 6, 24, 48, 168]
EXOG_ROLL_WINDOWS = [6, 24, 72]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _run_git_command(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def _get_git_info(repo_root: Path) -> dict[str, str | bool]:
    git_sha = _run_git_command(["rev-parse", "--short", "HEAD"], repo_root)
    git_branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    dirty_out = _run_git_command(["status", "--porcelain"], repo_root)
    dirty_repo = dirty_out not in {"", "unknown"}
    return {
        "git_sha": git_sha,
        "git_branch": git_branch,
        "dirty_repo": dirty_repo,
    }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)


def _write_params_yaml(path: Path, payload: dict) -> None:
    def _format_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return str(value)
        return json.dumps(str(value), ensure_ascii=False)

    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {_format_value(sub_value)}")
        else:
            lines.append(f"{key}: {_format_value(value)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_experiment_registry(registry_path: Path, row: dict[str, str]) -> None:
    fieldnames = [
        "run_id",
        "git_sha",
        "git_branch",
        "dirty_repo",
        "data_hash",
        "config_hash",
        "seed",
        "cv_rmse",
        "lb_score",
        "model_path",
        "submission_path",
        "started_at",
    ]

    needs_header = not registry_path.exists()
    with registry_path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _add_harmonics(df: pd.DataFrame, column: str, period: int, harmonics: tuple[int, ...] = (1, 2, 3)) -> None:
    for k in harmonics:
        radians = 2.0 * np.pi * k * (df[column] / period)
        df[f"{column}_sin_{k}"] = np.sin(radians)
        df[f"{column}_cos_{k}"] = np.cos(radians)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    start_ts = pd.to_datetime(result["delivery_start"], errors="coerce")
    end_ts = pd.to_datetime(result["delivery_end"], errors="coerce")
    epoch = pd.Timestamp("1970-01-01")

    result["delivery_start_hour"] = start_ts.dt.hour
    result["delivery_start_day"] = start_ts.dt.day
    result["delivery_start_month"] = start_ts.dt.month
    result["delivery_start_week"] = start_ts.dt.isocalendar().week.astype("float64")
    result["delivery_start_dow"] = start_ts.dt.dayofweek
    result["delivery_start_is_weekend"] = start_ts.dt.dayofweek.isin([5, 6]).astype(int)

    result["delivery_end_hour"] = end_ts.dt.hour
    result["delivery_end_dow"] = end_ts.dt.dayofweek

    _add_harmonics(result, "delivery_start_hour", 24)
    _add_harmonics(result, "delivery_start_dow", 7)

    result["hour_x_dow"] = result["delivery_start_hour"] * result["delivery_start_dow"]
    result["weekend_x_hour"] = result["delivery_start_is_weekend"] * result["delivery_start_hour"]
    result["month_x_market"] = result["market"].astype(str) + "_m" + result["delivery_start_month"].astype(str)

    result["delivery_duration_hours"] = (end_ts - start_ts).dt.total_seconds() / 3600.0
    result["delivery_start_ts"] = ((start_ts - epoch) // pd.Timedelta(seconds=1)).astype("int64")

    if {"air_temperature_2m", "dew_point_temperature_2m"}.issubset(result.columns):
        result["temp_dew_spread"] = result["air_temperature_2m"] - result["dew_point_temperature_2m"]

    if {"air_temperature_2m", "apparent_temperature_2m"}.issubset(result.columns):
        result["temp_apparent_spread"] = result["air_temperature_2m"] - result["apparent_temperature_2m"]

    if {"wind_speed_80m", "wind_speed_10m"}.issubset(result.columns):
        result["wind_speed_ratio_80m_10m"] = result["wind_speed_80m"] / (result["wind_speed_10m"].abs() + 1e-3)

    if {
        "global_horizontal_irradiance",
        "diffuse_horizontal_irradiance",
        "direct_normal_irradiance",
    }.issubset(result.columns):
        result["irradiance_total"] = (
            result["global_horizontal_irradiance"]
            + result["diffuse_horizontal_irradiance"]
            + result["direct_normal_irradiance"]
        )

    if {"cloud_cover_total", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"}.issubset(result.columns):
        result["cloud_cover_layers_sum"] = (
            result["cloud_cover_low"] + result["cloud_cover_mid"] + result["cloud_cover_high"]
        )
        result["cloud_cover_gap"] = result["cloud_cover_total"] - result["cloud_cover_layers_sum"]

    return result.drop(columns=["delivery_start", "delivery_end"])


def add_residual_load_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_res = train_df.copy()
    test_res = test_df.copy()

    demand_col = "demand_forecast" if "demand_forecast" in train_res.columns else "load_forecast"
    required = {demand_col, "wind_forecast", "solar_forecast"}
    if not required.issubset(train_res.columns) or not required.issubset(test_res.columns):
        return train_res, test_res

    train_res["net_load"] = (
        train_res[demand_col].astype(float) - train_res["wind_forecast"].astype(float) - train_res["solar_forecast"].astype(float)
    )
    test_res["net_load"] = (
        test_res[demand_col].astype(float) - test_res["wind_forecast"].astype(float) - test_res["solar_forecast"].astype(float)
    )
    train_res["residual_load"] = train_res["net_load"]
    test_res["residual_load"] = test_res["net_load"]

    train_res = train_res.sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)
    test_res = test_res.sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)

    train_res["residual_load_pctl"] = np.nan
    test_res["residual_load_pctl"] = np.nan
    train_res["residual_load_bin10"] = np.nan
    test_res["residual_load_bin10"] = np.nan
    train_res["residual_load_z"] = np.nan
    test_res["residual_load_z"] = np.nan

    for market in sorted(train_res["market"].dropna().unique()):
        train_mask = train_res["market"] == market
        test_mask = test_res["market"] == market

        train_vals = train_res.loc[train_mask, "residual_load"].to_numpy(dtype=float)
        if len(train_vals) == 0:
            continue

        sorted_vals = np.sort(train_vals)
        denom = max(len(sorted_vals) - 1, 1)

        def empirical_percentile(values: np.ndarray) -> np.ndarray:
            left = np.searchsorted(sorted_vals, values, side="left")
            right = np.searchsorted(sorted_vals, values, side="right")
            rank = 0.5 * (left + right)
            return rank / denom

        train_pct = empirical_percentile(train_vals)
        train_res.loc[train_mask, "residual_load_pctl"] = train_pct

        if test_mask.any():
            test_vals = test_res.loc[test_mask, "residual_load"].to_numpy(dtype=float)
            test_res.loc[test_mask, "residual_load_pctl"] = empirical_percentile(test_vals)

        quantile_edges = np.quantile(sorted_vals, np.linspace(0.0, 1.0, 11))
        quantile_edges = np.unique(quantile_edges)
        if len(quantile_edges) > 1:
            train_bins = pd.cut(train_res.loc[train_mask, "residual_load"], bins=quantile_edges, labels=False, include_lowest=True)
            test_bins = pd.cut(test_res.loc[test_mask, "residual_load"], bins=quantile_edges, labels=False, include_lowest=True)
            train_res.loc[train_mask, "residual_load_bin10"] = train_bins.astype(float)
            test_res.loc[test_mask, "residual_load_bin10"] = test_bins.astype(float)

        mean_val = float(np.mean(train_vals))
        std_val = float(np.std(train_vals))
        std_val = std_val if std_val > 1e-9 else 1.0
        train_res.loc[train_mask, "residual_load_z"] = (train_res.loc[train_mask, "residual_load"] - mean_val) / std_val
        if test_mask.any():
            test_res.loc[test_mask, "residual_load_z"] = (test_res.loc[test_mask, "residual_load"] - mean_val) / std_val

    train_res["is_high_residual"] = (train_res["residual_load_pctl"] >= 0.80).astype(int)
    test_res["is_high_residual"] = (test_res["residual_load_pctl"] >= 0.80).astype(int)

    comb_train = train_res.copy()
    comb_test = test_res.copy()
    comb_train["__is_train"] = 1
    comb_test["__is_train"] = 0
    combined = pd.concat([comb_train, comb_test], ignore_index=True, sort=False)
    combined = combined.sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)

    grouped = combined.groupby("market", sort=False)
    combined["residual_load_lag1"] = grouped["residual_load"].shift(1)
    combined["residual_load_lag24"] = grouped["residual_load"].shift(24)
    combined["residual_load_delta_1"] = combined["residual_load"] - combined["residual_load_lag1"]
    combined["residual_load_delta_24"] = combined["residual_load"] - combined["residual_load_lag24"]

    residual_lag1 = grouped["residual_load"].shift(1)
    combined["residual_load_rollmean_30d"] = (
        residual_lag1.groupby(combined["market"]).rolling(30 * 24, min_periods=24).mean().reset_index(level=0, drop=True)
    )
    combined["residual_load_vs_30d"] = combined["residual_load"] - combined["residual_load_rollmean_30d"]

    train_out = combined.loc[combined["__is_train"] == 1].drop(columns=["__is_train"]).copy()
    test_out = combined.loc[combined["__is_train"] == 0].drop(columns=["__is_train"]).copy()

    train_out = train_out.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
    test_out = test_out.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
    return train_out, test_out


def add_train_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy().sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)
    grouped = result.groupby("market", sort=False)

    for lag in LAG_STEPS:
        result[f"target_lag_{lag}"] = grouped["target"].shift(lag)

    shifted_target = grouped["target"].shift(1)
    for window in ROLL_WINDOWS:
        result[f"target_roll_mean_{window}"] = (
            shifted_target.groupby(result["market"]).rolling(window, min_periods=4).mean().reset_index(level=0, drop=True)
        )
        result[f"target_roll_std_{window}"] = (
            shifted_target.groupby(result["market"]).rolling(window, min_periods=4).std().reset_index(level=0, drop=True)
        )

    if "load_forecast" in result.columns:
        lf_lag_1 = grouped["load_forecast"].shift(1)
        lf_lag_24 = grouped["load_forecast"].shift(24)
        result["load_forecast_delta_1"] = result["load_forecast"] - lf_lag_1
        result["load_forecast_delta_24"] = result["load_forecast"] - lf_lag_24

    return result.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)


def add_exogenous_lag_block(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    enabled_markets: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_aug = train_df.copy()
    test_aug = test_df.copy()

    candidate_cols = [
        "demand_forecast",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
        "net_load",
    ]
    exog_cols = [col for col in candidate_cols if col in train_aug.columns and col in test_aug.columns]
    if not exog_cols:
        return train_aug, test_aug

    comb_train = train_aug.copy()
    comb_test = test_aug.copy()
    comb_train["__is_train"] = 1
    comb_test["__is_train"] = 0
    combined = pd.concat([comb_train, comb_test], ignore_index=True, sort=False)
    combined = combined.sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)
    grouped = combined.groupby("market", sort=False)

    enabled_mask = np.ones(len(combined), dtype=bool)
    if enabled_markets is not None:
        enabled_mask = combined["market"].astype(str).isin(sorted(enabled_markets)).to_numpy()
    combined["exog_lag_enabled"] = enabled_mask.astype(int)

    for col in exog_cols:
        series = combined[col].astype(float)
        shifted = grouped[col].shift(1)

        for lag in EXOG_LAG_STEPS:
            lag_col = f"{col}_lag_{lag}"
            ramp_col = f"{col}_ramp_{lag}"
            combined[lag_col] = grouped[col].shift(lag)
            combined[ramp_col] = series - combined[lag_col]
            if enabled_markets is not None:
                combined.loc[~enabled_mask, lag_col] = 0.0
                combined.loc[~enabled_mask, ramp_col] = 0.0

        for window in EXOG_ROLL_WINDOWS:
            roll_mean_col = f"{col}_roll_mean_{window}"
            roll_std_col = f"{col}_roll_std_{window}"
            rolling_grouped = shifted.groupby(combined["market"]).rolling(window, min_periods=4)
            combined[roll_mean_col] = rolling_grouped.mean().reset_index(level=0, drop=True)
            combined[roll_std_col] = rolling_grouped.std().reset_index(level=0, drop=True)
            if enabled_markets is not None:
                combined.loc[~enabled_mask, roll_mean_col] = 0.0
                combined.loc[~enabled_mask, roll_std_col] = 0.0

    train_out = combined.loc[combined["__is_train"] == 1].drop(columns=["__is_train"]).copy()
    test_out = combined.loc[combined["__is_train"] == 0].drop(columns=["__is_train"]).copy()
    train_out = train_out.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
    test_out = test_out.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
    return train_out, test_out


def add_test_base_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy().sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)

    for lag in LAG_STEPS:
        result[f"target_lag_{lag}"] = np.nan

    for window in ROLL_WINDOWS:
        result[f"target_roll_mean_{window}"] = np.nan
        result[f"target_roll_std_{window}"] = np.nan

    if "load_forecast" in result.columns:
        grouped = result.groupby("market", sort=False)
        lf_lag_1 = grouped["load_forecast"].shift(1)
        lf_lag_24 = grouped["load_forecast"].shift(24)
        result["load_forecast_delta_1"] = result["load_forecast"] - lf_lag_1
        result["load_forecast_delta_24"] = result["load_forecast"] - lf_lag_24

    return result.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)


def make_market_day_folds(
    market_df: pd.DataFrame,
    n_splits: int,
    purge_days: int,
    min_train_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    day_series = pd.to_datetime(market_df["delivery_start_ts"], unit="s", errors="coerce").dt.floor("D")
    unique_days = np.array(sorted(day_series.dropna().unique()))

    if len(unique_days) < (n_splits + 2):
        return []

    val_size_days = max(2, len(unique_days) // (n_splits + 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold in range(n_splits):
        train_end = val_size_days * (fold + 1)
        if train_end < min_train_days:
            continue

        valid_start = train_end + purge_days
        valid_end = min(valid_start + val_size_days, len(unique_days))
        if valid_end <= valid_start:
            break

        train_days = unique_days[:train_end]
        valid_days = unique_days[valid_start:valid_end]

        train_idx = market_df.index[day_series.isin(train_days)].to_numpy()
        valid_idx = market_df.index[day_series.isin(valid_days)].to_numpy()

        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue

        folds.append((train_idx, valid_idx))

    return folds


def _f1_score_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return 2.0 * precision * recall / (precision + recall + 1e-9)


def _fit_ridge_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: list[str],
) -> dict[str, object]:
    if not numeric_cols:
        return {"kind": "constant", "mean": float(y_train.mean()), "cols": [], "fill": {}}

    fill = {col: float(x_train[col].median()) for col in numeric_cols}
    x_mat = x_train[numeric_cols].copy()
    for col, val in fill.items():
        x_mat[col] = x_mat[col].fillna(val)

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=25.0, solver="svd"),
    )
    model.fit(x_mat, y_train)
    return {"kind": "ridge", "model": model, "cols": numeric_cols, "fill": fill}


def _predict_baseline(
    baseline: dict[str, object],
    x_df: pd.DataFrame,
) -> np.ndarray:
    if baseline["kind"] == "constant":
        return np.full(len(x_df), float(baseline["mean"]), dtype=float)

    cols: list[str] = baseline["cols"]  # type: ignore[assignment]
    fill: dict[str, float] = baseline["fill"]  # type: ignore[assignment]
    model = baseline["model"]  # type: ignore[assignment]

    x_mat = x_df[cols].copy()
    for col, val in fill.items():
        x_mat[col] = x_mat[col].fillna(val)
    return model.predict(x_mat)


def _fit_market_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_cols: list[str],
    numeric_cols: list[str],
    seed: int,
    use_gpu: bool,
    gpu_devices: str,
    fast_dev: bool,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    catboost_device_params: dict[str, str] = {}
    if use_gpu:
        catboost_device_params = {
            "task_type": "GPU",
            "devices": gpu_devices,
        }

    pos_spike_thr = float(y_train.quantile(0.95))
    neg_spike_thr = float(y_train.quantile(0.05))

    spike_labels_train = ((y_train >= pos_spike_thr) | (y_train <= neg_spike_thr)).astype(int)
    peak_abs_thr = float(np.quantile(np.abs(y_train.to_numpy(dtype=float)), 0.9))
    peak_labels_train = (np.abs(y_train.to_numpy(dtype=float)) >= peak_abs_thr).astype(int)
    spike_weights = np.ones(len(y_train), dtype=float)
    spike_weights[y_train.values >= pos_spike_thr] = 4.0
    spike_weights[y_train.values <= neg_spike_thr] = 3.0
    peak_weights = np.where(peak_labels_train == 1, 5.0, 1.0)

    normal_iterations = 900 if fast_dev else 2800
    normal_lr = 0.05 if fast_dev else 0.028
    normal_depth = 7 if fast_dev else 8
    normal_es_rounds = 100 if fast_dev else 200
    spike_iterations = 1100 if fast_dev else 3200
    spike_lr = 0.04 if fast_dev else 0.022
    spike_depth = 7 if fast_dev else 8
    spike_es_rounds = 100 if fast_dev else 200
    spike_clf_iterations = 300 if fast_dev else 900
    peak_iterations = 1200 if fast_dev else 3400
    peak_lr = 0.04 if fast_dev else 0.02
    peak_depth = 7 if fast_dev else 8
    peak_es_rounds = 120 if fast_dev else 220
    peak_clf_iterations = 320 if fast_dev else 900
    hgb_max_iter = 250 if fast_dev else 700
    hgb_depth = 6 if fast_dev else 7

    cat_normal = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=normal_iterations,
        learning_rate=normal_lr,
        depth=normal_depth,
        l2_leaf_reg=24,
        bagging_temperature=0.5,
        random_strength=0.9,
        random_seed=seed,
        verbose=0,
        **catboost_device_params,
    )
    cat_normal.fit(
        x_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=normal_es_rounds,
    )

    cat_spike = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="RMSE",
        iterations=spike_iterations,
        learning_rate=spike_lr,
        depth=spike_depth,
        l2_leaf_reg=28,
        bagging_temperature=0.7,
        random_strength=1.1,
        random_seed=seed + 13,
        verbose=0,
        **catboost_device_params,
    )
    cat_spike.fit(
        x_train,
        y_train,
        cat_features=cat_cols,
        sample_weight=spike_weights,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=spike_es_rounds,
    )

    cat_peak = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=peak_iterations,
        learning_rate=peak_lr,
        depth=peak_depth,
        l2_leaf_reg=30,
        bagging_temperature=0.9,
        random_strength=1.2,
        random_seed=seed + 29,
        verbose=0,
        **catboost_device_params,
    )
    cat_peak.fit(
        x_train,
        y_train,
        cat_features=cat_cols,
        sample_weight=peak_weights,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=peak_es_rounds,
    )

    spike_classifier: CatBoostClassifier | None = None
    peak_classifier: CatBoostClassifier | None = None
    p_spike_valid = np.full(len(x_valid), 0.15)
    p_peak_valid = np.full(len(x_valid), 0.10)

    if spike_labels_train.sum() >= 30 and spike_labels_train.sum() < len(spike_labels_train) - 30:
        spike_classifier = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=spike_clf_iterations,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=14,
            random_seed=seed + 101,
            verbose=0,
            **catboost_device_params,
        )
        spike_classifier.fit(
            x_train,
            spike_labels_train,
            cat_features=cat_cols,
        )
        p_spike_valid = spike_classifier.predict_proba(x_valid)[:, 1]
    p_spike_valid = np.clip(p_spike_valid, 0.0, 1.0)

    if peak_labels_train.sum() >= 30 and peak_labels_train.sum() < len(peak_labels_train) - 30:
        peak_classifier = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=peak_clf_iterations,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=16,
            random_seed=seed + 131,
            verbose=0,
            **catboost_device_params,
        )
        peak_classifier.fit(
            x_train,
            peak_labels_train,
            cat_features=cat_cols,
        )
        p_peak_valid = peak_classifier.predict_proba(x_valid)[:, 1]
    p_peak_valid = np.clip(p_peak_valid, 0.0, 1.0)

    cat_normal_valid = np.asarray(cat_normal.predict(x_valid), dtype=float)
    cat_spike_valid = np.asarray(cat_spike.predict(x_valid), dtype=float)
    cat_peak_valid = np.asarray(cat_peak.predict(x_valid), dtype=float)

    hgb = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_depth=hgb_depth,
        max_iter=hgb_max_iter,
        min_samples_leaf=30,
        l2_regularization=1.5,
        random_state=seed,
    )
    hgb.fit(x_train[numeric_cols], y_train)
    hgb_valid = np.asarray(hgb.predict(x_valid[numeric_cols]), dtype=float)

    models = {
        "cat_normal": cat_normal,
        "cat_spike": cat_spike,
        "cat_peak": cat_peak,
        "spike_classifier": spike_classifier,
        "peak_classifier": peak_classifier,
        "hgb": hgb,
        "pos_spike_thr": pos_spike_thr,
        "neg_spike_thr": neg_spike_thr,
    }

    valid_outputs = {
        "cat_normal": cat_normal_valid,
        "cat_spike": cat_spike_valid,
        "cat_peak": cat_peak_valid,
        "hgb": hgb_valid,
        "p_spike": p_spike_valid,
        "p_peak": p_peak_valid,
    }
    return models, valid_outputs


def _predict_model_components(
    row_df: pd.DataFrame,
    numeric_cols: list[str],
    models: dict[str, object],
) -> dict[str, float]:
    cat_normal: CatBoostRegressor = models["cat_normal"]  # type: ignore[assignment]
    cat_spike: CatBoostRegressor = models["cat_spike"]  # type: ignore[assignment]
    cat_peak: CatBoostRegressor = models["cat_peak"]  # type: ignore[assignment]
    spike_classifier: CatBoostClassifier | None = models["spike_classifier"]  # type: ignore[assignment]
    peak_classifier: CatBoostClassifier | None = models["peak_classifier"]  # type: ignore[assignment]
    hgb: HistGradientBoostingRegressor = models["hgb"]  # type: ignore[assignment]

    cat_normal_pred = float(cat_normal.predict(row_df)[0])
    cat_spike_pred = float(cat_spike.predict(row_df)[0])
    cat_peak_pred = float(cat_peak.predict(row_df)[0])

    if spike_classifier is not None:
        p_spike = float(spike_classifier.predict_proba(row_df)[:, 1][0])
    else:
        p_spike = 0.15
    if peak_classifier is not None:
        p_peak = float(peak_classifier.predict_proba(row_df)[:, 1][0])
    else:
        p_peak = 0.10

    hgb_pred = float(hgb.predict(row_df[numeric_cols])[0])
    return {
        "cat_normal": cat_normal_pred,
        "cat_spike": cat_spike_pred,
        "cat_peak": cat_peak_pred,
        "hgb": hgb_pred,
        "p_spike": float(np.clip(p_spike, 0.0, 1.0)),
        "p_peak": float(np.clip(p_peak, 0.0, 1.0)),
    }


def _build_meta_matrix(
    cat_normal: np.ndarray,
    cat_spike: np.ndarray,
    cat_peak: np.ndarray,
    hgb: np.ndarray,
    p_spike: np.ndarray,
    p_peak: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            cat_normal,
            cat_spike,
            cat_peak,
            hgb,
            p_spike,
            p_peak,
            p_peak * cat_peak,
        ]
    )


def _stack_residual_prediction(
    stacker,
    components: dict[str, float],
) -> float:
    row = _build_meta_matrix(
        cat_normal=np.array([components["cat_normal"]], dtype=float),
        cat_spike=np.array([components["cat_spike"]], dtype=float),
        cat_peak=np.array([components["cat_peak"]], dtype=float),
        hgb=np.array([components["hgb"]], dtype=float),
        p_spike=np.array([components["p_spike"]], dtype=float),
        p_peak=np.array([components["p_peak"]], dtype=float),
    )
    return float(stacker.predict(row)[0])


def _hybrid_predict_market(
    test_market_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    cat_cols: list[str],
    num_fill_values: dict[str, float],
    direct_feature_cols: list[str],
    direct_numeric_cols: list[str],
    direct_cat_cols: list[str],
    direct_num_fill_values: dict[str, float],
    history_targets: list[float],
    recursive_models: dict[str, object],
    direct_mid_models: dict[str, object],
    direct_long_models: dict[str, object],
    stacker,
    baseline_model: dict[str, object],
    recursive_steps: int,
    mid_horizon_steps: int,
) -> pd.Series:
    preds: list[float] = []

    for step_idx, idx in enumerate(test_market_df.index):
        use_recursive = step_idx < recursive_steps
        use_mid_horizon = recursive_steps <= step_idx < (recursive_steps + mid_horizon_steps)

        if not use_recursive:
            row_direct = test_market_df.loc[idx, direct_feature_cols].copy()
            row_df_direct = pd.DataFrame([row_direct], columns=direct_feature_cols)
            row_direct_id = row_df_direct.index[0]

            for col, fill_value in direct_num_fill_values.items():
                if pd.isna(row_df_direct.at[row_direct_id, col]):
                    row_df_direct.at[row_direct_id, col] = fill_value

            for col in direct_cat_cols:
                row_df_direct.at[row_direct_id, col] = str(row_df_direct.at[row_direct_id, col])

            components = _predict_model_components(
                row_df=row_df_direct,
                numeric_cols=direct_numeric_cols,
                models=direct_mid_models if use_mid_horizon else direct_long_models,
            )
            baseline_pred = float(_predict_baseline(baseline_model, row_df_direct)[0])
            residual_pred = _stack_residual_prediction(
                stacker=stacker,
                components=components,
            )
            pred = baseline_pred + residual_pred
            preds.append(pred)
            history_targets.append(pred)
            continue

        row = test_market_df.loc[idx, feature_cols].copy()

        for lag in LAG_STEPS:
            row[f"target_lag_{lag}"] = history_targets[-lag] if len(history_targets) >= lag else np.nan

        for window in ROLL_WINDOWS:
            if len(history_targets) == 0:
                row[f"target_roll_mean_{window}"] = np.nan
                row[f"target_roll_std_{window}"] = np.nan
            else:
                tail = np.array(history_targets[-window:], dtype=float)
                row[f"target_roll_mean_{window}"] = float(np.mean(tail))
                row[f"target_roll_std_{window}"] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0

        row_df = pd.DataFrame([row], columns=feature_cols)
        row_id = row_df.index[0]

        for col, fill_value in num_fill_values.items():
            if pd.isna(row_df.at[row_id, col]):
                row_df.at[row_id, col] = fill_value

        for col in cat_cols:
            row_df.at[row_id, col] = str(row_df.at[row_id, col])

        components = _predict_model_components(
            row_df=row_df,
            numeric_cols=numeric_cols,
            models=recursive_models,
        )
        baseline_pred = float(_predict_baseline(baseline_model, row_df)[0])
        residual_pred = _stack_residual_prediction(
            stacker=stacker,
            components=components,
        )
        pred = baseline_pred + residual_pred
        preds.append(pred)
        history_targets.append(pred)

    return pd.Series(preds, index=test_market_df.index, dtype=float)


def _build_default_config(config_name: str) -> dict:
    return {
        "config_name": config_name,
        "fast_dev": False,
        "exclude_2023": False,
        "exclude_2023_keep_from_month": 10,
        "use_gpu": False,
        "gpu_devices": "0",
        "recursive_steps": 12,
        "mid_horizon_steps": 48,
        "mid_expert_recent_fraction": 0.45,
        "n_splits": 5,
        "purge_days": 2,
        "min_train_days": 10,
        "base_seed": 42,
        "final_seed": 999,
        "correction_alpha": 0.6,
        "min_market_rows": 400,
    }


def run_with_ledger(
    config_name: str,
    official_run: bool,
    lb_score: float | None,
    fast_dev: bool = False,
    exclude_2023: bool = False,
    exclude_2023_keep_from_month: int = 10,
    use_gpu: bool = False,
    gpu_devices: str = "0",
) -> dict[str, str]:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    runs_dir = base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test_for_participants.csv"
    sample_submission_path = data_dir / "sample_submission.csv"

    for path in [train_path, test_path, sample_submission_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    git_info = _get_git_info(base_dir)
    if official_run and bool(git_info["dirty_repo"]):
        raise RuntimeError("Official runs require a clean repository (dirty_repo=false).")

    started_at = datetime.now(timezone.utc).isoformat()
    safe_config_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in config_name).strip("-")
    if not safe_config_name:
        safe_config_name = "default"
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    git_sha = str(git_info["git_sha"])
    run_id = f"{run_timestamp}_{git_sha}_{safe_config_name}"

    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    train_log_path = run_dir / "train.log"
    params_path = run_dir / "params.yaml"
    metrics_path = run_dir / "metrics.json"
    oof_path = run_dir / "oof.csv"
    submission_path = run_dir / "submission.csv"
    model_manifest_path = run_dir / "models_manifest.txt"
    registry_path = base_dir / "experiments.csv"

    config = _build_default_config(safe_config_name)
    config["fast_dev"] = bool(fast_dev)
    config["exclude_2023"] = bool(exclude_2023)
    keep_from_month = int(exclude_2023_keep_from_month)
    keep_from_month = min(max(keep_from_month, 1), 12)
    config["exclude_2023_keep_from_month"] = keep_from_month
    config["use_gpu"] = bool(use_gpu)
    config["gpu_devices"] = str(gpu_devices)
    if bool(config["fast_dev"]):
        config["recursive_steps"] = 6
        config["mid_horizon_steps"] = 24
        config["mid_expert_recent_fraction"] = 0.35
        config["n_splits"] = 3
        config["purge_days"] = 1
        config["min_train_days"] = 7
    config_hash = _hash_text(_stable_json(config))
    train_hash = _sha256_file(train_path)
    test_hash = _sha256_file(test_path)
    data_hash = _hash_text(f"{train_hash}:{test_hash}")

    _write_params_yaml(
        params_path,
        {
            "run_id": run_id,
            "started_at": started_at,
            "git_sha": git_info["git_sha"],
            "git_branch": git_info["git_branch"],
            "dirty_repo": bool(git_info["dirty_repo"]),
            "data_hash": data_hash,
            "config_hash": config_hash,
            "config": config,
        },
    )

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    with train_log_path.open("w", encoding="utf-8") as log_file:
        tee_stream = _Tee(orig_stdout, log_file)
        sys.stdout = tee_stream
        sys.stderr = tee_stream
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            sample_submission = pd.read_csv(sample_submission_path)

            if bool(config["exclude_2023"]):
                train_start = pd.to_datetime(train_df["delivery_start"], errors="coerce")
                keep_from_month = int(config["exclude_2023_keep_from_month"])
                keep_2023_tail = (train_start.dt.year == 2023) & (train_start.dt.month >= keep_from_month)
                keep_mask = (train_start.dt.year != 2023) | keep_2023_tail
                removed_count = int((~keep_mask).sum())
                train_df = train_df.loc[keep_mask].reset_index(drop=True)
                kept_2023_count = int(keep_2023_tail.sum())
                print(
                    "Excluded most 2023 observations from training: "
                    f"removed {removed_count} rows, kept {kept_2023_count} rows from "
                    f"2023-{keep_from_month:02d} onward"
                )
                if train_df.empty:
                    raise ValueError("Training data is empty after excluding 2023 observations.")

            train_df = add_time_features(train_df)
            test_df = add_time_features(test_df)

            train_df, test_df = add_residual_load_features(train_df, test_df)
            train_df, test_df = add_exogenous_lag_block(train_df, test_df)

            train_df = add_train_lag_features(train_df)
            test_df = add_test_base_features(test_df)

            feature_cols = [col for col in train_df.columns if col not in ["id", "target", "market"]]
            constant_cols = [col for col in feature_cols if train_df[col].nunique(dropna=False) <= 1]
            if constant_cols:
                train_df = train_df.drop(columns=constant_cols)
                test_df = test_df.drop(columns=constant_cols)
                feature_cols = [col for col in feature_cols if col not in constant_cols]

            cat_cols = train_df[feature_cols].select_dtypes(include=["object", "category", "string"]).columns.tolist()
            numeric_cols = [col for col in feature_cols if col not in cat_cols]
            direct_feature_cols = [
                col
                for col in feature_cols
                if not col.startswith("target_lag_") and not col.startswith("target_roll_")
            ]
            direct_cat_cols = [col for col in cat_cols if col in direct_feature_cols]
            direct_numeric_cols = [col for col in direct_feature_cols if col not in direct_cat_cols]

            # Global model (trained on all markets) provides a stabilizing signal for local stackers.
            global_train_x = train_df[direct_feature_cols].copy()
            global_test_x = test_df[direct_feature_cols].copy()

            global_num_fill = {col: float(global_train_x[col].median()) for col in direct_numeric_cols}
            for col, fill_val in global_num_fill.items():
                global_train_x[col] = global_train_x[col].fillna(fill_val)
                global_test_x[col] = global_test_x[col].fillna(fill_val)
            for col in direct_cat_cols:
                global_train_x[col] = global_train_x[col].fillna("missing").astype(str)
                global_test_x[col] = global_test_x[col].fillna("missing").astype(str)

            global_oof = np.full(len(train_df), np.nan, dtype=float)
            global_folds = make_market_day_folds(
                market_df=train_df,
                n_splits=int(config["n_splits"]),
                purge_days=int(config["purge_days"]),
                min_train_days=int(config["min_train_days"]),
            )
            if len(global_folds) < 2:
                raise ValueError("Global model requires at least 2 CV folds.")
            global_fold_iterations = 700 if bool(config["fast_dev"]) else 1600
            global_fold_es_rounds = 80 if bool(config["fast_dev"]) else 150

            for fold_id, (g_train_idx, g_valid_idx) in enumerate(global_folds, start=1):
                global_device_params: dict[str, str] = {}
                if bool(config["use_gpu"]):
                    global_device_params = {
                        "task_type": "GPU",
                        "devices": str(config["gpu_devices"]),
                    }
                global_model = CatBoostRegressor(
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    iterations=global_fold_iterations,
                    learning_rate=0.03,
                    depth=8,
                    l2_leaf_reg=24,
                    bagging_temperature=0.5,
                    random_strength=0.8,
                    random_seed=int(config["base_seed"]) + fold_id + 500,
                    verbose=0,
                    **global_device_params,
                )
                global_model.fit(
                    global_train_x.iloc[g_train_idx],
                    train_df["target"].iloc[g_train_idx],
                    cat_features=direct_cat_cols,
                    eval_set=(global_train_x.iloc[g_valid_idx], train_df["target"].iloc[g_valid_idx]),
                    use_best_model=True,
                    early_stopping_rounds=global_fold_es_rounds,
                )
                global_oof[g_valid_idx] = np.asarray(global_model.predict(global_train_x.iloc[g_valid_idx]), dtype=float)

            global_valid_cov = float(np.mean(~np.isnan(global_oof)))
            print(f"Global OOF coverage: {global_valid_cov * 100:.2f}%")
            if global_valid_cov < 0.70:
                print(
                    "Warning: low global OOF coverage. "
                    "Global model may be weak in early folds."
                )

            final_global_device_params: dict[str, str] = {}
            if bool(config["use_gpu"]):
                final_global_device_params = {
                    "task_type": "GPU",
                    "devices": str(config["gpu_devices"]),
                }

            final_global_model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=900 if bool(config["fast_dev"]) else 2000,
                learning_rate=0.028,
                depth=8,
                l2_leaf_reg=26,
                bagging_temperature=0.5,
                random_strength=0.8,
                random_seed=int(config["final_seed"]) + 800,
                verbose=0,
                **final_global_device_params,
            )
            final_global_model.fit(
                global_train_x,
                train_df["target"],
                cat_features=direct_cat_cols,
            )
            global_test_pred = np.asarray(final_global_model.predict(global_test_x), dtype=float)

            print(f"Run ID: {run_id}")
            print(f"CatBoost device: {'GPU' if bool(config['use_gpu']) else 'CPU'} (devices={config['gpu_devices']})")
            print(f"Run mode: {'FAST_DEV' if bool(config['fast_dev']) else 'FULL'}")
            print(f"Using {len(feature_cols)} features total")
            print(f"Categorical ({len(cat_cols)}): {cat_cols}")
            print(
                "Inference mode: "
                f"recursive first {int(config['recursive_steps'])} steps, "
                f"mid-horizon expert next {int(config['mid_horizon_steps'])} steps, "
                "then long-horizon expert"
            )
            print("Blending mode: residual baseline + OOF stacker + global/local signal")
            print("Experts: normal + spike + peak (soft-gated)")

            predictions_by_id: dict[int, float] = {}
            market_scores: list[tuple[str, float]] = []
            market_diagnostics: list[dict[str, object]] = []
            model_manifest_rows: list[str] = []
            oof_frames: list[pd.DataFrame] = []

            for market in sorted(train_df["market"].dropna().unique()):
                market_train = train_df[train_df["market"] == market].sort_values(["delivery_start_ts", "id"]).reset_index(drop=True)
                market_test = test_df[test_df["market"] == market].sort_values(["delivery_start_ts", "id"]).reset_index(drop=True)

                if len(market_train) < int(config["min_market_rows"]) or market_test.empty:
                    continue

                for col in numeric_cols:
                    median_value = float(market_train[col].median())
                    market_train[col] = market_train[col].fillna(median_value)
                    market_test[col] = market_test[col].fillna(median_value)

                for col in cat_cols:
                    market_train[col] = market_train[col].fillna("missing").astype(str)
                    market_test[col] = market_test[col].fillna("missing").astype(str)

                folds = make_market_day_folds(
                    market_df=market_train,
                    n_splits=int(config["n_splits"]),
                    purge_days=int(config["purge_days"]),
                    min_train_days=int(config["min_train_days"]),
                )
                if len(folds) < 2:
                    continue

                y_all = market_train["target"].copy()
                oof_baseline = np.full(len(market_train), np.nan, dtype=float)
                oof_cat_normal = np.full(len(market_train), np.nan, dtype=float)
                oof_cat_spike = np.full(len(market_train), np.nan, dtype=float)
                oof_cat_peak = np.full(len(market_train), np.nan, dtype=float)
                oof_hgb = np.full(len(market_train), np.nan, dtype=float)
                oof_p_spike = np.full(len(market_train), np.nan, dtype=float)
                oof_p_peak = np.full(len(market_train), np.nan, dtype=float)
                fold_records: list[dict[str, float | int]] = []

                for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
                    x_train = market_train.iloc[train_idx][feature_cols].copy()
                    y_train = y_all.iloc[train_idx]
                    x_valid = market_train.iloc[valid_idx][feature_cols].copy()
                    y_valid = y_all.iloc[valid_idx]

                    baseline_model_fold = _fit_ridge_baseline(
                        x_train=x_train,
                        y_train=y_train,
                        numeric_cols=direct_numeric_cols,
                    )
                    baseline_train = _predict_baseline(baseline_model_fold, x_train)
                    baseline_valid = _predict_baseline(baseline_model_fold, x_valid)
                    y_train_residual = y_train.to_numpy(dtype=float) - baseline_train
                    y_valid_residual = y_valid.to_numpy(dtype=float) - baseline_valid

                    models, valid_outputs = _fit_market_models(
                        x_train=x_train,
                        y_train=pd.Series(y_train_residual, index=y_train.index),
                        x_valid=x_valid,
                        y_valid=pd.Series(y_valid_residual, index=y_valid.index),
                        cat_cols=cat_cols,
                        numeric_cols=numeric_cols,
                        seed=int(config["base_seed"]) + fold_id,
                        use_gpu=bool(config["use_gpu"]),
                        gpu_devices=str(config["gpu_devices"]),
                        fast_dev=bool(config["fast_dev"]),
                    )

                    cat_normal_pred = np.asarray(valid_outputs["cat_normal"], dtype=float)
                    cat_spike_pred = np.asarray(valid_outputs["cat_spike"], dtype=float)
                    cat_peak_pred = np.asarray(valid_outputs["cat_peak"], dtype=float)
                    hgb_pred = np.asarray(valid_outputs["hgb"], dtype=float)
                    p_spike_pred = np.asarray(valid_outputs["p_spike"], dtype=float)
                    p_peak_pred = np.asarray(valid_outputs["p_peak"], dtype=float)

                    oof_baseline[valid_idx] = baseline_valid
                    oof_cat_normal[valid_idx] = cat_normal_pred
                    oof_cat_spike[valid_idx] = cat_spike_pred
                    oof_cat_peak[valid_idx] = cat_peak_pred
                    oof_hgb[valid_idx] = hgb_pred
                    oof_p_spike[valid_idx] = p_spike_pred
                    oof_p_peak[valid_idx] = p_peak_pred

                    fold_blend = baseline_valid + (
                        0.35 * (p_spike_pred * cat_spike_pred + (1.0 - p_spike_pred) * cat_normal_pred)
                        + 0.35 * hgb_pred
                        + 0.30 * (p_peak_pred * cat_peak_pred + (1.0 - p_peak_pred) * cat_normal_pred)
                    )
                    rmse_fold = float(mean_squared_error(y_valid, fold_blend) ** 0.5)
                    fold_records.append(
                        {
                            "fold": fold_id,
                            "rmse_fold": rmse_fold,
                        }
                    )
                    print(f"{market} | Fold {fold_id}/{len(folds)} | Residual blend RMSE={rmse_fold:.6f}")

                fold_rmse_list = [float(record["rmse_fold"]) for record in fold_records]
                if not fold_rmse_list:
                    continue
                fold_rmse_mean_all = float(np.mean(fold_rmse_list))
                fold_rmse_mean_last2 = float(np.mean(fold_rmse_list[-2:]))
                fold_rmse_std = float(np.std(fold_rmse_list))
                fold_rmse_worst = float(np.max(fold_rmse_list))

                print(
                    f"{market} | Fold diagnostics | mean_all={fold_rmse_mean_all:.6f}, "
                    f"mean_last2={fold_rmse_mean_last2:.6f}, std={fold_rmse_std:.6f}, "
                    f"worst={fold_rmse_worst:.6f}"
                )

                valid_mask = (
                    (~np.isnan(oof_baseline))
                    & (~np.isnan(oof_cat_normal))
                    & (~np.isnan(oof_cat_spike))
                    & (~np.isnan(oof_cat_peak))
                    & (~np.isnan(oof_hgb))
                    & (~np.isnan(oof_p_spike))
                    & (~np.isnan(oof_p_peak))
                )
                if not np.any(valid_mask):
                    continue

                meta_x = _build_meta_matrix(
                    cat_normal=oof_cat_normal[valid_mask],
                    cat_spike=oof_cat_spike[valid_mask],
                    cat_peak=oof_cat_peak[valid_mask],
                    hgb=oof_hgb[valid_mask],
                    p_spike=oof_p_spike[valid_mask],
                    p_peak=oof_p_peak[valid_mask],
                )
                y_residual_target = y_all[valid_mask].to_numpy(dtype=float) - oof_baseline[valid_mask]
                stacker = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=20.0, solver="svd"),
                )
                stacker.fit(meta_x, y_residual_target)

                oof_residual = stacker.predict(meta_x)
                oof_blend = oof_baseline[valid_mask] + oof_residual
                market_oof_rmse = float(mean_squared_error(y_all[valid_mask], oof_blend) ** 0.5)
                market_scores.append((market, market_oof_rmse))
                market_diagnostics.append(
                    {
                        "market": market,
                        "fold_rmse_list": fold_rmse_list,
                        "fold_rmse_mean_all": fold_rmse_mean_all,
                        "fold_rmse_mean_last2": fold_rmse_mean_last2,
                        "fold_rmse_std": fold_rmse_std,
                        "fold_rmse_worst": fold_rmse_worst,
                        "oof_rmse": market_oof_rmse,
                    }
                )

                oof_market = pd.DataFrame(
                    {
                        "id": market_train.loc[valid_mask, "id"].to_numpy(),
                        "market": market,
                        "target": y_all[valid_mask].to_numpy(),
                        "pred": oof_blend,
                    }
                )
                oof_frames.append(oof_market)

                residual_df = pd.DataFrame(
                    {
                        "hour": market_train.loc[valid_mask, "delivery_start_hour"].to_numpy(),
                        "dow": market_train.loc[valid_mask, "delivery_start_dow"].to_numpy(),
                        "residual": (y_all[valid_mask].to_numpy() - oof_blend),
                    }
                )
                residual_correction = residual_df.groupby(["hour", "dow"])["residual"].mean().to_dict()

                x_full = market_train[feature_cols].copy()
                y_full = y_all.copy()
                baseline_full = _fit_ridge_baseline(
                    x_train=x_full,
                    y_train=y_full,
                    numeric_cols=direct_numeric_cols,
                )

                holdout_size = max(24, int(len(market_train) * 0.1))
                x_fit = x_full.iloc[:-holdout_size]
                y_fit = y_full.iloc[:-holdout_size]
                x_cal = x_full.iloc[-holdout_size:]
                y_cal = y_full.iloc[-holdout_size:]
                baseline_fit = _predict_baseline(baseline_full, x_fit)
                baseline_cal = _predict_baseline(baseline_full, x_cal)

                final_models, _ = _fit_market_models(
                    x_train=x_fit,
                    y_train=pd.Series(y_fit.to_numpy(dtype=float) - baseline_fit, index=y_fit.index),
                    x_valid=x_cal,
                    y_valid=pd.Series(y_cal.to_numpy(dtype=float) - baseline_cal, index=y_cal.index),
                    cat_cols=cat_cols,
                    numeric_cols=numeric_cols,
                    seed=int(config["final_seed"]),
                    use_gpu=bool(config["use_gpu"]),
                    gpu_devices=str(config["gpu_devices"]),
                    fast_dev=bool(config["fast_dev"]),
                )

                recent_fraction = float(config["mid_expert_recent_fraction"])
                recent_fraction = min(max(recent_fraction, 0.15), 0.90)
                mid_recent_size = max(24 * 10, int(len(x_fit) * recent_fraction))
                if mid_recent_size >= len(x_fit):
                    mid_recent_size = max(24 * 5, len(x_fit) // 2)

                x_fit_mid = x_fit.iloc[-mid_recent_size:]
                y_fit_mid = y_fit.iloc[-mid_recent_size:]
                baseline_fit_mid = baseline_fit[-mid_recent_size:]

                final_direct_mid_models, _ = _fit_market_models(
                    x_train=x_fit_mid[direct_feature_cols],
                    y_train=pd.Series(y_fit_mid.to_numpy(dtype=float) - baseline_fit_mid, index=y_fit_mid.index),
                    x_valid=x_cal[direct_feature_cols],
                    y_valid=pd.Series(y_cal.to_numpy(dtype=float) - baseline_cal, index=y_cal.index),
                    cat_cols=direct_cat_cols,
                    numeric_cols=direct_numeric_cols,
                    seed=int(config["final_seed"]) + 37,
                    use_gpu=bool(config["use_gpu"]),
                    gpu_devices=str(config["gpu_devices"]),
                    fast_dev=bool(config["fast_dev"]),
                )

                final_direct_long_models, _ = _fit_market_models(
                    x_train=x_fit[direct_feature_cols],
                    y_train=pd.Series(y_fit.to_numpy(dtype=float) - baseline_fit, index=y_fit.index),
                    x_valid=x_cal[direct_feature_cols],
                    y_valid=pd.Series(y_cal.to_numpy(dtype=float) - baseline_cal, index=y_cal.index),
                    cat_cols=direct_cat_cols,
                    numeric_cols=direct_numeric_cols,
                    seed=int(config["final_seed"]) + 59,
                    use_gpu=bool(config["use_gpu"]),
                    gpu_devices=str(config["gpu_devices"]),
                    fast_dev=bool(config["fast_dev"]),
                )

                num_fill_values = {col: float(market_train[col].median()) for col in numeric_cols}
                direct_num_fill_values = {col: float(market_train[col].median()) for col in direct_numeric_cols}
                history_targets = y_full.astype(float).tolist()
                test_preds = _hybrid_predict_market(
                    test_market_df=market_test,
                    feature_cols=feature_cols,
                    numeric_cols=numeric_cols,
                    cat_cols=cat_cols,
                    num_fill_values=num_fill_values,
                    direct_feature_cols=direct_feature_cols,
                    direct_numeric_cols=direct_numeric_cols,
                    direct_cat_cols=direct_cat_cols,
                    direct_num_fill_values=direct_num_fill_values,
                    history_targets=history_targets,
                    recursive_models=final_models,
                    direct_mid_models=final_direct_mid_models,
                    direct_long_models=final_direct_long_models,
                    stacker=stacker,
                    baseline_model=baseline_full,
                    recursive_steps=int(config["recursive_steps"]),
                    mid_horizon_steps=int(config["mid_horizon_steps"]),
                )

                correction_values = [
                    float(config["correction_alpha"]) * residual_correction.get((int(row["delivery_start_hour"]), int(row["delivery_start_dow"])), 0.0)
                    for _, row in market_test.iterrows()
                ]
                calibrated_preds = test_preds.to_numpy() + np.array(correction_values, dtype=float)

                for i, pred in enumerate(calibrated_preds):
                    test_id = int(market_test.iloc[i]["id"])
                    predictions_by_id[test_id] = float(pred)

                stacker_ridge: Ridge = stacker.named_steps["ridge"]  # type: ignore[index]
                stacker_coef = np.asarray(stacker_ridge.coef_, dtype=float)
                model_manifest_rows.append(
                    "market="
                    f"{market}, oof_rmse={market_oof_rmse:.6f}, "
                    f"recursive_steps={int(config['recursive_steps'])}, "
                    f"mid_horizon_steps={int(config['mid_horizon_steps'])}, "
                    f"fold_rmse_mean_last2={fold_rmse_mean_last2:.6f}, "
                    f"fold_rmse_worst={fold_rmse_worst:.6f}, "
                    f"stacker_intercept={float(stacker_ridge.intercept_):.6f}, "
                    f"stacker_coef={','.join(f'{c:.6f}' for c in stacker_coef.tolist())}"
                )
                print(
                    f"{market} | OOF RMSE={market_oof_rmse:.6f} | learned stacker with {len(stacker_coef)} features"
                )

            submission = sample_submission[["id"]].copy()
            submission["target"] = submission["id"].map(predictions_by_id)

            if submission["target"].isna().any():
                missing_count = int(submission["target"].isna().sum())
                raise ValueError(f"Missing predictions for {missing_count} ids.")

            submission.to_csv(submission_path, index=False)

            if oof_frames:
                oof_df = pd.concat(oof_frames, ignore_index=True)
                oof_df.to_csv(oof_path, index=False)
                cv_rmse = float(mean_squared_error(oof_df["target"], oof_df["pred"]) ** 0.5)
            else:
                oof_df = pd.DataFrame(columns=["id", "market", "target", "pred"])
                oof_df.to_csv(oof_path, index=False)
                cv_rmse = float("nan")

            model_manifest_path.write_text("\n".join(model_manifest_rows) + "\n", encoding="utf-8")

            if market_scores:
                print("Per-market CV RMSE (day folds):")
                for market, score in sorted(market_scores, key=lambda x: x[1]):
                    print(f"  {market}: {score:.6f}")
                print(f"Mean per-market RMSE: {np.mean([score for _, score in market_scores]):.6f}")
            if market_diagnostics:
                mean_recent2 = float(np.mean([float(d["fold_rmse_mean_last2"]) for d in market_diagnostics]))
                mean_allfold = float(np.mean([float(d["fold_rmse_mean_all"]) for d in market_diagnostics]))
                mean_worst = float(np.mean([float(d["fold_rmse_worst"]) for d in market_diagnostics]))
                print(
                    f"Fold diagnostics (market-mean) | mean_all={mean_allfold:.6f}, "
                    f"mean_last2={mean_recent2:.6f}, mean_worst={mean_worst:.6f}"
                )
            print(submission.head())

            metrics = {
                "run_id": run_id,
                "started_at": started_at,
                "cv_rmse": cv_rmse,
                "lb_score": lb_score,
                "market_scores": [{"market": market, "rmse": score} for market, score in market_scores],
                "market_diagnostics": market_diagnostics,
                "fold_rmse_mean_all": (
                    float(np.mean([float(d["fold_rmse_mean_all"]) for d in market_diagnostics]))
                    if market_diagnostics
                    else float("nan")
                ),
                "fold_rmse_mean_last2": (
                    float(np.mean([float(d["fold_rmse_mean_last2"]) for d in market_diagnostics]))
                    if market_diagnostics
                    else float("nan")
                ),
                "fold_rmse_mean_worst": (
                    float(np.mean([float(d["fold_rmse_worst"]) for d in market_diagnostics]))
                    if market_diagnostics
                    else float("nan")
                ),
                "git_sha": git_sha,
                "git_branch": str(git_info["git_branch"]),
                "dirty_repo": bool(git_info["dirty_repo"]),
                "data_hash": data_hash,
                "config_hash": config_hash,
                "config": config,
                "model_path": str(model_manifest_path),
                "submission_path": str(submission_path),
                "oof_path": str(oof_path),
            }
            metrics_path.write_text(_stable_json(metrics) + "\n", encoding="utf-8")

            registry_row = {
                "run_id": run_id,
                "git_sha": git_sha,
                "git_branch": str(git_info["git_branch"]),
                "dirty_repo": str(bool(git_info["dirty_repo"])).lower(),
                "data_hash": data_hash,
                "config_hash": config_hash,
                "seed": str(config["base_seed"]),
                "cv_rmse": "" if np.isnan(cv_rmse) else f"{cv_rmse:.6f}",
                "lb_score": "" if lb_score is None else str(lb_score),
                "model_path": str(model_manifest_path.relative_to(base_dir)),
                "submission_path": str(submission_path.relative_to(base_dir)),
                "started_at": started_at,
            }
            _append_experiment_registry(registry_path, registry_row)

            latest_submission_path = base_dir / "submission_per_market.csv"
            submission.to_csv(latest_submission_path, index=False)

            print(f"Saved run artifacts under: {run_dir}")
            print(f"Saved latest submission copy: {latest_submission_path}")
            print(f"Registry updated: {registry_path}")
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "registry": str(registry_path),
        "submission": str(submission_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train per-market model with strict run ledger logging.")
    parser.add_argument("--config-name", default="per_market_v1", help="Config name for run_id and ledger entries.")
    parser.add_argument(
        "--official-run",
        action="store_true",
        help="Require clean git repository before training run.",
    )
    parser.add_argument(
        "--lb-score",
        type=float,
        default=None,
        help="Optional leaderboard score to include at run time.",
    )
    parser.add_argument(
        "--fast-dev",
        action="store_true",
        help="Run a much faster development profile (fewer folds and lighter models).",
    )
    parser.add_argument(
        "--exclude-2023",
        action="store_true",
        help="Exclude most of 2023 from training, while keeping late-2023 tail by month threshold.",
    )
    parser.add_argument(
        "--exclude-2023-keep-from-month",
        type=int,
        default=10,
        help="When --exclude-2023 is set, keep 2023 rows from this month onward (1-12, default: 10).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for CatBoost models (HistGradientBoosting stays on CPU).",
    )
    parser.add_argument(
        "--gpu-devices",
        default="0",
        help="GPU device string for CatBoost, e.g. '0' or '0:1'.",
    )
    args = parser.parse_args()
    run_with_ledger(
        config_name=args.config_name,
        official_run=args.official_run,
        lb_score=args.lb_score,
        fast_dev=args.fast_dev,
        exclude_2023=args.exclude_2023,
        exclude_2023_keep_from_month=args.exclude_2023_keep_from_month,
        use_gpu=args.use_gpu,
        gpu_devices=args.gpu_devices,
    )


if __name__ == "__main__":
    main()
