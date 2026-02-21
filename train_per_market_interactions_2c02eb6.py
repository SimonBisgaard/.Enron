from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error


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


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_global_model() -> CatBoostRegressor:
    return CatBoostRegressor(
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


def _make_local_model() -> CatBoostRegressor:
    return CatBoostRegressor(
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
) -> None:
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_path = Path(args.sample_submission)

    model_params = {
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
        },
    }

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

        model = _make_global_model()
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

        tr_feat, va_feat = build_feature_table(tr, va.drop(columns=["target"]).copy())

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
            use_residual_stacking=use_residual_stacking,
            residual_oof_folds=residual_oof_folds,
            residual_oof_val_days=residual_oof_val_days,
            residual_oof_step_days=residual_oof_step_days,
            residual_oof_min_train_days=residual_oof_min_train_days,
        )
        va_feat = va_feat.copy()
        va_feat["global_pred_feature"] = artifacts.global_model.predict(va_feat[feat_cols])

        pred = np.full(len(va_feat), np.nan, dtype=float)
        key = va_feat[["market"]].copy()
        for market, idx in key.groupby("market", dropna=False).groups.items():
            model = artifacts.local_models.get(str(market))
            if model is None:
                pred[idx] = va_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float)
            else:
                local_pred = model.predict(va_feat.loc[idx, artifacts.feature_cols])
                if artifacts.local_target_is_residual:
                    pred[idx] = va_feat.loc[idx, "global_pred_feature"].to_numpy(dtype=float) + local_pred
                else:
                    pred[idx] = local_pred
        if np.isnan(pred).any():
            raise ValueError(f"NaNs in CV predictions for fold {fold_idx}")

        fold_rmse = _rmse(va["target"].to_numpy(dtype=float), pred)
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
        oof.loc[va.index] = pred

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
    return overall, pd.DataFrame(fold_rows), cv_oof


def train_global_and_local_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    *,
    use_residual_stacking: bool,
    residual_oof_folds: int,
    residual_oof_val_days: int,
    residual_oof_step_days: int,
    residual_oof_min_train_days: int,
) -> TrainArtifacts:
    global_model = _make_global_model()
    global_model.fit(train_df[feature_cols], train_df["target"], cat_features=cat_cols)
    train_df = train_df.copy()
    train_df["global_pred_feature"] = global_model.predict(train_df[feature_cols])

    local_feature_cols = feature_cols + ["global_pred_feature"]
    if use_residual_stacking:
        oof_global = compute_global_oof_predictions(
            train_df,
            feature_cols,
            cat_cols,
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
        model = _make_local_model()
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
    parser.add_argument(
        "--train-start-oct-2023",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict training rows to delivery_start >= 2023-10-01 (default: disabled).",
    )
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-val-days", type=int, default=14)
    parser.add_argument("--cv-step-days", type=int, default=14)
    parser.add_argument("--cv-min-train-days", type=int, default=90)
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

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample = pd.read_csv(args.sample_submission)

    if args.exclude_2023:
        train_df = apply_exclude_2023(train_df, keep_from_month=args.exclude_2023_keep_from_month)
    if args.train_start_oct_2023:
        train_df = apply_train_start_cutoff(train_df, start_date="2023-10-01")

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
            use_residual_stacking=args.use_residual_stacking,
            residual_oof_folds=args.residual_oof_folds,
            residual_oof_val_days=args.residual_oof_val_days,
            residual_oof_step_days=args.residual_oof_step_days,
            residual_oof_min_train_days=args.residual_oof_min_train_days,
        )

    train_feat, test_feat = build_feature_table(train_df, test_df)
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
        use_residual_stacking=args.use_residual_stacking,
        residual_oof_folds=args.residual_oof_folds,
        residual_oof_val_days=args.residual_oof_val_days,
        residual_oof_step_days=args.residual_oof_step_days,
        residual_oof_min_train_days=args.residual_oof_min_train_days,
    )

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

    print(f"Saved submission: {sub_path}")
    print(f"Saved latest copy: {latest_path}")
    print(f"Features used: {len(artifacts.feature_cols)}")
    print(f"Categorical features: {artifacts.cat_cols}")
    print(f"Markets modeled: {sorted(artifacts.local_models.keys())}")
    print(f"Local target is residual: {artifacts.local_target_is_residual}")
    print(f"CV RMSE: {cv_rmse}")
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
        )

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


if __name__ == "__main__":
    main()
