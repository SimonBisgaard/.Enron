from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_error

FORECAST_COLS = ["load_forecast", "wind_forecast", "solar_forecast"]
FORECAST_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
FORECAST_DIFFS = [1, 2, 3, 6, 24]
FORECAST_ROLL_MEANS = [3, 6, 12, 24, 48, 168]
FORECAST_ROLL_STDS = [6, 24]
FORECAST_ROLL_MAXS = [3, 6, 24]
EPS = 1e-6


DEFAULT_MODEL_PARAMS: dict[str, dict[str, object]] = {
    "full_regressor": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3200,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 24,
        "bagging_temperature": 0.5,
        "random_strength": 0.9,
        "verbose": 0,
    },
    "fallback_regressor": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3200,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 24,
        "bagging_temperature": 0.5,
        "random_strength": 0.9,
        "verbose": 0,
    },
    "spike_classifier": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1200,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 18,
        "verbose": 0,
    },
    "uplift_regressor": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2200,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 28,
        "bagging_temperature": 0.8,
        "random_strength": 1.0,
        "verbose": 0,
    },
}


@dataclass
class SplitData:
    train: pd.DataFrame
    test: pd.DataFrame
    meteo_cols: list[str]


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return x


def load_model_params(params_in: str | None) -> dict[str, dict[str, object]]:
    params: dict[str, dict[str, object]] = {k: dict(v) for k, v in DEFAULT_MODEL_PARAMS.items()}
    if not params_in:
        return params

    path = Path(params_in)
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Hyperparameter JSON must be an object.")

    for section, defaults in DEFAULT_MODEL_PARAMS.items():
        override = payload.get(section, {})
        if isinstance(override, dict):
            merged = dict(defaults)
            merged.update(override)
            params[section] = merged
    return params


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    start = pd.to_datetime(out["delivery_start"], errors="coerce")
    end = pd.to_datetime(out["delivery_end"], errors="coerce")
    epoch = pd.Timestamp("1970-01-01")

    out["hour"] = start.dt.hour
    out["dow"] = start.dt.dayofweek
    out["month"] = start.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_peak_17_20"] = out["hour"].isin([17, 18, 19, 20]).astype(int)
    out["is_18_19"] = out["hour"].isin([18, 19]).astype(int)
    out["delivery_start_ts"] = ((start - epoch) // pd.Timedelta(seconds=1)).astype("int64")
    out["delivery_duration_h"] = (end - start).dt.total_seconds() / 3600.0
    return out


def add_missingness_features(df: pd.DataFrame, meteo_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["meteo_missing_count"] = out[meteo_cols].isna().sum(axis=1)
    out["meteo_missing_any"] = (out["meteo_missing_count"] > 0).astype(int)
    return out


def add_forecast_temporal_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    tr["__is_train"] = 1
    te["__is_train"] = 0
    all_df = pd.concat([tr, te], ignore_index=True, sort=False)
    all_df = all_df.sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)
    g = all_df.groupby("market", sort=False)

    for col in FORECAST_COLS:
        if col not in all_df.columns:
            continue
        s = all_df[col].astype(float)
        lag1 = g[col].shift(1)

        for lag in FORECAST_LAGS:
            all_df[f"{col}_lag_{lag}"] = g[col].shift(lag)

        for k in FORECAST_DIFFS:
            all_df[f"{col}_diff_{k}"] = s - g[col].shift(k)

        for w in FORECAST_ROLL_MEANS:
            all_df[f"{col}_roll_mean_{w}"] = (
                lag1.groupby(all_df["market"]).rolling(w, min_periods=3).mean().reset_index(level=0, drop=True)
            )
        for w in FORECAST_ROLL_STDS:
            all_df[f"{col}_roll_std_{w}"] = (
                lag1.groupby(all_df["market"]).rolling(w, min_periods=3).std().reset_index(level=0, drop=True)
            )
        for w in FORECAST_ROLL_MAXS:
            all_df[f"{col}_roll_max_{w}"] = (
                lag1.groupby(all_df["market"]).rolling(w, min_periods=3).max().reset_index(level=0, drop=True)
            )
        all_df[f"{col}_roll_min_24"] = (
            lag1.groupby(all_df["market"]).rolling(24, min_periods=3).min().reset_index(level=0, drop=True)
        )

    all_df["is_night"] = (all_df["solar_forecast"].fillna(0.0) <= 0.0).astype(int)

    out_train = all_df.loc[all_df["__is_train"] == 1].drop(columns="__is_train").copy()
    out_test = all_df.loc[all_df["__is_train"] == 0].drop(columns="__is_train").copy()
    out_train = out_train.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
    out_test = out_test.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
    return out_train, out_test


def add_stress_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    for df in [tr, te]:
        df["residual_load"] = df["load_forecast"].astype(float) - df["wind_forecast"].astype(float) - df["solar_forecast"].astype(float)
        df["residual_load_1819"] = df["residual_load"] * df["is_18_19"]
        df["residual_load_night"] = df["residual_load"] * df["is_night"]
        df["stress_ratio"] = df["load_forecast"].astype(float) / (df["wind_forecast"].astype(float) + 1000.0)
        df["stress_ratio_1819"] = df["stress_ratio"] * df["is_18_19"]
        df["stress_ratio_night"] = df["stress_ratio"] * df["is_night"]
        df["stress_ratio_sq"] = df["stress_ratio"] ** 2

    q_df = tr.loc[tr["is_18_19"] == 1]
    load_q90 = q_df.groupby("market")["load_forecast"].quantile(0.9).to_dict()
    wind_q10 = q_df.groupby("market")["wind_forecast"].quantile(0.1).to_dict()

    for df in [tr, te]:
        l90 = df["market"].map(load_q90).astype(float)
        w10 = df["market"].map(wind_q10).astype(float)
        df["excess_load_90"] = np.clip(df["load_forecast"].astype(float) - l90, a_min=0.0, a_max=None)
        df["wind_deficit_10"] = np.clip(w10 - df["wind_forecast"].astype(float), a_min=0.0, a_max=None)
        df["stress_excess"] = df["excess_load_90"] * df["wind_deficit_10"]
        df["stress_excess_1819"] = df["stress_excess"] * df["is_18_19"]

        if "load_forecast_lag_1" in df.columns:
            df["load_diff_1"] = df["load_forecast"].astype(float) - df["load_forecast_lag_1"].astype(float)
        else:
            df["load_diff_1"] = np.nan
        df["neg_load_ramp_1"] = np.clip(-df["load_diff_1"], a_min=0.0, a_max=None)
        df["stress_ramp_1819"] = df["is_18_19"] * df["wind_deficit_10"] * df["neg_load_ramp_1"]
    return tr, te


def _apply_fold_stress_thresholds(train_part: pd.DataFrame, target_part: pd.DataFrame) -> pd.DataFrame:
    out = target_part.copy()
    q_df = train_part.loc[train_part["is_18_19"] == 1]
    if q_df.empty:
        return out

    load_q90 = q_df.groupby("market")["load_forecast"].quantile(0.9).to_dict()
    wind_q10 = q_df.groupby("market")["wind_forecast"].quantile(0.1).to_dict()
    l90 = out["market"].map(load_q90).astype(float)
    w10 = out["market"].map(wind_q10).astype(float)
    out["excess_load_90"] = np.clip(out["load_forecast"].astype(float) - l90, a_min=0.0, a_max=None)
    out["wind_deficit_10"] = np.clip(w10 - out["wind_forecast"].astype(float), a_min=0.0, a_max=None)
    out["stress_excess"] = out["excess_load_90"] * out["wind_deficit_10"]
    out["stress_excess_1819"] = out["stress_excess"] * out["is_18_19"]
    if "load_forecast_lag_1" in out.columns:
        out["load_diff_1"] = out["load_forecast"].astype(float) - out["load_forecast_lag_1"].astype(float)
    out["neg_load_ramp_1"] = np.clip(-out["load_diff_1"], a_min=0.0, a_max=None)
    out["stress_ramp_1819"] = out["is_18_19"] * out["wind_deficit_10"] * out["neg_load_ramp_1"]
    return out


def make_day_folds(
    df: pd.DataFrame,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    day_series = pd.to_datetime(df["delivery_start_ts"], unit="s", errors="coerce").dt.floor("D")
    unique_days = np.array(sorted(day_series.dropna().unique()))
    if len(unique_days) < (val_days + min_train_days):
        return []

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        shift = (n_folds - 1 - k) * step_days
        val_end = len(unique_days) - shift
        val_start = val_end - val_days
        train_end = val_start
        if val_start <= 0 or train_end < min_train_days:
            continue

        train_days = unique_days[:train_end]
        valid_days = unique_days[val_start:val_end]

        train_idx = df.index[day_series.isin(train_days)].to_numpy()
        valid_idx = df.index[day_series.isin(valid_days)].to_numpy()
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue
        folds.append((train_idx, valid_idx))
    return folds


def run_cv(
    train_df: pd.DataFrame,
    all_features: list[str],
    fallback_features: list[str],
    cat_features: list[str],
    seed: int,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
    model_params: dict[str, dict[str, object]],
) -> dict[str, object]:
    folds = make_day_folds(
        df=train_df,
        n_folds=n_folds,
        val_days=val_days,
        step_days=step_days,
        min_train_days=min_train_days,
    )
    if not folds:
        return {
            "cv_rmse": None,
            "folds": [],
            "per_market_rmse": {},
            "rmse_18_19": None,
            "oof_coverage": 0.0,
        }

    y = train_df["target"].to_numpy(dtype=float)
    oof = np.full(len(train_df), np.nan, dtype=float)
    fold_rows: list[dict[str, object]] = []

    for i, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"[CV] Fold {i}/{len(folds)}: train_rows={len(tr_idx)} valid_rows={len(va_idx)}")
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        tr = _apply_fold_stress_thresholds(tr, tr)
        va = _apply_fold_stress_thresholds(tr, va)

        print(f"[CV] Fold {i}: fit full model")
        full_model = fit_cat_regressor(
            tr[all_features], tr["target"], cat_features, seed=seed + 100 * i, model_params=model_params["full_regressor"]
        )
        print(f"[CV] Fold {i}: fit fallback model")
        fallback_model = fit_cat_regressor(
            tr[fallback_features],
            tr["target"],
            cat_features,
            seed=seed + 100 * i + 1,
            model_params=model_params["fallback_regressor"],
        )
        print(f"[CV] Fold {i}: fit spike classifier")
        spike_clf = fit_spike_classifier(
            tr[fallback_features],
            tr["target"],
            tr["market"],
            cat_features,
            seed=seed + 100 * i + 2,
            model_params=model_params["spike_classifier"],
        )

        tr_full_pred = np.asarray(full_model.predict(tr[all_features]), dtype=float)
        tr_fallback_pred = np.asarray(fallback_model.predict(tr[fallback_features]), dtype=float)
        tr_missing = tr["meteo_missing_any"].to_numpy(dtype=int) == 1
        tr_baseline = np.where(tr_missing, tr_fallback_pred, tr_full_pred)
        tr_p_spike = np.asarray(spike_clf.predict_proba(tr[fallback_features])[:, 1], dtype=float)
        print(f"[CV] Fold {i}: fit uplift model")
        uplift_model = fit_uplift_model(
            tr[fallback_features],
            tr["target"],
            tr_baseline,
            tr_p_spike,
            cat_features,
            seed=seed + 100 * i + 3,
            model_params=model_params["uplift_regressor"],
        )

        va_full_pred = np.asarray(full_model.predict(va[all_features]), dtype=float)
        va_fallback_pred = np.asarray(fallback_model.predict(va[fallback_features]), dtype=float)
        va_missing = va["meteo_missing_any"].to_numpy(dtype=int) == 1
        va_baseline = np.where(va_missing, va_fallback_pred, va_full_pred)
        va_p_spike = np.asarray(spike_clf.predict_proba(va[fallback_features])[:, 1], dtype=float)
        va_uplift = np.asarray(uplift_model.predict(va[fallback_features]), dtype=float)
        va_pred = va_baseline + va_p_spike * va_uplift
        oof[va_idx] = va_pred

        fold_rmse = float(mean_squared_error(va["target"], va_pred) ** 0.5)
        fold_1819_mask = va["is_18_19"].to_numpy(dtype=int) == 1
        fold_rmse_1819 = (
            float(mean_squared_error(va.loc[fold_1819_mask, "target"], va_pred[fold_1819_mask]) ** 0.5)
            if np.any(fold_1819_mask)
            else None
        )
        fold_rows.append(
            {
                "fold": i,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "rmse": fold_rmse,
                "rmse_18_19": fold_rmse_1819,
            }
        )
        print(f"[CV] Fold {i}: rmse={fold_rmse:.6f} rmse_18_19={fold_rmse_1819}")

    valid_mask = np.isfinite(oof)
    cv_rmse = float(mean_squared_error(y[valid_mask], oof[valid_mask]) ** 0.5) if np.any(valid_mask) else None

    per_market: dict[str, float] = {}
    if np.any(valid_mask):
        tmp = train_df.loc[valid_mask, ["market"]].copy()
        tmp["target"] = y[valid_mask]
        tmp["pred"] = oof[valid_mask]
        for mkt, g in tmp.groupby("market"):
            per_market[str(mkt)] = float(mean_squared_error(g["target"], g["pred"]) ** 0.5)

    mask_1819 = valid_mask & (train_df["is_18_19"].to_numpy(dtype=int) == 1)
    rmse_1819 = float(mean_squared_error(y[mask_1819], oof[mask_1819]) ** 0.5) if np.any(mask_1819) else None

    return {
        "cv_rmse": cv_rmse,
        "folds": fold_rows,
        "per_market_rmse": per_market,
        "rmse_18_19": rmse_1819,
        "oof_coverage": float(np.mean(valid_mask)),
    }


def prepare_data(train_path: Path, test_path: Path, exclude_2023: bool, keep_from_month: int) -> SplitData:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if exclude_2023:
        start = pd.to_datetime(train_df["delivery_start"], errors="coerce")
        keep = (start.dt.year != 2023) | ((start.dt.year == 2023) & (start.dt.month >= keep_from_month))
        train_df = train_df.loc[keep].reset_index(drop=True)

    core_non_meteo = {
        "id",
        "target",
        "market",
        "delivery_start",
        "delivery_end",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
    }
    meteo_cols = [c for c in train_df.columns if c not in core_non_meteo]

    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)
    train_df = add_missingness_features(train_df, meteo_cols)
    test_df = add_missingness_features(test_df, meteo_cols)
    train_df, test_df = add_forecast_temporal_features(train_df, test_df)
    train_df, test_df = add_stress_features(train_df, test_df)

    return SplitData(train=train_df, test=test_df, meteo_cols=meteo_cols)


def build_feature_sets(df: pd.DataFrame, meteo_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    exclude = {"id", "target", "delivery_start", "delivery_end"}
    all_features = [c for c in df.columns if c not in exclude]

    fallback_blocked_prefixes = tuple(
        [
            "global_horizontal_irradiance",
            "diffuse_horizontal_irradiance",
            "direct_normal_irradiance",
            "cloud_cover_",
            "precipitation_amount",
            "visibility",
            "air_temperature_2m",
            "apparent_temperature_2m",
            "dew_point_temperature_2m",
            "wet_bulb_temperature_2m",
            "surface_pressure",
            "freezing_level_height",
            "relative_humidity_2m",
            "convective_available_potential_energy",
            "lifted_index",
            "convective_inhibition",
            "wind_speed_80m",
            "wind_direction_80m",
            "wind_gust_speed_10m",
            "wind_speed_10m",
        ]
    )

    fallback_features: list[str] = []
    for c in all_features:
        if c in meteo_cols:
            continue
        if any(c.startswith(p) for p in fallback_blocked_prefixes):
            continue
        fallback_features.append(c)

    cat_features = ["market"]
    return all_features, fallback_features, cat_features


def fit_cat_regressor(
    x: pd.DataFrame,
    y: pd.Series,
    cat_features: list[str],
    seed: int,
    model_params: dict[str, object],
) -> CatBoostRegressor:
    params = dict(model_params)
    params["random_seed"] = seed
    model = CatBoostRegressor(**params)
    model.fit(x, y, cat_features=cat_features)
    return model


def fit_spike_classifier(
    x: pd.DataFrame,
    y: pd.Series,
    market: pd.Series,
    cat_features: list[str],
    seed: int,
    model_params: dict[str, object],
) -> CatBoostClassifier:
    q = y.groupby(market).quantile(0.99).to_dict()
    spike = (y.to_numpy(dtype=float) >= market.map(q).to_numpy(dtype=float)).astype(int)
    pos = float(spike.sum())
    neg = float(len(spike) - pos)
    scale_pos_weight = max(1.0, neg / max(pos, 1.0))

    params = dict(model_params)
    params["scale_pos_weight"] = scale_pos_weight
    params["random_seed"] = seed
    clf = CatBoostClassifier(**params)
    clf.fit(x, spike, cat_features=cat_features)
    return clf


def fit_uplift_model(
    x: pd.DataFrame,
    target: pd.Series,
    baseline_pred: np.ndarray,
    p_spike: np.ndarray,
    cat_features: list[str],
    seed: int,
    model_params: dict[str, object],
) -> CatBoostRegressor:
    residual = target.to_numpy(dtype=float) - baseline_pred
    abs_res = np.abs(residual)
    thr = float(np.quantile(abs_res, 0.95))
    focus_mask = (abs_res >= thr) | (p_spike >= np.quantile(p_spike, 0.90))
    if focus_mask.sum() < 300:
        focus_mask = abs_res >= np.quantile(abs_res, 0.90)

    params = dict(model_params)
    params["random_seed"] = seed
    uplift = CatBoostRegressor(**params)
    uplift.fit(x.loc[focus_mask], residual[focus_mask], cat_features=cat_features)
    return uplift


def main() -> None:
    parser = argparse.ArgumentParser(description="Full + Fallback + Spike pipeline for intraday RMSE forecasting.")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--config-name", default="intraday_full_fallback_spike_v1")
    parser.add_argument("--exclude-2023", action="store_true")
    parser.add_argument("--exclude-2023-keep-from-month", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--params-in", default=None, help="Path to JSON file with model hyperparameters.")
    parser.add_argument("--params-out", default=None, help="Optional path to write effective hyperparameters JSON.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-val-days", type=int, default=14)
    parser.add_argument("--cv-step-days", type=int, default=14)
    parser.add_argument("--cv-min-train-days", type=int, default=120)
    args = parser.parse_args()
    model_params = load_model_params(args.params_in)

    started_at = datetime.now(timezone.utc)
    run_id = f"{started_at.strftime('%Y%m%d-%H%M%S')}_{args.config_name}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    split = prepare_data(
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        exclude_2023=bool(args.exclude_2023),
        keep_from_month=max(1, min(12, int(args.exclude_2023_keep_from_month))),
    )
    print(f"[RUN] Loaded data: train_rows={len(split.train)} test_rows={len(split.test)}")

    all_features, fallback_features, cat_features = build_feature_sets(split.train, split.meteo_cols)
    num_features = [c for c in all_features if c not in cat_features]

    for df in [split.train, split.test]:
        for c in num_features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in cat_features:
            df[c] = df[c].fillna("missing").astype(str)
    print(
        f"[RUN] Feature sets: all={len(all_features)} fallback={len(fallback_features)} "
        f"categorical={len(cat_features)}"
    )

    print("[RUN] Starting CV...")
    cv_report = run_cv(
        train_df=split.train,
        all_features=all_features,
        fallback_features=fallback_features,
        cat_features=cat_features,
        seed=args.seed,
        n_folds=max(1, int(args.cv_folds)),
        val_days=max(1, int(args.cv_val_days)),
        step_days=max(1, int(args.cv_step_days)),
        min_train_days=max(30, int(args.cv_min_train_days)),
        model_params=model_params,
    )
    print(
        f"[RUN] CV complete: cv_rmse={cv_report.get('cv_rmse')} "
        f"cv_rmse_18_19={cv_report.get('rmse_18_19')} "
        f"oof_coverage={cv_report.get('oof_coverage'):.4%}"
    )

    print("[RUN] Fitting final full model")
    full_model = fit_cat_regressor(
        split.train[all_features],
        split.train["target"],
        cat_features,
        seed=args.seed,
        model_params=model_params["full_regressor"],
    )
    print("[RUN] Fitting final fallback model")
    fallback_model = fit_cat_regressor(
        split.train[fallback_features],
        split.train["target"],
        cat_features,
        seed=args.seed + 1,
        model_params=model_params["fallback_regressor"],
    )
    print("[RUN] Fitting final spike classifier")
    spike_clf = fit_spike_classifier(
        split.train[fallback_features],
        split.train["target"],
        split.train["market"],
        cat_features,
        seed=args.seed + 2,
        model_params=model_params["spike_classifier"],
    )

    train_full_pred = np.asarray(full_model.predict(split.train[all_features]), dtype=float)
    train_fallback_pred = np.asarray(fallback_model.predict(split.train[fallback_features]), dtype=float)
    train_missing = split.train["meteo_missing_any"].to_numpy(dtype=int) == 1
    train_baseline = np.where(train_missing, train_fallback_pred, train_full_pred)
    train_p_spike = np.asarray(spike_clf.predict_proba(split.train[fallback_features])[:, 1], dtype=float)
    print("[RUN] Fitting final uplift model")
    uplift_model = fit_uplift_model(
        split.train[fallback_features],
        split.train["target"],
        train_baseline,
        train_p_spike,
        cat_features,
        seed=args.seed + 3,
        model_params=model_params["uplift_regressor"],
    )

    test_full_pred = np.asarray(full_model.predict(split.test[all_features]), dtype=float)
    test_fallback_pred = np.asarray(fallback_model.predict(split.test[fallback_features]), dtype=float)
    test_missing = split.test["meteo_missing_any"].to_numpy(dtype=int) == 1
    baseline = np.where(test_missing, test_fallback_pred, test_full_pred)
    p_spike = np.asarray(spike_clf.predict_proba(split.test[fallback_features])[:, 1], dtype=float)
    uplift = np.asarray(uplift_model.predict(split.test[fallback_features]), dtype=float)
    pred_final = baseline + p_spike * uplift

    print("[RUN] Predicting test set and building submission")
    sample = pd.read_csv(args.sample_submission)
    id_to_pred = pd.Series(pred_final, index=split.test["id"].to_numpy())
    submission = sample[["id"]].copy()
    submission["target"] = submission["id"].map(id_to_pred).astype(float)
    if submission["target"].isna().any():
        raise ValueError("Missing predictions in submission output.")
    submission_path = run_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    train_uplift = np.asarray(uplift_model.predict(split.train[fallback_features]), dtype=float)
    train_final = train_baseline + train_p_spike * train_uplift
    train_rmse = float(mean_squared_error(split.train["target"], train_final) ** 0.5)

    params = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "config_name": args.config_name,
        "seed": args.seed,
        "params_in": args.params_in,
        "params_out": args.params_out,
        "exclude_2023": bool(args.exclude_2023),
        "exclude_2023_keep_from_month": int(args.exclude_2023_keep_from_month),
        "num_train_rows": int(len(split.train)),
        "num_test_rows": int(len(split.test)),
        "num_all_features": int(len(all_features)),
        "num_fallback_features": int(len(fallback_features)),
        "model_params": model_params,
        "cv_folds": int(args.cv_folds),
        "cv_val_days": int(args.cv_val_days),
        "cv_step_days": int(args.cv_step_days),
        "cv_min_train_days": int(args.cv_min_train_days),
    }
    metrics = {
        "run_id": run_id,
        "train_rmse_in_sample": train_rmse,
        "cv_rmse": cv_report.get("cv_rmse"),
        "cv_rmse_18_19": cv_report.get("rmse_18_19"),
        "cv_oof_coverage": cv_report.get("oof_coverage"),
        "cv_folds": cv_report.get("folds"),
        "cv_per_market_rmse": cv_report.get("per_market_rmse"),
        "test_missing_meteo_rate": _safe_float(split.test["meteo_missing_any"].mean()),
        "p_spike_mean_test": _safe_float(np.mean(p_spike)),
    }

    (run_dir / "params.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "model_params.json").write_text(json.dumps(model_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.params_out:
        Path(args.params_out).write_text(json.dumps(model_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    latest_submission = Path("submission_intraday_pipeline.csv")
    submission.to_csv(latest_submission, index=False)

    print(f"Run ID: {run_id}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved latest copy: {latest_submission}")
    print(f"In-sample train RMSE: {train_rmse:.6f}")
    print(f"CV RMSE: {cv_report.get('cv_rmse')}")
    print(f"CV RMSE (18-19): {cv_report.get('rmse_18_19')}")
    print(f"CV OOF coverage: {cv_report.get('oof_coverage'):.4%}")
    print(f"Test meteo-missing rate: {split.test['meteo_missing_any'].mean():.4%}")


if __name__ == "__main__":
    main()
