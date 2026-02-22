from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from intraday_full_fallback_spike import (
    _apply_fold_stress_thresholds,
    _safe_float,
    build_feature_sets,
    fit_cat_regressor,
    fit_spike_classifier,
    fit_uplift_model,
    make_day_folds,
    prepare_data,
)

ORTHOGONAL_MODEL_PARAMS_DEFAULT: dict[str, dict[str, object]] = {
    "full_regressor": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3600,
        "learning_rate": 0.018,
        "depth": 8,
        "l2_leaf_reg": 26,
        "bagging_temperature": 0.6,
        "random_strength": 1.0,
        "verbose": 0,
    },
    "fallback_regressor": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 3400,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 24,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "verbose": 0,
    },
    "spike_classifier": {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 1400,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 20,
        "verbose": 0,
    },
    "uplift_regressor": {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 2600,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 30,
        "bagging_temperature": 0.9,
        "random_strength": 1.1,
        "verbose": 0,
    },
}


def load_orthogonal_model_params(params_in: str | None) -> dict[str, dict[str, object]]:
    params: dict[str, dict[str, object]] = {
        name: dict(values) for name, values in ORTHOGONAL_MODEL_PARAMS_DEFAULT.items()
    }
    if not params_in:
        return params

    path = Path(params_in)
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Hyperparameter JSON must be an object.")

    for section, defaults in ORTHOGONAL_MODEL_PARAMS_DEFAULT.items():
        override = payload.get(section, {})
        if not isinstance(override, dict):
            continue
        merged = dict(defaults)
        merged.update(override)
        params[section] = merged
    return params


def _build_tuning_defaults_from_params(
    model_params: dict[str, dict[str, object]],
    *,
    spike_power: float,
    uplift_clip_quantile: float,
) -> dict[str, float]:
    return {
        "full_iterations": int(model_params["full_regressor"].get("iterations", 3600)),
        "full_learning_rate": float(model_params["full_regressor"].get("learning_rate", 0.018)),
        "full_depth": int(model_params["full_regressor"].get("depth", 8)),
        "full_l2_leaf_reg": float(model_params["full_regressor"].get("l2_leaf_reg", 26)),
        "full_bagging_temperature": float(model_params["full_regressor"].get("bagging_temperature", 0.6)),
        "full_random_strength": float(model_params["full_regressor"].get("random_strength", 1.0)),
        "fallback_iterations": int(model_params["fallback_regressor"].get("iterations", 3400)),
        "fallback_learning_rate": float(model_params["fallback_regressor"].get("learning_rate", 0.02)),
        "fallback_depth": int(model_params["fallback_regressor"].get("depth", 8)),
        "fallback_l2_leaf_reg": float(model_params["fallback_regressor"].get("l2_leaf_reg", 24)),
        "fallback_bagging_temperature": float(model_params["fallback_regressor"].get("bagging_temperature", 0.5)),
        "fallback_random_strength": float(model_params["fallback_regressor"].get("random_strength", 1.0)),
        "spike_iterations": int(model_params["spike_classifier"].get("iterations", 1400)),
        "spike_learning_rate": float(model_params["spike_classifier"].get("learning_rate", 0.03)),
        "spike_depth": int(model_params["spike_classifier"].get("depth", 7)),
        "spike_l2_leaf_reg": float(model_params["spike_classifier"].get("l2_leaf_reg", 20)),
        "uplift_iterations": int(model_params["uplift_regressor"].get("iterations", 2600)),
        "uplift_learning_rate": float(model_params["uplift_regressor"].get("learning_rate", 0.02)),
        "uplift_depth": int(model_params["uplift_regressor"].get("depth", 8)),
        "uplift_l2_leaf_reg": float(model_params["uplift_regressor"].get("l2_leaf_reg", 30)),
        "uplift_bagging_temperature": float(model_params["uplift_regressor"].get("bagging_temperature", 0.9)),
        "uplift_random_strength": float(model_params["uplift_regressor"].get("random_strength", 1.1)),
        "spike_power": float(spike_power),
        "uplift_clip_quantile": float(uplift_clip_quantile),
    }


def _params_from_optuna_trial(
    trial: Any,
    base_params: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], float, float]:
    params: dict[str, dict[str, object]] = {k: dict(v) for k, v in base_params.items()}

    params["full_regressor"].update(
        {
            "iterations": trial.suggest_int("full_iterations", 1800, 5200, step=100),
            "learning_rate": trial.suggest_float("full_learning_rate", 0.008, 0.08, log=True),
            "depth": trial.suggest_int("full_depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("full_l2_leaf_reg", 6.0, 80.0, log=True),
            "bagging_temperature": trial.suggest_float("full_bagging_temperature", 0.0, 2.5),
            "random_strength": trial.suggest_float("full_random_strength", 0.2, 2.5),
            "verbose": 0,
        }
    )
    params["fallback_regressor"].update(
        {
            "iterations": trial.suggest_int("fallback_iterations", 1500, 5000, step=100),
            "learning_rate": trial.suggest_float("fallback_learning_rate", 0.008, 0.08, log=True),
            "depth": trial.suggest_int("fallback_depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("fallback_l2_leaf_reg", 6.0, 80.0, log=True),
            "bagging_temperature": trial.suggest_float("fallback_bagging_temperature", 0.0, 2.5),
            "random_strength": trial.suggest_float("fallback_random_strength", 0.2, 2.5),
            "verbose": 0,
        }
    )
    params["spike_classifier"].update(
        {
            "iterations": trial.suggest_int("spike_iterations", 600, 2600, step=100),
            "learning_rate": trial.suggest_float("spike_learning_rate", 0.008, 0.1, log=True),
            "depth": trial.suggest_int("spike_depth", 5, 10),
            "l2_leaf_reg": trial.suggest_float("spike_l2_leaf_reg", 4.0, 60.0, log=True),
            "verbose": 0,
        }
    )
    params["uplift_regressor"].update(
        {
            "iterations": trial.suggest_int("uplift_iterations", 1200, 4600, step=100),
            "learning_rate": trial.suggest_float("uplift_learning_rate", 0.008, 0.08, log=True),
            "depth": trial.suggest_int("uplift_depth", 6, 10),
            "l2_leaf_reg": trial.suggest_float("uplift_l2_leaf_reg", 6.0, 100.0, log=True),
            "bagging_temperature": trial.suggest_float("uplift_bagging_temperature", 0.0, 2.5),
            "random_strength": trial.suggest_float("uplift_random_strength", 0.2, 2.5),
            "verbose": 0,
        }
    )
    spike_power = float(trial.suggest_float("spike_power", 0.7, 2.2))
    uplift_clip_quantile = float(trial.suggest_float("uplift_clip_quantile", 0.95, 0.999))
    return params, spike_power, uplift_clip_quantile


def tune_orthogonal_hyperparameters(
    train_df: pd.DataFrame,
    *,
    all_features: list[str],
    fallback_features: list[str],
    cat_features: list[str],
    seed: int,
    base_model_params: dict[str, dict[str, object]],
    base_spike_power: float,
    base_uplift_clip_quantile: float,
    cv_folds: int,
    cv_val_days: int,
    cv_step_days: int,
    cv_min_train_days: int,
    n_trials: int,
    timeout_seconds: float,
) -> tuple[dict[str, dict[str, object]], float, float, dict[str, Any]]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for --tune-hparams. Install it (e.g. `uv add optuna`) and rerun."
        ) from exc

    if n_trials <= 0:
        raise ValueError("--tune-trials must be > 0 when tuning is enabled.")
    if timeout_seconds <= 0.0:
        raise ValueError("--tune-time-budget-minutes must be > 0 when tuning is enabled.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    defaults_for_enqueue = _build_tuning_defaults_from_params(
        base_model_params,
        spike_power=base_spike_power,
        uplift_clip_quantile=base_uplift_clip_quantile,
    )
    study.enqueue_trial(defaults_for_enqueue)

    trial_logs: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        trial_start = time.perf_counter()
        model_params, spike_power, uplift_clip_quantile = _params_from_optuna_trial(
            trial,
            base_model_params,
        )
        try:
            oof_df, cv_report = run_oof_cv_for_calibration(
                train_df,
                all_features=all_features,
                fallback_features=fallback_features,
                cat_features=cat_features,
                seed=seed + int(trial.number) * 1000,
                model_params=model_params,
                cv_folds=cv_folds,
                cv_val_days=cv_val_days,
                cv_step_days=cv_step_days,
                cv_min_train_days=cv_min_train_days,
                spike_power=spike_power,
                uplift_clip_quantile=uplift_clip_quantile,
            )
            del oof_df
            cv_rmse = cv_report.get("cv_rmse")
            coverage = float(cv_report.get("oof_coverage", 0.0))
        except Exception as exc:
            cv_rmse = float("inf")
            coverage = 0.0
            cv_report = {"oof_coverage": coverage}
            trial.set_user_attr("exception", str(exc))

        elapsed_s = time.perf_counter() - trial_start
        trial.set_user_attr("elapsed_seconds", float(elapsed_s))
        trial.set_user_attr("oof_coverage", coverage)
        trial_logs.append(
            {
                "trial": int(trial.number),
                "cv_rmse": float(cv_rmse),
                "elapsed_seconds": float(elapsed_s),
                "oof_coverage": coverage,
                "spike_power": float(spike_power),
                "uplift_clip_quantile": float(uplift_clip_quantile),
            }
        )
        print(
            f"[TUNE] trial={trial.number} rmse={cv_rmse:.6f} "
            f"elapsed={elapsed_s/60.0:.2f}m coverage={coverage:.4%}"
        )
        return float(cv_rmse)

    started = time.perf_counter()
    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=float(timeout_seconds),
        show_progress_bar=False,
        gc_after_trial=True,
    )
    tuning_elapsed = time.perf_counter() - started

    best_model_params, best_spike_power, best_uplift_clip_quantile = _params_from_optuna_trial(
        study.best_trial,
        base_model_params,
    )
    report = {
        "best_value": float(study.best_value),
        "best_trial_number": int(study.best_trial.number),
        "best_trial_params": dict(study.best_trial.params),
        "best_trial_user_attrs": dict(study.best_trial.user_attrs),
        "n_trials_completed": int(len(study.trials)),
        "elapsed_seconds": float(tuning_elapsed),
        "trials": trial_logs,
    }
    return best_model_params, best_spike_power, best_uplift_clip_quantile, report


def _parse_peak_hours(raw: str) -> list[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--peak-hours must contain at least one hour.")
    out: list[int] = []
    for v in vals:
        h = int(v)
        if h < 0 or h > 23:
            raise ValueError(f"Invalid hour in --peak-hours: {h}")
        out.append(h)
    return sorted(set(out))


def _market_hour_quantile_stats(
    train_df: pd.DataFrame,
    *,
    lower_q: float,
    upper_q: float,
) -> dict[str, Any]:
    work = train_df[["market", "delivery_start", "target"]].copy()
    work["delivery_start"] = pd.to_datetime(work["delivery_start"], errors="coerce")
    work["hour"] = work["delivery_start"].dt.hour
    work["market"] = work["market"].astype(str)
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    work = work.dropna(subset=["target", "hour"])
    if work.empty:
        raise ValueError("Training data has no valid rows for quantile guardrails.")

    by_mh = work.groupby(["market", "hour"], dropna=False)["target"]
    by_m = work.groupby("market", dropna=False)["target"]
    mh_low = {(str(k[0]), int(k[1])): float(v) for k, v in by_mh.quantile(lower_q).to_dict().items()}
    mh_high = {(str(k[0]), int(k[1])): float(v) for k, v in by_mh.quantile(upper_q).to_dict().items()}
    m_low = {str(k): float(v) for k, v in by_m.quantile(lower_q).to_dict().items()}
    m_high = {str(k): float(v) for k, v in by_m.quantile(upper_q).to_dict().items()}
    return {
        "market_hour_low": mh_low,
        "market_hour_high": mh_high,
        "market_low": m_low,
        "market_high": m_high,
        "global_low": float(work["target"].quantile(lower_q)),
        "global_high": float(work["target"].quantile(upper_q)),
    }


def _apply_quantile_guardrails(
    pred: np.ndarray,
    key_df: pd.DataFrame,
    stats: dict[str, Any],
) -> tuple[np.ndarray, int]:
    work = key_df[["market", "delivery_start"]].copy()
    work["market"] = work["market"].astype(str)
    work["hour"] = pd.to_datetime(work["delivery_start"], errors="coerce").dt.hour.fillna(-1).astype(int)

    keys = list(zip(work["market"], work["hour"]))
    low = pd.Series(keys).map(stats["market_hour_low"]).to_numpy(dtype=float)
    high = pd.Series(keys).map(stats["market_hour_high"]).to_numpy(dtype=float)

    m_low = work["market"].map(stats["market_low"]).to_numpy(dtype=float)
    m_high = work["market"].map(stats["market_high"]).to_numpy(dtype=float)
    low = np.where(np.isnan(low), m_low, low)
    high = np.where(np.isnan(high), m_high, high)
    low = np.where(np.isnan(low), float(stats["global_low"]), low)
    high = np.where(np.isnan(high), float(stats["global_high"]), high)

    swap = high < low
    if np.any(swap):
        tmp = low.copy()
        low[swap] = high[swap]
        high[swap] = tmp[swap]

    clipped = np.clip(pred, low, high)
    n_clipped = int(np.count_nonzero(np.abs(clipped - pred) > 1e-12))
    return clipped.astype(float), n_clipped


def _fit_peak_hour_bias_calibrator(
    oof_df: pd.DataFrame,
    *,
    peak_hours: list[int],
    min_rows: int,
) -> pd.DataFrame:
    if oof_df.empty:
        return pd.DataFrame(columns=["market", "hour", "bias", "rows"])
    tmp = oof_df.copy()
    tmp["delivery_start"] = pd.to_datetime(tmp["delivery_start"], errors="coerce")
    tmp["hour"] = tmp["delivery_start"].dt.hour
    tmp = tmp[tmp["hour"].isin(peak_hours)].copy()
    if tmp.empty:
        return pd.DataFrame(columns=["market", "hour", "bias", "rows"])

    tmp["resid"] = tmp["target"] - tmp["pred"]
    out = (
        tmp.groupby(["market", "hour"], dropna=False)["resid"]
        .agg(bias="mean", rows="size")
        .reset_index()
    )
    out = out[out["rows"] >= int(min_rows)].copy()
    out["market"] = out["market"].astype(str)
    out["hour"] = out["hour"].astype(int)
    out["bias"] = out["bias"].astype(float)
    out["rows"] = out["rows"].astype(int)
    return out.sort_values(["market", "hour"]).reset_index(drop=True)


def _apply_peak_hour_bias_calibrator(
    pred: np.ndarray,
    key_df: pd.DataFrame,
    calibrator_df: pd.DataFrame,
) -> tuple[np.ndarray, int]:
    if calibrator_df.empty:
        return pred, 0
    key = key_df[["market", "delivery_start"]].copy()
    key["market"] = key["market"].astype(str)
    key["hour"] = pd.to_datetime(key["delivery_start"], errors="coerce").dt.hour.fillna(-1).astype(int)
    lookup = {(str(r.market), int(r.hour)): float(r.bias) for r in calibrator_df.itertuples(index=False)}
    adj = pd.Series(list(zip(key["market"], key["hour"]))).map(lookup).fillna(0.0).to_numpy(dtype=float)
    out = pred + adj
    n = int(np.count_nonzero(np.abs(adj) > 1e-12))
    return out.astype(float), n


def _predict_pipeline(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    all_features: list[str],
    fallback_features: list[str],
    cat_features: list[str],
    seed: int,
    model_params: dict[str, dict[str, object]],
    spike_power: float,
    uplift_clip_quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    full_model = fit_cat_regressor(
        train_df[all_features],
        train_df["target"],
        cat_features,
        seed=seed,
        model_params=model_params["full_regressor"],
    )
    fallback_model = fit_cat_regressor(
        train_df[fallback_features],
        train_df["target"],
        cat_features,
        seed=seed + 1,
        model_params=model_params["fallback_regressor"],
    )
    spike_clf = fit_spike_classifier(
        train_df[fallback_features],
        train_df["target"],
        train_df["market"],
        cat_features,
        seed=seed + 2,
        model_params=model_params["spike_classifier"],
    )

    tr_full_pred = np.asarray(full_model.predict(train_df[all_features]), dtype=float)
    tr_fallback_pred = np.asarray(fallback_model.predict(train_df[fallback_features]), dtype=float)
    tr_missing = train_df["meteo_missing_any"].to_numpy(dtype=int) == 1
    tr_baseline = np.where(tr_missing, tr_fallback_pred, tr_full_pred)
    tr_p_spike = np.asarray(spike_clf.predict_proba(train_df[fallback_features])[:, 1], dtype=float)
    uplift_model = fit_uplift_model(
        train_df[fallback_features],
        train_df["target"],
        tr_baseline,
        tr_p_spike,
        cat_features,
        seed=seed + 3,
        model_params=model_params["uplift_regressor"],
    )
    tr_uplift = np.asarray(uplift_model.predict(train_df[fallback_features]), dtype=float)
    uplift_cap = float(np.quantile(np.abs(tr_uplift), uplift_clip_quantile))
    if not np.isfinite(uplift_cap) or uplift_cap <= 0.0:
        uplift_cap = float(np.quantile(np.abs(tr_uplift), 0.95)) if len(tr_uplift) else 1.0
    uplift_cap = max(uplift_cap, 1e-3)

    va_full_pred = np.asarray(full_model.predict(pred_df[all_features]), dtype=float)
    va_fallback_pred = np.asarray(fallback_model.predict(pred_df[fallback_features]), dtype=float)
    va_missing = pred_df["meteo_missing_any"].to_numpy(dtype=int) == 1
    va_baseline = np.where(va_missing, va_fallback_pred, va_full_pred)
    va_p_spike = np.asarray(spike_clf.predict_proba(pred_df[fallback_features])[:, 1], dtype=float)
    va_uplift = np.asarray(uplift_model.predict(pred_df[fallback_features]), dtype=float)
    va_uplift = np.clip(va_uplift, -uplift_cap, uplift_cap)
    va_gate = np.power(np.clip(va_p_spike, 0.0, 1.0), spike_power)
    pred = va_baseline + va_gate * va_uplift

    info = {
        "uplift_cap_abs": float(uplift_cap),
        "train_spike_prob_mean": float(np.mean(tr_p_spike)),
        "pred_spike_prob_mean": float(np.mean(va_p_spike)),
    }
    return pred.astype(float), info


def run_oof_cv_for_calibration(
    train_df: pd.DataFrame,
    *,
    all_features: list[str],
    fallback_features: list[str],
    cat_features: list[str],
    seed: int,
    model_params: dict[str, dict[str, object]],
    cv_folds: int,
    cv_val_days: int,
    cv_step_days: int,
    cv_min_train_days: int,
    spike_power: float,
    uplift_clip_quantile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = make_day_folds(
        df=train_df,
        n_folds=max(1, int(cv_folds)),
        val_days=max(1, int(cv_val_days)),
        step_days=max(1, int(cv_step_days)),
        min_train_days=max(30, int(cv_min_train_days)),
    )
    if not folds:
        return pd.DataFrame(), {"cv_rmse": None, "oof_coverage": 0.0, "folds": []}

    oof = np.full(len(train_df), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for i, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        tr = _apply_fold_stress_thresholds(tr, tr)
        va = _apply_fold_stress_thresholds(tr, va)
        pred, info = _predict_pipeline(
            tr,
            va,
            all_features=all_features,
            fallback_features=fallback_features,
            cat_features=cat_features,
            seed=seed + i * 100,
            model_params=model_params,
            spike_power=spike_power,
            uplift_clip_quantile=uplift_clip_quantile,
        )
        oof[va_idx] = pred
        fold_rmse = float(mean_squared_error(va["target"], pred) ** 0.5)
        fold_rows.append(
            {
                "fold": i,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "rmse": fold_rmse,
                "uplift_cap_abs": info["uplift_cap_abs"],
                "pred_spike_prob_mean": info["pred_spike_prob_mean"],
            }
        )
        print(f"[CV] fold={i}/{len(folds)} rmse={fold_rmse:.6f} uplift_cap={info['uplift_cap_abs']:.4f}")

    valid = np.isfinite(oof)
    cv_rmse = float(mean_squared_error(train_df.loc[valid, "target"], oof[valid]) ** 0.5) if np.any(valid) else None
    report = {
        "cv_rmse": cv_rmse,
        "oof_coverage": float(np.mean(valid)),
        "folds": fold_rows,
    }
    oof_df = train_df.loc[valid, ["id", "market", "delivery_start", "target"]].copy()
    oof_df["pred"] = oof[valid]
    return oof_df, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Orthogonal Full+Fallback+Spike+Uplift pipeline with OOF bias calibration and "
            "market-hour quantile guardrails."
        )
    )
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="orthogonal_spike_fallback_plus")
    parser.add_argument("--exclude-2023", action="store_true")
    parser.add_argument("--exclude-2023-keep-from-month", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--params-in",
        default=None,
        help="Optional JSON overrides for this script's own hyperparameter defaults.",
    )
    parser.add_argument(
        "--params-out",
        default=None,
        help="Optional path to write the effective merged hyperparameters JSON.",
    )
    parser.add_argument("--spike-power", type=float, default=1.2)
    parser.add_argument("--uplift-clip-quantile", type=float, default=0.99)
    parser.add_argument(
        "--tune-hparams",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Optuna hyperparameter tuning before final fit.",
    )
    parser.add_argument("--tune-trials", type=int, default=30)
    parser.add_argument("--tune-time-budget-minutes", type=float, default=90.0)
    parser.add_argument("--tune-cv-folds", type=int, default=None)
    parser.add_argument("--tune-cv-val-days", type=int, default=None)
    parser.add_argument("--tune-cv-step-days", type=int, default=None)
    parser.add_argument("--tune-cv-min-train-days", type=int, default=None)
    parser.add_argument("--cv-calibration", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-val-days", type=int, default=14)
    parser.add_argument("--cv-step-days", type=int, default=14)
    parser.add_argument("--cv-min-train-days", type=int, default=120)
    parser.add_argument("--peak-hours", default="17,18,19,20")
    parser.add_argument("--peak-bias-min-rows", type=int, default=12)
    parser.add_argument("--apply-market-hour-guardrails", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guardrail-lower-quantile", type=float, default=0.005)
    parser.add_argument("--guardrail-upper-quantile", type=float, default=0.995)
    args = parser.parse_args()

    if not (0.0 < args.uplift_clip_quantile < 1.0):
        raise ValueError("--uplift-clip-quantile must be in (0,1).")
    if not (0.0 < args.guardrail_lower_quantile < 1.0):
        raise ValueError("--guardrail-lower-quantile must be in (0,1).")
    if not (0.0 < args.guardrail_upper_quantile < 1.0):
        raise ValueError("--guardrail-upper-quantile must be in (0,1).")
    if args.guardrail_lower_quantile >= args.guardrail_upper_quantile:
        raise ValueError("--guardrail-lower-quantile must be < --guardrail-upper-quantile.")
    if args.spike_power <= 0.0:
        raise ValueError("--spike-power must be > 0.")
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

    peak_hours = _parse_peak_hours(args.peak_hours)
    model_params = load_orthogonal_model_params(args.params_in)
    spike_power = float(args.spike_power)
    uplift_clip_quantile = float(args.uplift_clip_quantile)
    started_at = datetime.now(timezone.utc)
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{args.name}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    split = prepare_data(
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        exclude_2023=bool(args.exclude_2023),
        keep_from_month=max(1, min(12, int(args.exclude_2023_keep_from_month))),
    )
    print(f"[RUN] loaded train_rows={len(split.train)} test_rows={len(split.test)}")

    all_features, fallback_features, cat_features = build_feature_sets(split.train, split.meteo_cols)
    num_features = [c for c in all_features if c not in cat_features]
    for df in [split.train, split.test]:
        for c in num_features:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in cat_features:
            df[c] = df[c].fillna("missing").astype(str)
    print(f"[RUN] features all={len(all_features)} fallback={len(fallback_features)} cat={len(cat_features)}")

    tuning_report: dict[str, Any] | None = None
    if args.tune_hparams:
        print(
            "[RUN] tuning hyperparameters "
            f"(trials={args.tune_trials}, time_budget_min={args.tune_time_budget_minutes}, "
            f"cv={tune_cv_folds}x{tune_cv_val_days}d)..."
        )
        model_params, spike_power, uplift_clip_quantile, tuning_report = tune_orthogonal_hyperparameters(
            split.train,
            all_features=all_features,
            fallback_features=fallback_features,
            cat_features=cat_features,
            seed=args.seed,
            base_model_params=model_params,
            base_spike_power=spike_power,
            base_uplift_clip_quantile=uplift_clip_quantile,
            cv_folds=tune_cv_folds,
            cv_val_days=tune_cv_val_days,
            cv_step_days=tune_cv_step_days,
            cv_min_train_days=tune_cv_min_train_days,
            n_trials=args.tune_trials,
            timeout_seconds=args.tune_time_budget_minutes * 60.0,
        )
        print(
            f"[RUN] tuning complete best_cv_rmse={tuning_report['best_value']:.6f} "
            f"best_trial={tuning_report['best_trial_number']}"
        )

    oof_df = pd.DataFrame()
    cv_report: dict[str, Any] = {"cv_rmse": None, "oof_coverage": 0.0, "folds": []}
    if args.cv_calibration:
        print("[RUN] running OOF CV for calibration...")
        oof_df, cv_report = run_oof_cv_for_calibration(
            split.train,
            all_features=all_features,
            fallback_features=fallback_features,
            cat_features=cat_features,
            seed=args.seed,
            model_params=model_params,
            cv_folds=args.cv_folds,
            cv_val_days=args.cv_val_days,
            cv_step_days=args.cv_step_days,
            cv_min_train_days=args.cv_min_train_days,
            spike_power=spike_power,
            uplift_clip_quantile=uplift_clip_quantile,
        )
        print(
            f"[RUN] cv complete rmse={cv_report['cv_rmse']} "
            f"coverage={cv_report['oof_coverage']:.4%}"
        )

    bias_calibrator = _fit_peak_hour_bias_calibrator(
        oof_df,
        peak_hours=peak_hours,
        min_rows=args.peak_bias_min_rows,
    )
    print(f"[RUN] peak-hour bias calibrator rows={len(bias_calibrator)}")

    print("[RUN] fitting final models + predicting test...")
    pred_test, final_info = _predict_pipeline(
        split.train,
        split.test,
        all_features=all_features,
        fallback_features=fallback_features,
        cat_features=cat_features,
        seed=args.seed,
        model_params=model_params,
        spike_power=spike_power,
        uplift_clip_quantile=uplift_clip_quantile,
    )

    pred_test, bias_adj_rows = _apply_peak_hour_bias_calibrator(
        pred_test,
        split.test[["market", "delivery_start"]],
        bias_calibrator,
    )

    guardrail_clipped_rows = 0
    if args.apply_market_hour_guardrails:
        q_stats = _market_hour_quantile_stats(
            split.train,
            lower_q=args.guardrail_lower_quantile,
            upper_q=args.guardrail_upper_quantile,
        )
        pred_test, guardrail_clipped_rows = _apply_quantile_guardrails(
            pred_test,
            split.test[["market", "delivery_start"]],
            q_stats,
        )

    sample = pd.read_csv(args.sample_submission)
    id_to_pred = pd.Series(pred_test, index=split.test["id"].to_numpy())
    submission = sample[["id"]].copy()
    submission["target"] = submission["id"].map(id_to_pred).astype(float)
    if submission["target"].isna().any():
        raise ValueError("Submission has NaN targets after mapping.")

    sub_path = run_dir / "submission.csv"
    submission.to_csv(sub_path, index=False)
    latest_path = Path("csv/submission_orthogonal_spike_fallback_plus.csv")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(latest_path, index=False)

    # In-sample fit diagnostic for sanity.
    pred_train, _ = _predict_pipeline(
        split.train,
        split.train,
        all_features=all_features,
        fallback_features=fallback_features,
        cat_features=cat_features,
        seed=args.seed,
        model_params=model_params,
        spike_power=spike_power,
        uplift_clip_quantile=uplift_clip_quantile,
    )
    train_rmse = float(mean_squared_error(split.train["target"], pred_train) ** 0.5)

    params = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "name": args.name,
        "seed": int(args.seed),
        "params_in": args.params_in,
        "params_out": args.params_out,
        "tune_hparams": bool(args.tune_hparams),
        "tune_trials": int(args.tune_trials),
        "tune_time_budget_minutes": float(args.tune_time_budget_minutes),
        "tune_cv_folds": int(tune_cv_folds),
        "tune_cv_val_days": int(tune_cv_val_days),
        "tune_cv_step_days": int(tune_cv_step_days),
        "tune_cv_min_train_days": int(tune_cv_min_train_days),
        "exclude_2023": bool(args.exclude_2023),
        "exclude_2023_keep_from_month": int(args.exclude_2023_keep_from_month),
        "cv_calibration": bool(args.cv_calibration),
        "cv_folds": int(args.cv_folds),
        "cv_val_days": int(args.cv_val_days),
        "cv_step_days": int(args.cv_step_days),
        "cv_min_train_days": int(args.cv_min_train_days),
        "spike_power": float(spike_power),
        "uplift_clip_quantile": float(uplift_clip_quantile),
        "peak_hours": peak_hours,
        "peak_bias_min_rows": int(args.peak_bias_min_rows),
        "apply_market_hour_guardrails": bool(args.apply_market_hour_guardrails),
        "guardrail_lower_quantile": float(args.guardrail_lower_quantile),
        "guardrail_upper_quantile": float(args.guardrail_upper_quantile),
        "num_train_rows": int(len(split.train)),
        "num_test_rows": int(len(split.test)),
        "num_all_features": int(len(all_features)),
        "num_fallback_features": int(len(fallback_features)),
        "model_params": model_params,
    }
    metrics = {
        "run_id": run_id,
        "train_rmse_in_sample": train_rmse,
        "cv_rmse": cv_report.get("cv_rmse"),
        "cv_oof_coverage": cv_report.get("oof_coverage"),
        "cv_folds": cv_report.get("folds"),
        "peak_bias_calibrator_rows": int(len(bias_calibrator)),
        "peak_bias_adjusted_rows_test": int(bias_adj_rows),
        "guardrail_clipped_rows_test": int(guardrail_clipped_rows),
        "uplift_cap_abs_final": float(final_info.get("uplift_cap_abs", 0.0)),
        "p_spike_mean_test": _safe_float(final_info.get("pred_spike_prob_mean")),
        "test_missing_meteo_rate": _safe_float(split.test["meteo_missing_any"].mean()),
        "tuning_best_cv_rmse": None if tuning_report is None else _safe_float(tuning_report.get("best_value")),
        "tuning_best_trial": None if tuning_report is None else int(tuning_report.get("best_trial_number", -1)),
    }

    (run_dir / "params.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "model_params.json").write_text(json.dumps(model_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if tuning_report is not None:
        (run_dir / "hparam_tuning.json").write_text(
            json.dumps(tuning_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if tuning_report.get("trials"):
            pd.DataFrame(tuning_report["trials"]).to_csv(run_dir / "hparam_tuning_trials.csv", index=False)
    if args.params_out:
        Path(args.params_out).write_text(json.dumps(model_params, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not oof_df.empty:
        oof_df.to_csv(run_dir / "cv_oof.csv", index=False)
    if not bias_calibrator.empty:
        bias_calibrator.to_csv(run_dir / "peak_bias_calibrator.csv", index=False)

    print(f"Run ID: {run_id}")
    print(f"Saved submission: {sub_path}")
    print(f"Saved latest copy: {latest_path}")
    print(f"In-sample train RMSE: {train_rmse:.6f}")
    print(f"CV RMSE: {cv_report.get('cv_rmse')}")
    if tuning_report is not None:
        print(
            f"Tuning best CV RMSE: {tuning_report['best_value']:.6f} "
            f"(trial {tuning_report['best_trial_number']})"
        )
    print(f"Peak bias calibrator rows: {len(bias_calibrator)}")
    print(f"Peak bias adjusted rows (test): {bias_adj_rows}")
    print(f"Guardrail clipped rows (test): {guardrail_clipped_rows}")
    print(f"Saved effective hyperparams: {run_dir / 'model_params.json'}")


if __name__ == "__main__":
    main()
