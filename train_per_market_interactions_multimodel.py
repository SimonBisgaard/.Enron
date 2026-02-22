from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_per_market_interactions_2c02eb6 import (
    apply_exclude_2023,
    apply_permutation_pruned_feature_policy,
    apply_train_start_cutoff,
    build_feature_table,
    make_time_series_folds,
    maybe_drop_redundant_features,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _format_duration(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _parse_model_families(raw: str) -> list[str]:
    allowed = {"catboost", "lightgbm", "xgboost", "ridge", "lasso", "rf"}
    families = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not families:
        raise ValueError("No model families specified.")
    unknown = [x for x in families if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown model families: {unknown}. Allowed: {sorted(allowed)}")
    return families


def _resolve_optional_families(families: list[str], allow_missing: bool) -> list[str]:
    resolved: list[str] = []
    for family in families:
        if family == "lightgbm":
            try:
                import lightgbm  # noqa: F401
            except ModuleNotFoundError as e:
                if allow_missing:
                    print(f"Skipping lightgbm (not installed): {e}")
                    continue
                raise
        if family == "xgboost":
            try:
                import xgboost  # noqa: F401
            except ModuleNotFoundError as e:
                if allow_missing:
                    print(f"Skipping xgboost (not installed): {e}")
                    continue
                raise
        resolved.append(family)

    if not resolved:
        raise ValueError("No usable model families after dependency checks.")
    return resolved


@dataclass
class FamilyBundle:
    family: str
    global_model: Any
    local_models: dict[str, Any]
    feature_cols: list[str]
    local_feature_cols: list[str]
    cat_cols: list[str]


def _cat_indices(feature_cols: list[str], cat_cols: list[str]) -> list[int]:
    cset = set(cat_cols)
    return [i for i, c in enumerate(feature_cols) if c in cset]


def _build_preprocessor(feature_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if c not in set(cat_cols)]
    present_cat = [c for c in feature_cols if c in set(cat_cols)]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if present_cat:
        transformers.append(("cat", cat_pipe, present_cat))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_model(
    family: str,
    *,
    feature_cols: list[str],
    cat_cols: list[str],
    random_seed: int,
) -> Any:
    if family == "catboost":
        return CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=2500,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=18.0,
            bagging_temperature=0.5,
            random_strength=1.0,
            random_seed=random_seed,
            verbose=0,
        )

    pre = _build_preprocessor(feature_cols, cat_cols)

    if family == "lightgbm":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=random_seed,
            n_jobs=-1,
        )
        return Pipeline(steps=[("pre", pre), ("model", model)])

    if family == "xgboost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=random_seed,
            n_jobs=-1,
            tree_method="hist",
        )
        return Pipeline(steps=[("pre", pre), ("model", model)])

    if family == "ridge":
        model = Ridge(alpha=1.0, random_state=random_seed)
        return Pipeline(steps=[("pre", pre), ("scale", StandardScaler(with_mean=False)), ("model", model)])

    if family == "lasso":
        model = Lasso(alpha=1e-4, max_iter=10000, random_state=random_seed)
        return Pipeline(steps=[("pre", pre), ("scale", StandardScaler(with_mean=False)), ("model", model)])

    if family == "rf":
        model = RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_seed,
            n_jobs=-1,
        )
        return Pipeline(steps=[("pre", pre), ("model", model)])

    raise ValueError(f"Unsupported family: {family}")


def _fit_model(model: Any, family: str, X: pd.DataFrame, y: np.ndarray, feature_cols: list[str], cat_cols: list[str]) -> None:
    if family == "catboost":
        model.fit(X[feature_cols], y, cat_features=_cat_indices(feature_cols, cat_cols))
        return
    model.fit(X[feature_cols], y)


def _predict_model(model: Any, family: str, X: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    pred = model.predict(X[feature_cols])
    return np.asarray(pred, dtype=float)


def _safe_name(raw: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(raw)).strip("_")


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


def _normalize_family_weights(weight_map: dict[str, float] | None, families: list[str]) -> dict[str, float]:
    if not families:
        raise ValueError("No model families available for weighting.")
    if weight_map is None:
        eq = 1.0 / float(len(families))
        return {f: eq for f in families}

    vals = np.array([float(weight_map.get(f, 0.0)) for f in families], dtype=float)
    if (vals < 0).any():
        raise ValueError("Ensemble weights must be non-negative.")
    s = float(vals.sum())
    if s <= 0.0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    vals = vals / s
    return {f: float(v) for f, v in zip(families, vals)}


def _parse_explicit_ensemble_weights(raw: str, families: list[str]) -> dict[str, float]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if not parts:
        raise ValueError("Empty --ensemble-weights value.")

    out: dict[str, float] = {}
    allowed = set(families)
    for p in parts:
        if "=" not in p:
            raise ValueError(
                "Explicit --ensemble-weights must use family=value pairs, "
                f"got '{p}'."
            )
        fam, val = p.split("=", 1)
        fam = fam.strip().lower()
        if fam not in allowed:
            raise ValueError(f"Unknown family in --ensemble-weights: {fam}. Allowed: {sorted(allowed)}")
        out[fam] = float(val.strip())
    return out


def _compute_market_hour_quantile_bounds(
    train_df_raw: pd.DataFrame,
    *,
    lower_q: float,
    upper_q: float,
) -> dict[str, Any]:
    work = train_df_raw[["market", "delivery_start", "target"]].copy()
    start = pd.to_datetime(work["delivery_start"], errors="coerce")
    work["hour"] = start.dt.hour
    work["market"] = work["market"].astype(str)
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    work = work.dropna(subset=["target", "hour"])
    if work.empty:
        return {
            "market_hour_low": {},
            "market_hour_high": {},
            "market_low": {},
            "market_high": {},
            "global_low": float("-inf"),
            "global_high": float("inf"),
        }

    mh_q = work.groupby(["market", "hour"], dropna=False)["target"].quantile([lower_q, upper_q]).unstack()
    m_q = work.groupby("market", dropna=False)["target"].quantile([lower_q, upper_q]).unstack()

    mh_low = {(str(m), int(h)): float(v) for (m, h), v in mh_q[lower_q].items()}
    mh_high = {(str(m), int(h)): float(v) for (m, h), v in mh_q[upper_q].items()}
    m_low = {str(m): float(v) for m, v in m_q[lower_q].items()}
    m_high = {str(m): float(v) for m, v in m_q[upper_q].items()}
    return {
        "market_hour_low": mh_low,
        "market_hour_high": mh_high,
        "market_low": m_low,
        "market_high": m_high,
        "global_low": float(work["target"].quantile(lower_q)),
        "global_high": float(work["target"].quantile(upper_q)),
    }


def _apply_market_hour_quantile_guardrails(
    pred: np.ndarray,
    key_df: pd.DataFrame,
    bounds: dict[str, Any],
) -> tuple[np.ndarray, int]:
    work = key_df[["market", "delivery_start"]].copy()
    work["market"] = work["market"].astype(str)
    work["hour"] = pd.to_datetime(work["delivery_start"], errors="coerce").dt.hour.fillna(-1).astype(int)

    mh_keys = list(zip(work["market"], work["hour"]))
    low = pd.Series(mh_keys).map(bounds["market_hour_low"]).to_numpy(dtype=float)
    high = pd.Series(mh_keys).map(bounds["market_hour_high"]).to_numpy(dtype=float)

    m_low = work["market"].map(bounds["market_low"]).to_numpy(dtype=float)
    m_high = work["market"].map(bounds["market_high"]).to_numpy(dtype=float)

    low = np.where(np.isnan(low), m_low, low)
    high = np.where(np.isnan(high), m_high, high)
    low = np.where(np.isnan(low), float(bounds["global_low"]), low)
    high = np.where(np.isnan(high), float(bounds["global_high"]), high)

    swap = high < low
    if np.any(swap):
        low2 = low.copy()
        low[swap] = high[swap]
        high[swap] = low2[swap]

    clipped = np.clip(pred, low, high)
    n_clipped = int(np.count_nonzero(np.abs(clipped - pred) > 1e-12))
    return clipped.astype(float), n_clipped


def _build_pair_corr_lookup(
    corr_source_df: pd.DataFrame,
    markets: list[str],
    *,
    lookback_hours: int,
    min_periods: int,
) -> tuple[dict[tuple[str, str], pd.Series], dict[tuple[str, str], float]]:
    src = corr_source_df[["delivery_start", "market", "target"]].copy()
    src["delivery_start"] = pd.to_datetime(src["delivery_start"], errors="coerce")
    src["market"] = src["market"].astype(str)
    src["target"] = pd.to_numeric(src["target"], errors="coerce")
    src = src.dropna(subset=["delivery_start", "market", "target"])
    if src.empty:
        return {}, {(i, j): 0.0 for i in markets for j in markets if i != j}

    pivot = (
        src.pivot_table(index="delivery_start", columns="market", values="target", aggfunc="mean")
        .sort_index()
    )
    pair_series: dict[tuple[str, str], pd.Series] = {}
    pair_last: dict[tuple[str, str], float] = {}

    for i in markets:
        for j in markets:
            if i == j:
                continue
            if i not in pivot.columns or j not in pivot.columns:
                pair_series[(i, j)] = pd.Series(dtype=float)
                pair_last[(i, j)] = 0.0
                continue
            s = pivot[i].rolling(window=lookback_hours, min_periods=min_periods).corr(pivot[j]).shift(1)
            s = s.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
            pair_series[(i, j)] = s
            valid = s.dropna()
            if not valid.empty:
                pair_last[(i, j)] = float(valid.iloc[-1])
            else:
                c = pivot[i].corr(pivot[j])
                pair_last[(i, j)] = float(c) if pd.notna(c) else 0.0
    return pair_series, pair_last


def _add_peer_global_prediction_features(
    df: pd.DataFrame,
    *,
    corr_source_df: pd.DataFrame,
    corr_lookback_hours: int,
    corr_min_periods: int,
) -> tuple[pd.DataFrame, list[str]]:
    if "global_pred_feature" not in df.columns:
        raise ValueError("Missing required column: global_pred_feature")

    out = df.copy()
    out["market"] = out["market"].astype(str)
    ds = pd.to_datetime(out["delivery_start"], errors="coerce")
    markets = sorted(out["market"].dropna().unique().tolist())
    if not markets:
        return out, []

    pred_pivot = (
        out.assign(_ds=ds)
        .pivot_table(index="_ds", columns="market", values="global_pred_feature", aggfunc="mean")
        .sort_index()
    )

    market_col_map: dict[str, str] = {}
    peer_cols: list[str] = []
    for market in markets:
        col = f"global_pred_market_{_safe_name(market)}"
        s = pred_pivot[market] if market in pred_pivot.columns else pd.Series(dtype=float)
        out[col] = ds.map(s)
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(out["global_pred_feature"]).astype(float)
        market_col_map[market] = col
        peer_cols.append(col)

    out["global_pred_self_market"] = out["global_pred_feature"].astype(float)
    peer_cols.append("global_pred_self_market")

    n = len(out)
    peer_mean = np.full(n, np.nan, dtype=float)
    peer_median = np.full(n, np.nan, dtype=float)
    peer_std = np.full(n, np.nan, dtype=float)
    market_series = out["market"].to_numpy()
    for market in markets:
        mask = market_series == market
        other_cols = [market_col_map[m] for m in markets if m != market]
        if not other_cols:
            peer_mean[mask] = out.loc[mask, "global_pred_self_market"].to_numpy(dtype=float)
            peer_median[mask] = peer_mean[mask]
            peer_std[mask] = 0.0
            continue
        part = out.loc[mask, other_cols]
        peer_mean[mask] = part.mean(axis=1).to_numpy(dtype=float)
        peer_median[mask] = part.median(axis=1).to_numpy(dtype=float)
        peer_std[mask] = part.std(axis=1).fillna(0.0).to_numpy(dtype=float)
    out["global_pred_peer_mean"] = peer_mean
    out["global_pred_peer_median"] = peer_median
    out["global_pred_peer_std"] = peer_std
    peer_cols.extend(["global_pred_peer_mean", "global_pred_peer_median", "global_pred_peer_std"])

    spread_cols: list[str] = []
    for market in markets:
        col = f"global_pred_spread_vs_{_safe_name(market)}"
        out[col] = out["global_pred_self_market"] - out[market_col_map[market]]
        spread_cols.append(col)
    peer_cols.extend(spread_cols)

    corr_series, corr_last = _build_pair_corr_lookup(
        corr_source_df,
        markets,
        lookback_hours=max(int(corr_lookback_hours), 24),
        min_periods=max(int(corr_min_periods), 12),
    )
    corr_signal = np.full(n, np.nan, dtype=float)
    corr_strength = np.zeros(n, dtype=float)
    for market in markets:
        mask = market_series == market
        if not np.any(mask):
            continue
        idx = np.where(mask)[0]
        ts = ds.iloc[idx]
        weighted = np.zeros(len(idx), dtype=float)
        denom = np.zeros(len(idx), dtype=float)
        for other in markets:
            if other == market:
                continue
            s = corr_series.get((market, other), pd.Series(dtype=float))
            if s.empty:
                cvals = np.full(len(idx), np.nan, dtype=float)
            else:
                cvals = ts.map(s).to_numpy(dtype=float)
            fallback = float(corr_last.get((market, other), 0.0))
            cvals = np.where(np.isnan(cvals), fallback, cvals)
            other_pred = out.loc[idx, market_col_map[other]].to_numpy(dtype=float)
            weighted += cvals * other_pred
            denom += np.abs(cvals)
        base = out.loc[idx, "global_pred_peer_mean"].to_numpy(dtype=float)
        signal = np.where(denom > 1e-6, weighted / denom, base)
        corr_signal[idx] = signal
        corr_strength[idx] = denom / max(len(markets) - 1, 1)

    out["global_pred_peer_corr_signal"] = corr_signal
    out["global_pred_peer_corr_strength"] = corr_strength
    peer_cols.extend(["global_pred_peer_corr_signal", "global_pred_peer_corr_strength"])

    return out, peer_cols


@dataclass
class HoldoutCalibrationArtifacts:
    predictions: pd.DataFrame
    family_rmse: dict[str, float]


def _fit_peak_hour_bias_calibrator_from_holdout(
    holdout_df: pd.DataFrame,
    *,
    families: list[str],
    family_weights: dict[str, float] | None,
    peak_hours: list[int],
    min_rows: int,
) -> pd.DataFrame:
    if holdout_df.empty:
        return pd.DataFrame(columns=["market", "hour", "bias", "rows"])
    w = _normalize_family_weights(family_weights, families)
    pred_cols = [f"pred_{f}" for f in families]
    if not set(pred_cols).issubset(holdout_df.columns):
        return pd.DataFrame(columns=["market", "hour", "bias", "rows"])

    stack = np.column_stack([holdout_df[c].to_numpy(dtype=float) for c in pred_cols])
    weights = np.array([w[f] for f in families], dtype=float)
    ensemble_pred = np.sum(stack * weights, axis=1)

    tmp = holdout_df[["market", "delivery_start", "target"]].copy()
    tmp["market"] = tmp["market"].astype(str)
    tmp["hour"] = pd.to_datetime(tmp["delivery_start"], errors="coerce").dt.hour
    tmp["target"] = pd.to_numeric(tmp["target"], errors="coerce")
    tmp["pred"] = ensemble_pred
    tmp = tmp.dropna(subset=["target", "hour"])
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
    work = key_df[["market", "delivery_start"]].copy()
    work["market"] = work["market"].astype(str)
    work["hour"] = pd.to_datetime(work["delivery_start"], errors="coerce").dt.hour.fillna(-1).astype(int)
    lookup = {
        (str(r.market), int(r.hour)): float(r.bias)
        for r in calibrator_df.itertuples(index=False)
    }
    adj = pd.Series(list(zip(work["market"], work["hour"]))).map(lookup).fillna(0.0).to_numpy(dtype=float)
    out = pred + adj
    n_adj = int(np.count_nonzero(np.abs(adj) > 1e-12))
    return out.astype(float), n_adj


def _train_multifamily_global_local(
    train_feat: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    families: list[str],
    seed: int,
    family_weights: dict[str, float] | None = None,
    enable_peer_global_features: bool = False,
    peer_corr_lookback_hours: int = 24 * 90,
    peer_corr_min_periods: int = 72,
) -> dict[str, FamilyBundle]:
    bundles: dict[str, FamilyBundle] = {}
    global_preds: dict[str, np.ndarray] = {}

    for i, family in enumerate(families):
        model = _make_model(
            family,
            feature_cols=feature_cols,
            cat_cols=cat_cols,
            random_seed=seed + i,
        )
        _fit_model(model, family, train_feat, train_feat["target"].to_numpy(dtype=float), feature_cols, cat_cols)
        global_preds[family] = _predict_model(model, family, train_feat, feature_cols)
        bundles[family] = FamilyBundle(
            family=family,
            global_model=model,
            local_models={},
            feature_cols=feature_cols,
            local_feature_cols=[],
            cat_cols=cat_cols,
        )

    weights = _normalize_family_weights(family_weights, families)
    global_stack = np.column_stack([global_preds[f] for f in families])
    weight_arr = np.array([weights[f] for f in families], dtype=float)
    global_pred_mean = np.sum(global_stack * weight_arr, axis=1)

    work = train_feat.copy()
    work["global_pred_feature"] = global_pred_mean
    peer_feature_cols: list[str] = []
    if enable_peer_global_features:
        work, peer_feature_cols = _add_peer_global_prediction_features(
            work,
            corr_source_df=train_feat,
            corr_lookback_hours=peer_corr_lookback_hours,
            corr_min_periods=peer_corr_min_periods,
        )

    local_feature_cols = feature_cols + ["global_pred_feature"] + peer_feature_cols
    for family in families:
        bundles[family].local_feature_cols = local_feature_cols

    for family in families:
        local_model_seed_base = seed + 1000 + families.index(family) * 100
        for j, (market, mdf) in enumerate(work.groupby("market", dropna=False)):
            local_model = _make_model(
                family,
                feature_cols=bundles[family].local_feature_cols,
                cat_cols=cat_cols,
                random_seed=local_model_seed_base + j,
            )
            _fit_model(
                local_model,
                family,
                mdf,
                mdf["target"].to_numpy(dtype=float),
                bundles[family].local_feature_cols,
                cat_cols,
            )
            bundles[family].local_models[str(market)] = local_model

    return bundles


def _predict_multifamily(
    test_feat: pd.DataFrame,
    test_market_keys: pd.DataFrame,
    bundles: dict[str, FamilyBundle],
    families: list[str],
    family_weights: dict[str, float] | None = None,
    enable_peer_global_features: bool = False,
    peer_corr_source_df: pd.DataFrame | None = None,
    peer_corr_lookback_hours: int = 24 * 90,
    peer_corr_min_periods: int = 72,
    return_family_preds: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    work = test_feat.copy()
    family_global_preds: dict[str, np.ndarray] = {}
    for family in families:
        b = bundles[family]
        family_global_preds[family] = _predict_model(b.global_model, family, work, b.feature_cols)

    weights = _normalize_family_weights(family_weights, families)
    global_stack = np.column_stack([family_global_preds[f] for f in families])
    weight_arr = np.array([weights[f] for f in families], dtype=float)
    global_mean = np.sum(global_stack * weight_arr, axis=1)
    work["global_pred_feature"] = global_mean
    if enable_peer_global_features:
        if peer_corr_source_df is None:
            raise ValueError("peer_corr_source_df is required when peer global features are enabled.")
        work, _ = _add_peer_global_prediction_features(
            work,
            corr_source_df=peer_corr_source_df,
            corr_lookback_hours=peer_corr_lookback_hours,
            corr_min_periods=peer_corr_min_periods,
        )

    family_local_preds = {f: np.full(len(work), np.nan, dtype=float) for f in families}
    for market, idx in test_market_keys.groupby("market", dropna=False).groups.items():
        for family in families:
            b = bundles[family]
            local_model = b.local_models.get(str(market))
            if local_model is None:
                family_local_preds[family][idx] = global_mean[idx]
                continue
            family_pred = _predict_model(local_model, family, work.loc[idx], b.local_feature_cols)
            family_local_preds[family][idx] = family_pred

    out_stack = np.column_stack([family_local_preds[f] for f in families])
    out = np.sum(out_stack * weight_arr, axis=1)

    if np.isnan(out).any():
        raise ValueError("NaNs in multi-family predictions.")
    if return_family_preds:
        return out.astype(float), family_local_preds
    return out.astype(float)


def _run_recent_holdout_calibration(
    train_df_raw: pd.DataFrame,
    *,
    holdout_days: int,
    add_temperature_demand: bool,
    add_physics_regime: bool,
    drop_redundant_features: bool,
    use_permutation_pruned_feature_set: bool,
    families: list[str],
    seed: int,
    enable_peer_global_features: bool,
    peer_corr_lookback_hours: int,
    peer_corr_min_periods: int,
) -> HoldoutCalibrationArtifacts | None:
    start_all = pd.to_datetime(train_df_raw["delivery_start"], errors="coerce")
    if start_all.isna().all():
        print("Holdout calibration skipped: could not parse delivery_start.")
        return None
    max_ts = start_all.max()
    cutoff = max_ts - pd.Timedelta(days=max(int(holdout_days), 1))
    tr_raw = train_df_raw.loc[start_all < cutoff].copy()
    va_raw = train_df_raw.loc[start_all >= cutoff].copy()
    if tr_raw.empty or va_raw.empty:
        print(
            "Holdout calibration skipped: insufficient rows "
            f"(train={len(tr_raw)}, holdout={len(va_raw)})."
        )
        return None

    tr_feat, va_feat = build_feature_table(
        tr_raw,
        va_raw.drop(columns=["target"]).copy(),
        add_temperature_demand=add_temperature_demand,
        add_physics_regime=add_physics_regime,
    )

    base_drop = {"id", "target", "delivery_start", "delivery_end"}
    feat_cols = [c for c in tr_feat.columns if c not in base_drop]
    feat_cols, _ = maybe_drop_redundant_features(feat_cols, enabled=drop_redundant_features)
    feat_cols, _ = apply_permutation_pruned_feature_policy(
        feat_cols,
        enabled=use_permutation_pruned_feature_set,
    )

    cat_cols = [
        c
        for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
        if c in feat_cols
    ]

    bundles = _train_multifamily_global_local(
        train_feat=tr_feat,
        feature_cols=feat_cols,
        cat_cols=cat_cols,
        families=families,
        seed=seed + 77777,
        family_weights=None,
        enable_peer_global_features=enable_peer_global_features,
        peer_corr_lookback_hours=peer_corr_lookback_hours,
        peer_corr_min_periods=peer_corr_min_periods,
    )

    va_keys = va_feat[["id", "market", "delivery_start"]].copy()
    _, family_preds = _predict_multifamily(
        va_feat,
        va_keys,
        bundles,
        families,
        family_weights=None,
        enable_peer_global_features=enable_peer_global_features,
        peer_corr_source_df=tr_feat,
        peer_corr_lookback_hours=peer_corr_lookback_hours,
        peer_corr_min_periods=peer_corr_min_periods,
        return_family_preds=True,
    )

    holdout = va_keys.copy()
    holdout = holdout.merge(va_raw[["id", "target"]], on="id", how="left")
    family_rmse: dict[str, float] = {}
    for family in families:
        col = f"pred_{family}"
        holdout[col] = family_preds[family]
        valid = holdout["target"].notna() & np.isfinite(holdout[col])
        if valid.any():
            family_rmse[family] = _rmse(
                holdout.loc[valid, "target"].to_numpy(dtype=float),
                holdout.loc[valid, col].to_numpy(dtype=float),
            )

    print(
        "Holdout calibration: "
        f"train_rows={len(tr_raw)} holdout_rows={len(va_raw)} cutoff={cutoff.date()}"
    )
    if family_rmse:
        msg = ", ".join(f"{f}={family_rmse[f]:.6f}" for f in families if f in family_rmse)
        print(f"Holdout family RMSE: {msg}")
    return HoldoutCalibrationArtifacts(predictions=holdout, family_rmse=family_rmse)


def run_time_series_cv(
    train_df_raw: pd.DataFrame,
    *,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
    add_temperature_demand: bool,
    add_physics_regime: bool,
    drop_redundant_features: bool,
    use_permutation_pruned_feature_set: bool,
    families: list[str],
    seed: int,
    ensemble_weight_map: dict[str, float] | None,
    apply_market_hour_quantile_guardrails: bool,
    guardrail_lower_quantile: float,
    guardrail_upper_quantile: float,
    peak_hour_bias_calibrator: bool,
    peak_hours: list[int],
    peak_bias_min_rows: int,
    enable_peer_global_features: bool,
    peer_corr_lookback_hours: int,
    peer_corr_min_periods: int,
) -> tuple[float | None, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv_start = time.perf_counter()
    folds = make_time_series_folds(
        train_df=train_df_raw,
        n_folds=n_folds,
        val_days=val_days,
        step_days=step_days,
        min_train_days=min_train_days,
    )
    if not folds:
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    start_all = pd.to_datetime(train_df_raw["delivery_start"], errors="coerce")
    fold_rows: list[dict[str, object]] = []
    oof_rows: list[pd.DataFrame] = []

    for fold_idx, (val_start, val_end) in enumerate(folds, start=1):
        fold_start = time.perf_counter()
        tr_mask = start_all < val_start
        va_mask = (start_all >= val_start) & (start_all < val_end)
        tr = train_df_raw.loc[tr_mask].copy()
        va = train_df_raw.loc[va_mask].copy()
        if tr.empty or va.empty:
            continue

        tr_feat, va_feat = build_feature_table(
            tr,
            va.drop(columns=["target"]).copy(),
            add_temperature_demand=add_temperature_demand,
            add_physics_regime=add_physics_regime,
        )

        base_drop = {"id", "target", "delivery_start", "delivery_end"}
        feat_cols = [c for c in tr_feat.columns if c not in base_drop]
        feat_cols, _ = maybe_drop_redundant_features(feat_cols, enabled=drop_redundant_features)
        feat_cols, _ = apply_permutation_pruned_feature_policy(
            feat_cols,
            enabled=use_permutation_pruned_feature_set,
        )

        cat_cols = [
            c
            for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
            if c in feat_cols
        ]

        bundles = _train_multifamily_global_local(
            train_feat=tr_feat,
            feature_cols=feat_cols,
            cat_cols=cat_cols,
            families=families,
            seed=seed + fold_idx * 10000,
            family_weights=ensemble_weight_map,
            enable_peer_global_features=enable_peer_global_features,
            peer_corr_lookback_hours=peer_corr_lookback_hours,
            peer_corr_min_periods=peer_corr_min_periods,
        )

        va_keys = va_feat[["id", "market", "delivery_start"]].copy()
        pred = _predict_multifamily(
            va_feat,
            va_keys,
            bundles,
            families,
            family_weights=ensemble_weight_map,
            enable_peer_global_features=enable_peer_global_features,
            peer_corr_source_df=tr_feat,
            peer_corr_lookback_hours=peer_corr_lookback_hours,
            peer_corr_min_periods=peer_corr_min_periods,
        )

        if peak_hour_bias_calibrator:
            tr_keys = tr_feat[["id", "market", "delivery_start"]].copy()
            _, tr_family_preds = _predict_multifamily(
                tr_feat,
                tr_keys,
                bundles,
                families,
                family_weights=ensemble_weight_map,
                enable_peer_global_features=enable_peer_global_features,
                peer_corr_source_df=tr_feat,
                peer_corr_lookback_hours=peer_corr_lookback_hours,
                peer_corr_min_periods=peer_corr_min_periods,
                return_family_preds=True,
            )
            tr_holdout = tr_keys.copy()
            tr_holdout["target"] = tr_feat["target"].to_numpy(dtype=float)
            for family in families:
                tr_holdout[f"pred_{family}"] = tr_family_preds[family]
            fold_cal = _fit_peak_hour_bias_calibrator_from_holdout(
                tr_holdout,
                families=families,
                family_weights=ensemble_weight_map,
                peak_hours=peak_hours,
                min_rows=peak_bias_min_rows,
            )
            pred, fold_adj = _apply_peak_hour_bias_calibrator(pred, va_keys, fold_cal)
            if fold_adj > 0:
                print(f"CV fold {fold_idx}: peak-hour calibrator adjusted rows={fold_adj}")

        if apply_market_hour_quantile_guardrails:
            bounds = _compute_market_hour_quantile_bounds(
                tr,
                lower_q=guardrail_lower_quantile,
                upper_q=guardrail_upper_quantile,
            )
            pred, n_clipped = _apply_market_hour_quantile_guardrails(pred, va_keys, bounds)
            if n_clipped > 0:
                print(f"CV fold {fold_idx}: guardrails clipped rows={n_clipped}")

        y_val = va["target"].to_numpy(dtype=float)
        fold_rmse = _rmse(y_val, pred)

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

        fold_elapsed = time.perf_counter() - fold_start
        cv_elapsed = time.perf_counter() - cv_start
        print(
            f"CV fold {fold_idx}/{len(folds)} RMSE={fold_rmse:.6f} "
            f"| fold_elapsed={_format_duration(fold_elapsed)} "
            f"| cv_elapsed={_format_duration(cv_elapsed)}"
        )

    cv_oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    overall = None
    if not cv_oof.empty:
        overall = _rmse(cv_oof["target"].to_numpy(dtype=float), cv_oof["pred"].to_numpy(dtype=float))

    cv_market = pd.DataFrame()
    if not cv_oof.empty:
        rows: list[dict[str, object]] = []
        for market, sdf in cv_oof.groupby("market", dropna=False):
            rows.append(
                {
                    "market": str(market),
                    "rows": int(len(sdf)),
                    "rmse": _rmse(sdf["target"].to_numpy(dtype=float), sdf["pred"].to_numpy(dtype=float)),
                }
            )
        cv_market = pd.DataFrame(rows).sort_values("market").reset_index(drop=True)

    cv_total_elapsed = time.perf_counter() - cv_start
    print(f"CV done | total_elapsed={_format_duration(cv_total_elapsed)} | oof_rmse={overall}")
    return overall, pd.DataFrame(fold_rows), cv_oof, cv_market


def _save_models(run_dir: Path, bundles: dict[str, FamilyBundle], families: list[str]) -> dict[str, object]:
    import joblib

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, object] = {"families": {}}
    for family in families:
        b = bundles[family]
        fam_meta: dict[str, object] = {"global_model": None, "local_models": {}}

        if family == "catboost":
            global_path = models_dir / f"global_{family}.cbm"
            b.global_model.save_model(global_path.as_posix())
            fam_meta["global_model"] = global_path.name

            for market, model in b.local_models.items():
                safe_market = market.replace(" ", "_")
                path = models_dir / f"local_{family}_{safe_market}.cbm"
                model.save_model(path.as_posix())
                fam_meta["local_models"][market] = path.name
        else:
            global_path = models_dir / f"global_{family}.joblib"
            joblib.dump(b.global_model, global_path)
            fam_meta["global_model"] = global_path.name

            for market, model in b.local_models.items():
                safe_market = market.replace(" ", "_")
                path = models_dir / f"local_{family}_{safe_market}.joblib"
                joblib.dump(model, path)
                fam_meta["local_models"][market] = path.name

        out["families"][family] = fam_meta

    print(f"Saved models dir: {models_dir}")
    return out


def main() -> None:
    overall_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Per-market interactions training with multi-model family ensemble.")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="per_market_interactions_multimodel")
    parser.add_argument("--exclude-2023", action="store_true")
    parser.add_argument("--exclude-2023-keep-from-month", type=int, default=10)
    parser.add_argument(
        "--train-start-oct-2023",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict training rows to delivery_start >= 2023-10-01.",
    )
    parser.add_argument(
        "--add-temperature-demand-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--add-physics-regime-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--drop-redundant-features",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--use-permutation-pruned-feature-set",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-val-days", type=int, default=14)
    parser.add_argument("--cv-step-days", type=int, default=14)
    parser.add_argument("--cv-min-train-days", type=int, default=90)
    parser.add_argument(
        "--models",
        default="catboost,lightgbm,xgboost,ridge,lasso,rf",
        help="Comma-separated model families.",
    )
    parser.add_argument(
        "--allow-missing-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, skip model families with missing dependencies.",
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Use weighted family averaging. If provided without value, uses 'auto' "
            "(inverse holdout RMSE). You can also pass explicit mapping, e.g. "
            "catboost=0.4,lightgbm=0.2,xgboost=0.1,ridge=0.1,lasso=0.1,rf=0.1"
        ),
    )
    parser.add_argument(
        "--ensemble-weight-calib-days",
        type=int,
        default=21,
        help="Recent holdout days used for auto ensemble weights / peak-hour calibration (default: 21).",
    )
    parser.add_argument(
        "--apply-market-hour-quantile-guardrails",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Clip predictions to market-hour quantile bands from training history.",
    )
    parser.add_argument("--guardrail-lower-quantile", type=float, default=0.01)
    parser.add_argument("--guardrail-upper-quantile", type=float, default=0.99)
    parser.add_argument(
        "--peak-hour-bias-calibrator",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply market-hour additive bias correction on selected peak hours.",
    )
    parser.add_argument(
        "--peak-hours",
        default="17,18,19,20",
        help="Comma-separated peak hours for bias calibrator (default: 17,18,19,20).",
    )
    parser.add_argument(
        "--peak-bias-min-rows",
        type=int,
        default=12,
        help="Minimum holdout rows per market-hour to estimate a peak-hour bias offset.",
    )
    parser.add_argument(
        "--enable-peer-global-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add peer global-prediction features to local models: cross-market global preds, "
            "spreads, peer aggregates, and correlation-weighted peer signal."
        ),
    )
    parser.add_argument(
        "--peer-corr-lookback-hours",
        type=int,
        default=24 * 90,
        help="Lookback window (hours) for moving inter-market correlation estimates.",
    )
    parser.add_argument(
        "--peer-corr-min-periods",
        type=int,
        default=72,
        help="Minimum periods for moving inter-market correlation estimates.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if not (0.0 < args.guardrail_lower_quantile < 1.0):
        raise ValueError("--guardrail-lower-quantile must be in (0,1).")
    if not (0.0 < args.guardrail_upper_quantile < 1.0):
        raise ValueError("--guardrail-upper-quantile must be in (0,1).")
    if args.guardrail_lower_quantile >= args.guardrail_upper_quantile:
        raise ValueError("--guardrail-lower-quantile must be < --guardrail-upper-quantile.")
    peak_hours = _parse_peak_hours(args.peak_hours)

    families = _parse_model_families(args.models)
    families = _resolve_optional_families(families, allow_missing=args.allow_missing_models)
    print(f"Using model families: {families}")

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_path = Path(args.sample_submission)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    if args.exclude_2023:
        train_df = apply_exclude_2023(train_df, keep_from_month=args.exclude_2023_keep_from_month)
    if args.train_start_oct_2023:
        train_df = apply_train_start_cutoff(train_df, start_date="2023-10-01")

    holdout_artifacts: HoldoutCalibrationArtifacts | None = None
    resolved_weight_map: dict[str, float] | None
    if args.ensemble_weights is None:
        resolved_weight_map = _normalize_family_weights(None, families)
    elif args.ensemble_weights != "auto":
        explicit_weight_map = _parse_explicit_ensemble_weights(args.ensemble_weights, families)
        resolved_weight_map = _normalize_family_weights(explicit_weight_map, families)
    else:
        resolved_weight_map = None

    needs_holdout = (args.ensemble_weights == "auto") or args.peak_hour_bias_calibrator
    if needs_holdout:
        holdout_phase_start = time.perf_counter()
        holdout_artifacts = _run_recent_holdout_calibration(
            train_df_raw=train_df,
            holdout_days=args.ensemble_weight_calib_days,
            add_temperature_demand=args.add_temperature_demand_features,
            add_physics_regime=args.add_physics_regime_features,
            drop_redundant_features=args.drop_redundant_features,
            use_permutation_pruned_feature_set=args.use_permutation_pruned_feature_set,
            families=families,
            seed=args.seed,
            enable_peer_global_features=args.enable_peer_global_features,
            peer_corr_lookback_hours=args.peer_corr_lookback_hours,
            peer_corr_min_periods=args.peer_corr_min_periods,
        )
        print(f"Holdout calibration elapsed: {_format_duration(time.perf_counter() - holdout_phase_start)}")

    if args.ensemble_weights == "auto":
        if holdout_artifacts is not None and holdout_artifacts.family_rmse:
            family_rmse = holdout_artifacts.family_rmse
            if any(f not in family_rmse for f in families):
                available = [family_rmse[f] for f in families if f in family_rmse]
                fallback_rmse = float(np.mean(available)) if available else 1.0
                for f in families:
                    family_rmse.setdefault(f, fallback_rmse)
            inv = {f: 1.0 / max(float(family_rmse[f]), 1e-6) ** 2 for f in families}
            resolved_weight_map = _normalize_family_weights(inv, families)
            msg = ", ".join(f"{f}={resolved_weight_map[f]:.4f}" for f in families)
            print(f"Auto ensemble weights (inverse holdout RMSE^2): {msg}")
        else:
            resolved_weight_map = _normalize_family_weights(None, families)
            print("Auto ensemble weights fallback: holdout unavailable, using equal weights.")

    if resolved_weight_map is None:
        resolved_weight_map = _normalize_family_weights(None, families)
    print("Ensemble weights: " + ", ".join(f"{f}={resolved_weight_map[f]:.4f}" for f in families))

    cv_rmse = None
    cv_results = pd.DataFrame()
    cv_oof = pd.DataFrame()
    cv_market = pd.DataFrame()
    if args.cv:
        cv_phase_start = time.perf_counter()
        cv_rmse, cv_results, cv_oof, cv_market = run_time_series_cv(
            train_df_raw=train_df,
            n_folds=args.cv_folds,
            val_days=args.cv_val_days,
            step_days=args.cv_step_days,
            min_train_days=args.cv_min_train_days,
            add_temperature_demand=args.add_temperature_demand_features,
            add_physics_regime=args.add_physics_regime_features,
            drop_redundant_features=args.drop_redundant_features,
            use_permutation_pruned_feature_set=args.use_permutation_pruned_feature_set,
            families=families,
            seed=args.seed,
            ensemble_weight_map=resolved_weight_map,
            apply_market_hour_quantile_guardrails=args.apply_market_hour_quantile_guardrails,
            guardrail_lower_quantile=args.guardrail_lower_quantile,
            guardrail_upper_quantile=args.guardrail_upper_quantile,
            peak_hour_bias_calibrator=args.peak_hour_bias_calibrator,
            peak_hours=peak_hours,
            peak_bias_min_rows=args.peak_bias_min_rows,
            enable_peer_global_features=args.enable_peer_global_features,
            peer_corr_lookback_hours=args.peer_corr_lookback_hours,
            peer_corr_min_periods=args.peer_corr_min_periods,
        )
        print(f"CV phase elapsed: {_format_duration(time.perf_counter() - cv_phase_start)}")

    feature_phase_start = time.perf_counter()
    train_feat, test_feat = build_feature_table(
        train_df,
        test_df,
        add_temperature_demand=args.add_temperature_demand_features,
        add_physics_regime=args.add_physics_regime_features,
    )

    base_drop = {"id", "target", "delivery_start", "delivery_end"}
    candidate_features = [c for c in train_feat.columns if c not in base_drop]
    candidate_features, dropped_redundant = maybe_drop_redundant_features(
        candidate_features,
        enabled=args.drop_redundant_features,
    )
    if dropped_redundant:
        print(f"Dropped redundant features: {len(dropped_redundant)}")

    candidate_features, dropped_policy = apply_permutation_pruned_feature_policy(
        candidate_features,
        enabled=args.use_permutation_pruned_feature_set,
    )
    if dropped_policy:
        print(
            "Permutation-pruned feature policy active: "
            f"kept={len(candidate_features)} dropped={len(dropped_policy)}"
        )
    print(f"Feature prep elapsed: {_format_duration(time.perf_counter() - feature_phase_start)}")

    cat_cols = [
        c
        for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
        if c in candidate_features
    ]

    train_phase_start = time.perf_counter()
    bundles = _train_multifamily_global_local(
        train_feat=train_feat,
        feature_cols=candidate_features,
        cat_cols=cat_cols,
        families=families,
        seed=args.seed,
        family_weights=resolved_weight_map,
        enable_peer_global_features=args.enable_peer_global_features,
        peer_corr_lookback_hours=args.peer_corr_lookback_hours,
        peer_corr_min_periods=args.peer_corr_min_periods,
    )
    print(f"Main train elapsed: {_format_duration(time.perf_counter() - train_phase_start)}")

    pred_phase_start = time.perf_counter()
    test_keys = test_feat[["id", "market", "delivery_start"]].copy()
    pred = _predict_multifamily(
        test_feat,
        test_keys,
        bundles,
        families,
        family_weights=resolved_weight_map,
        enable_peer_global_features=args.enable_peer_global_features,
        peer_corr_source_df=train_feat,
        peer_corr_lookback_hours=args.peer_corr_lookback_hours,
        peer_corr_min_periods=args.peer_corr_min_periods,
    )

    peak_calibrator_df = pd.DataFrame(columns=["market", "hour", "bias", "rows"])
    peak_adjusted_rows = 0
    if args.peak_hour_bias_calibrator:
        if holdout_artifacts is not None:
            peak_calibrator_df = _fit_peak_hour_bias_calibrator_from_holdout(
                holdout_artifacts.predictions,
                families=families,
                family_weights=resolved_weight_map,
                peak_hours=peak_hours,
                min_rows=args.peak_bias_min_rows,
            )
        if peak_calibrator_df.empty:
            train_keys = train_feat[["id", "market", "delivery_start"]].copy()
            _, train_family_preds = _predict_multifamily(
                train_feat,
                train_keys,
                bundles,
                families,
                family_weights=resolved_weight_map,
                enable_peer_global_features=args.enable_peer_global_features,
                peer_corr_source_df=train_feat,
                peer_corr_lookback_hours=args.peer_corr_lookback_hours,
                peer_corr_min_periods=args.peer_corr_min_periods,
                return_family_preds=True,
            )
            in_sample = train_keys.copy()
            in_sample["target"] = train_feat["target"].to_numpy(dtype=float)
            for family in families:
                in_sample[f"pred_{family}"] = train_family_preds[family]
            peak_calibrator_df = _fit_peak_hour_bias_calibrator_from_holdout(
                in_sample,
                families=families,
                family_weights=resolved_weight_map,
                peak_hours=peak_hours,
                min_rows=args.peak_bias_min_rows,
            )
        pred, peak_adjusted_rows = _apply_peak_hour_bias_calibrator(pred, test_keys, peak_calibrator_df)
        print(
            "Peak-hour bias calibrator: "
            f"rows_in_calibrator={len(peak_calibrator_df)} adjusted_rows={peak_adjusted_rows}"
        )

    guardrail_clipped_rows = 0
    if args.apply_market_hour_quantile_guardrails:
        bounds = _compute_market_hour_quantile_bounds(
            train_df,
            lower_q=args.guardrail_lower_quantile,
            upper_q=args.guardrail_upper_quantile,
        )
        pred, guardrail_clipped_rows = _apply_market_hour_quantile_guardrails(pred, test_keys, bounds)
        print(f"Market-hour guardrails clipped rows: {guardrail_clipped_rows}")

    out_sub = sample[["id"]].copy()
    pred_map = pd.Series(pred, index=test_feat["id"].astype(int))
    out_sub["target"] = out_sub["id"].astype(int).map(pred_map)
    if out_sub["target"].isna().any():
        raise ValueError("Submission has NaN targets after mapping.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.out_dir) / f"{stamp}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    submission_path = run_dir / "submission.csv"
    out_sub.to_csv(submission_path, index=False)

    latest_path = Path("csv/submission_per_market_interactions_multimodel.csv")
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    out_sub.to_csv(latest_path, index=False)

    if not cv_results.empty:
        cv_results.to_csv(run_dir / "cv_results.csv", index=False)
    if not cv_oof.empty:
        cv_oof.to_csv(run_dir / "cv_oof.csv", index=False)
    if not cv_market.empty:
        cv_market.to_csv(run_dir / "cv_market_results.csv", index=False)

    model_file_map: dict[str, object] = {}
    if args.save_models:
        model_file_map = _save_models(run_dir, bundles, families)

    local_feature_cols = bundles[families[0]].local_feature_cols if families else []
    model_metadata = {
        "model_families": families,
        "ensemble_weights": resolved_weight_map,
        "candidate_features_before_global_pred": candidate_features,
        "local_feature_cols": local_feature_cols,
        "cat_cols": cat_cols,
        "cv_rmse": cv_rmse,
        "holdout_family_rmse": (holdout_artifacts.family_rmse if holdout_artifacts is not None else {}),
        "peak_bias_calibrator_rows": int(len(peak_calibrator_df)),
        "peak_bias_adjusted_rows": int(peak_adjusted_rows),
        "guardrail_clipped_rows": int(guardrail_clipped_rows),
        "train_args": vars(args),
        "data_hashes": {
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "sample_submission_sha256": _sha256_file(sample_path),
        },
        "model_files": model_file_map,
    }
    (run_dir / "model_metadata.json").write_text(
        json.dumps(model_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_config = {
        "script": Path(__file__).name,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "train_args": vars(args),
        "features": {
            "candidate_features_before_global_pred": candidate_features,
            "local_feature_cols": local_feature_cols,
            "cat_cols": cat_cols,
        },
        "metrics": {
            "cv_rmse": cv_rmse,
            "peak_bias_calibrator_rows": int(len(peak_calibrator_df)),
            "peak_bias_adjusted_rows": int(peak_adjusted_rows),
            "guardrail_clipped_rows": int(guardrail_clipped_rows),
        },
        "model_families": families,
        "ensemble_weights": resolved_weight_map,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Predict+save elapsed: {_format_duration(time.perf_counter() - pred_phase_start)}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved latest copy: {latest_path}")
    print(f"Model families: {families}")
    print("Ensemble weights: " + ", ".join(f"{f}={resolved_weight_map[f]:.4f}" for f in families))
    print(f"Features used: {len(candidate_features)}")
    print(f"Local features used: {len(local_feature_cols)}")
    print(f"Categorical features: {cat_cols}")
    print(f"Peer global features enabled: {args.enable_peer_global_features}")
    print(f"Peak-hour calibrator rows: {len(peak_calibrator_df)}")
    print(f"Guardrail clipped rows: {guardrail_clipped_rows}")
    print(f"CV RMSE: {cv_rmse}")
    print(f"Saved metadata: {run_dir / 'model_metadata.json'}")
    print(f"Total wall time: {_format_duration(time.perf_counter() - overall_start)}")


if __name__ == "__main__":
    main()
