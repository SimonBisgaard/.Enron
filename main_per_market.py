from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

LAG_STEPS = [1, 2, 24, 48, 168]
ROLL_WINDOWS = [24, 48, 168]


def _add_harmonics(df: pd.DataFrame, column: str, period: int, harmonics: tuple[int, ...] = (1, 2, 3)) -> None:
    for k in harmonics:
        radians = 2.0 * np.pi * k * (df[column] / period)
        df[f"{column}_sin_{k}"] = np.sin(radians)
        df[f"{column}_cos_{k}"] = np.cos(radians)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    start_ts = pd.to_datetime(result["delivery_start"], errors="coerce")
    end_ts = pd.to_datetime(result["delivery_end"], errors="coerce")

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
    result["delivery_start_ts"] = start_ts.astype("int64") // 10**9

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


def make_market_timestamp_folds(
    market_df: pd.DataFrame,
    n_splits: int,
    purge_timestamps: int,
    min_train_timestamps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_ts = np.array(sorted(market_df["delivery_start_ts"].unique()))
    if len(unique_ts) < (n_splits + 2):
        return []

    val_size = max(24, len(unique_ts) // (n_splits + 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold in range(n_splits):
        train_end = val_size * (fold + 1)
        if train_end < min_train_timestamps:
            continue

        valid_start = train_end + purge_timestamps
        valid_end = min(valid_start + val_size, len(unique_ts))
        if valid_end <= valid_start:
            break

        train_ts = unique_ts[:train_end]
        valid_ts = unique_ts[valid_start:valid_end]

        train_idx = market_df.index[market_df["delivery_start_ts"].isin(train_ts)].to_numpy()
        valid_idx = market_df.index[market_df["delivery_start_ts"].isin(valid_ts)].to_numpy()

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


def _calibrate_spike_threshold(probabilities: np.ndarray, spike_truth: np.ndarray) -> float:
    thresholds = np.linspace(0.15, 0.85, 15)
    scores = [
        _f1_score_binary(spike_truth.astype(int), (probabilities >= thr).astype(int))
        for thr in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def _fit_market_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_cols: list[str],
    numeric_cols: list[str],
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray | float]]:
    pos_spike_thr = float(y_train.quantile(0.95))
    neg_spike_thr = float(y_train.quantile(0.05))

    spike_labels_train = ((y_train >= pos_spike_thr) | (y_train <= neg_spike_thr)).astype(int)
    spike_labels_valid = ((y_valid >= pos_spike_thr) | (y_valid <= neg_spike_thr)).astype(int)

    spike_weights = np.ones(len(y_train), dtype=float)
    spike_weights[y_train.values >= pos_spike_thr] = 4.0
    spike_weights[y_train.values <= neg_spike_thr] = 3.0

    cat_normal = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2800,
        learning_rate=0.028,
        depth=8,
        l2_leaf_reg=24,
        bagging_temperature=0.5,
        random_strength=0.9,
        random_seed=seed,
        verbose=0,
    )
    cat_normal.fit(
        x_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=200,
    )

    cat_spike = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="RMSE",
        iterations=3200,
        learning_rate=0.022,
        depth=8,
        l2_leaf_reg=28,
        bagging_temperature=0.7,
        random_strength=1.1,
        random_seed=seed + 13,
        verbose=0,
    )
    cat_spike.fit(
        x_train,
        y_train,
        cat_features=cat_cols,
        sample_weight=spike_weights,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=200,
    )

    spike_classifier: CatBoostClassifier | None = None
    spike_threshold = 0.30
    p_spike_valid = np.full(len(x_valid), 0.15)

    if spike_labels_train.sum() >= 30 and spike_labels_train.sum() < len(spike_labels_train) - 30:
        spike_classifier = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=900,
            learning_rate=0.035,
            depth=6,
            l2_leaf_reg=14,
            random_seed=seed + 101,
            verbose=0,
        )
        spike_classifier.fit(
            x_train,
            spike_labels_train,
            cat_features=cat_cols,
        )
        p_spike_valid = spike_classifier.predict_proba(x_valid)[:, 1]
        spike_threshold = _calibrate_spike_threshold(p_spike_valid, spike_labels_valid.to_numpy())

    cat_normal_valid = cat_normal.predict(x_valid)
    cat_spike_valid = cat_spike.predict(x_valid)
    cat_regime_valid = np.where(p_spike_valid >= spike_threshold, cat_spike_valid, cat_normal_valid)

    hgb = HistGradientBoostingRegressor(
        learning_rate=0.04,
        max_depth=7,
        max_iter=700,
        min_samples_leaf=30,
        l2_regularization=1.5,
        random_state=seed,
    )
    hgb.fit(x_train[numeric_cols], y_train)
    hgb_valid = hgb.predict(x_valid[numeric_cols])

    models = {
        "cat_normal": cat_normal,
        "cat_spike": cat_spike,
        "spike_classifier": spike_classifier,
        "spike_threshold": spike_threshold,
        "hgb": hgb,
        "pos_spike_thr": pos_spike_thr,
        "neg_spike_thr": neg_spike_thr,
    }

    valid_outputs = {
        "cat_regime": cat_regime_valid,
        "hgb": hgb_valid,
    }
    return models, valid_outputs


def _recursive_predict_market(
    test_market_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    cat_cols: list[str],
    num_fill_values: dict[str, float],
    history_targets: list[float],
    models: dict[str, object],
    cat_weight: float,
    hgb_weight: float,
) -> pd.Series:
    preds: list[float] = []

    cat_normal: CatBoostRegressor = models["cat_normal"]  # type: ignore[assignment]
    cat_spike: CatBoostRegressor = models["cat_spike"]  # type: ignore[assignment]
    spike_classifier: CatBoostClassifier | None = models["spike_classifier"]  # type: ignore[assignment]
    spike_threshold: float = float(models["spike_threshold"])
    hgb: HistGradientBoostingRegressor = models["hgb"]  # type: ignore[assignment]

    for idx in test_market_df.index:
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

        cat_normal_pred = float(cat_normal.predict(row_df)[0])
        cat_spike_pred = float(cat_spike.predict(row_df)[0])

        if spike_classifier is not None:
            p_spike = float(spike_classifier.predict_proba(row_df)[:, 1][0])
        else:
            p_spike = 0.15

        cat_regime_pred = cat_spike_pred if p_spike >= spike_threshold else cat_normal_pred
        hgb_pred = float(hgb.predict(row_df[numeric_cols])[0])

        pred = cat_weight * cat_regime_pred + hgb_weight * hgb_pred
        preds.append(pred)
        history_targets.append(pred)

    return pd.Series(preds, index=test_market_df.index, dtype=float)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test_for_participants.csv"
    sample_submission_path = data_dir / "sample_submission.csv"
    output_path = base_dir / "submission_per_market.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_submission_path)

    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)

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

    print(f"Using {len(feature_cols)} features total")
    print(f"Categorical ({len(cat_cols)}): {cat_cols}")

    predictions_by_id: dict[int, float] = {}
    market_scores: list[tuple[str, float]] = []

    for market in sorted(train_df["market"].dropna().unique()):
        market_train = train_df[train_df["market"] == market].sort_values(["delivery_start_ts", "id"]).reset_index(drop=True)
        market_test = test_df[test_df["market"] == market].sort_values(["delivery_start_ts", "id"]).reset_index(drop=True)

        if len(market_train) < 400 or market_test.empty:
            continue

        for col in numeric_cols:
            median_value = float(market_train[col].median())
            market_train[col] = market_train[col].fillna(median_value)
            market_test[col] = market_test[col].fillna(median_value)

        for col in cat_cols:
            market_train[col] = market_train[col].fillna("missing").astype(str)
            market_test[col] = market_test[col].fillna("missing").astype(str)

        folds = make_market_timestamp_folds(
            market_df=market_train,
            n_splits=5,
            purge_timestamps=36,
            min_train_timestamps=24 * 10,
        )
        if len(folds) < 2:
            continue

        y_all = market_train["target"].copy()
        oof_cat = np.full(len(market_train), np.nan, dtype=float)
        oof_hgb = np.full(len(market_train), np.nan, dtype=float)
        fold_records: list[dict[str, float | int]] = []

        for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
            x_train = market_train.iloc[train_idx][feature_cols].copy()
            y_train = y_all.iloc[train_idx]
            x_valid = market_train.iloc[valid_idx][feature_cols].copy()
            y_valid = y_all.iloc[valid_idx]

            models, valid_outputs = _fit_market_models(
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                cat_cols=cat_cols,
                numeric_cols=numeric_cols,
                seed=42 + fold_id,
            )

            cat_pred = np.asarray(valid_outputs["cat_regime"], dtype=float)
            hgb_pred = np.asarray(valid_outputs["hgb"], dtype=float)

            oof_cat[valid_idx] = cat_pred
            oof_hgb[valid_idx] = hgb_pred

            rmse_cat = float(mean_squared_error(y_valid, cat_pred) ** 0.5)
            rmse_hgb = float(mean_squared_error(y_valid, hgb_pred) ** 0.5)
            fold_records.append(
                {
                    "fold": fold_id,
                    "rmse_cat": rmse_cat,
                    "rmse_hgb": rmse_hgb,
                }
            )
            print(f"{market} | Fold {fold_id}/{len(folds)} | RMSE cat={rmse_cat:.6f}, hgb={rmse_hgb:.6f}")

        valid_mask = (~np.isnan(oof_cat)) & (~np.isnan(oof_hgb))
        if not np.any(valid_mask):
            continue

        recent = sorted(fold_records, key=lambda x: int(x["fold"]))[-2:]
        cat_recent_rmse = float(np.mean([float(record["rmse_cat"]) for record in recent]))
        hgb_recent_rmse = float(np.mean([float(record["rmse_hgb"]) for record in recent]))

        inv_cat = 1.0 / (cat_recent_rmse + 1e-9)
        inv_hgb = 1.0 / (hgb_recent_rmse + 1e-9)
        cat_weight = inv_cat / (inv_cat + inv_hgb)
        hgb_weight = inv_hgb / (inv_cat + inv_hgb)

        oof_blend = cat_weight * oof_cat[valid_mask] + hgb_weight * oof_hgb[valid_mask]
        market_oof_rmse = float(mean_squared_error(y_all[valid_mask], oof_blend) ** 0.5)
        market_scores.append((market, market_oof_rmse))

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

        holdout_size = max(24, int(len(market_train) * 0.1))
        x_fit = x_full.iloc[:-holdout_size]
        y_fit = y_full.iloc[:-holdout_size]
        x_cal = x_full.iloc[-holdout_size:]
        y_cal = y_full.iloc[-holdout_size:]

        final_models, _ = _fit_market_models(
            x_train=x_fit,
            y_train=y_fit,
            x_valid=x_cal,
            y_valid=y_cal,
            cat_cols=cat_cols,
            numeric_cols=numeric_cols,
            seed=999,
        )

        num_fill_values = {col: float(market_train[col].median()) for col in numeric_cols}
        history_targets = y_full.astype(float).tolist()
        test_preds = _recursive_predict_market(
            test_market_df=market_test,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            cat_cols=cat_cols,
            num_fill_values=num_fill_values,
            history_targets=history_targets,
            models=final_models,
            cat_weight=cat_weight,
            hgb_weight=hgb_weight,
        )

        correction_alpha = 0.6
        correction_values = [
            correction_alpha * residual_correction.get((int(row["delivery_start_hour"]), int(row["delivery_start_dow"])), 0.0)
            for _, row in market_test.iterrows()
        ]
        calibrated_preds = test_preds.to_numpy() + np.array(correction_values, dtype=float)

        for i, pred in enumerate(calibrated_preds):
            test_id = int(market_test.iloc[i]["id"])
            predictions_by_id[test_id] = float(pred)

        print(
            f"{market} | OOF RMSE={market_oof_rmse:.6f} | recent-blend weights: cat={cat_weight:.3f}, hgb={hgb_weight:.3f}"
        )

    submission = sample_submission[["id"]].copy()
    submission["target"] = submission["id"].map(predictions_by_id)

    if submission["target"].isna().any():
        missing_count = int(submission["target"].isna().sum())
        raise ValueError(f"Missing predictions for {missing_count} ids.")

    submission.to_csv(output_path, index=False)

    print(f"Saved per-market submission: {output_path}")
    if market_scores:
        print("Per-market CV RMSE (timestamp folds):")
        for market, score in sorted(market_scores, key=lambda x: x[1]):
            print(f"  {market}: {score:.6f}")
        print(f"Mean per-market RMSE: {np.mean([score for _, score in market_scores]):.6f}")
    print(submission.head())


if __name__ == "__main__":
    main()
