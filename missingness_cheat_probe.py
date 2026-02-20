from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    start = pd.to_datetime(out["delivery_start"], errors="coerce")
    out["hour"] = start.dt.hour
    out["dow"] = start.dt.dayofweek
    out["month"] = start.dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    # Basic seasonality harmonics.
    out["hour_sin"] = np.sin(2.0 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2.0 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * out["dow"] / 7.0)
    out["month_sin"] = np.sin(2.0 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * out["month"] / 12.0)
    return out


def get_meteo_columns(train_df: pd.DataFrame) -> list[str]:
    core = {
        "id",
        "target",
        "market",
        "delivery_start",
        "delivery_end",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
    }
    return [c for c in train_df.columns if c not in core]


def build_simple_model(train_df: pd.DataFrame) -> tuple[CatBoostRegressor, list[str], list[str]]:
    feature_cols = [
        "market",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
        "hour",
        "dow",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]
    cat_cols = ["market"]

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=2000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=20,
        bagging_temperature=0.5,
        random_strength=0.9,
        random_seed=42,
        verbose=0,
    )
    model.fit(train_df[feature_cols], train_df["target"], cat_features=cat_cols)
    return model, feature_cols, cat_cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe missingness hypothesis with two submissions:\n"
            "1) simple forecast+time baseline\n"
            "2) same baseline, but add bump on test rows where meteo is missing."
        )
    )
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument(
        "--baseline-submission",
        default="submissio.csv",
        help="Existing baseline submission to impute into on missing-weather rows.",
    )
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="missingness_probe")
    parser.add_argument("--missing-std-multiplier", type=float, default=10.0)
    parser.add_argument(
        "--min-missing-cols-to-impute",
        type=int,
        default=8,
        help="Only treat test rows with at least this many missing meteo columns as missing-heavy.",
    )
    parser.add_argument(
        "--use-market-std",
        action="store_true",
        help="Use market-specific train target std for bump; default is global train std.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_path = Path(args.sample_submission)
    baseline_submission_path = Path(args.baseline_submission)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)
    baseline_submission = pd.read_csv(baseline_submission_path)

    meteo_cols = get_meteo_columns(train_df)
    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)

    model, feature_cols, _ = build_simple_model(train_df)
    base_pred = np.asarray(model.predict(test_df[feature_cols]), dtype=float)

    test_missing_count = test_df[meteo_cols].isna().sum(axis=1).to_numpy(dtype=int)
    test_missing_any = test_missing_count >= int(args.min_missing_cols_to_impute)
    if args.use_market_std:
        std_by_market = train_df.groupby("market")["target"].std().to_dict()
        per_row_std = test_df["market"].map(std_by_market).fillna(train_df["target"].std()).to_numpy(dtype=float)
        bump = float(args.missing_std_multiplier) * per_row_std
    else:
        global_std = float(train_df["target"].std())
        bump = np.full(len(test_df), float(args.missing_std_multiplier) * global_std, dtype=float)

    boosted_pred = base_pred.copy()
    boosted_pred[test_missing_any] = boosted_pred[test_missing_any] + bump[test_missing_any]

    run_dir = out_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_sub = sample[["id"]].copy()
    base_sub["target"] = base_sub["id"].map(pd.Series(base_pred, index=test_df["id"].to_numpy()))
    boosted_sub = sample[["id"]].copy()
    boosted_sub["target"] = boosted_sub["id"].map(pd.Series(boosted_pred, index=test_df["id"].to_numpy()))

    baseline_submission = baseline_submission.copy()
    if not {"id", "target"}.issubset(baseline_submission.columns):
        raise ValueError("Baseline submission must contain columns: id,target")
    baseline_submission["id"] = pd.to_numeric(baseline_submission["id"], errors="coerce")
    baseline_submission["target"] = pd.to_numeric(baseline_submission["target"], errors="coerce")
    baseline_submission = baseline_submission.dropna(subset=["id"]).copy()
    baseline_submission["id"] = baseline_submission["id"].astype(int)

    simple_map = pd.Series(base_pred, index=test_df["id"].astype(int).to_numpy())
    missing_ids = test_df.loc[test_missing_any, "id"].astype(int).to_numpy()
    imputed_sub = baseline_submission.copy()
    replace_mask = imputed_sub["id"].isin(missing_ids)
    imputed_sub.loc[replace_mask, "target"] = imputed_sub.loc[replace_mask, "id"].map(simple_map)

    base_path = run_dir / "submission_simple_baseline.csv"
    boosted_path = run_dir / "submission_simple_missing_boost.csv"
    imputed_path = run_dir / "submission_baseline_imputed_simple_on_missing.csv"
    base_sub.to_csv(base_path, index=False)
    boosted_sub.to_csv(boosted_path, index=False)
    imputed_sub.to_csv(imputed_path, index=False)

    print(f"Saved baseline submission: {base_path}")
    print(f"Saved missing-boosted submission: {boosted_path}")
    print(f"Saved baseline-imputed submission: {imputed_path}")
    print(
        "Missing-heavy rows in test: "
        f"{int(test_missing_any.sum())} / {len(test_missing_any)} ({test_missing_any.mean():.4%}) "
        f"with min_missing_cols_to_impute={args.min_missing_cols_to_impute}"
    )
    if args.use_market_std:
        print(f"Boost rule: +{args.missing_std_multiplier} * market_std(target_train) on missing rows")
    else:
        print(f"Boost rule: +{args.missing_std_multiplier} * global_std(target_train) on missing rows")


if __name__ == "__main__":
    main()
