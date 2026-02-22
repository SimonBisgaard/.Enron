from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor

from train_per_market_interactions_commit_baseline import (
    apply_exclude_2023,
    apply_train_start_cutoff,
    build_feature_table,
    make_time_series_folds,
)
from train_per_market_interactions_multimodel import (
    FamilyBundle,
    _format_duration,
    _fit_model,
    _make_model,
    _parse_model_families,
    _predict_model,
    _predict_multifamily,
    _resolve_optional_families,
    _rmse,
    _save_models,
    _sha256_file,
)

# Strict "legacy" profile (commit-baseline equivalent, no newer feature additions).
LEGACY_FEATURE_FLAGS = {
    "use_dynamic_target_profiles": False,
    "use_temporal_regime": False,
    "use_volatility_regime": False,
    "use_wind_proxy": False,
    "use_ws80_turbine_features": False,
    "use_anomaly_features": False,
    "use_cross_market_rank_features": False,
    "use_peak_interactions": False,
}


def _make_2c02_catboost(*, local: bool) -> CatBoostRegressor:
    if local:
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


def _train_multifamily_global_local_legacy_2c02(
    train_feat: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    families: list[str],
    seed: int,
) -> dict[str, FamilyBundle]:
    bundles: dict[str, FamilyBundle] = {}
    global_preds: dict[str, Any] = {}

    for i, family in enumerate(families):
        if family == "catboost":
            model = _make_2c02_catboost(local=False)
        else:
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
            local_feature_cols=feature_cols + ["global_pred_feature"],
            cat_cols=cat_cols,
        )

    global_stack = pd.DataFrame({f: global_preds[f] for f in families})
    work = train_feat.copy()
    work["global_pred_feature"] = global_stack.mean(axis=1).to_numpy(dtype=float)

    for family in families:
        local_seed_base = seed + 1000 + families.index(family) * 100
        for j, (market, mdf) in enumerate(work.groupby("market", dropna=False)):
            if family == "catboost":
                local_model = _make_2c02_catboost(local=True)
            else:
                local_model = _make_model(
                    family,
                    feature_cols=bundles[family].local_feature_cols,
                    cat_cols=cat_cols,
                    random_seed=local_seed_base + j,
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


def run_time_series_cv(
    train_df_raw: pd.DataFrame,
    *,
    n_folds: int,
    val_days: int,
    step_days: int,
    min_train_days: int,
    families: list[str],
    seed: int,
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
            **LEGACY_FEATURE_FLAGS,
        )

        base_drop = {"id", "target", "delivery_start", "delivery_end"}
        feat_cols = [c for c in tr_feat.columns if c not in base_drop]
        cat_cols = [
            c
            for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
            if c in feat_cols
        ]

        bundles = _train_multifamily_global_local_legacy_2c02(
            train_feat=tr_feat,
            feature_cols=feat_cols,
            cat_cols=cat_cols,
            families=families,
            seed=seed + fold_idx * 10000,
        )

        va_keys = va_feat[["id", "market"]].copy()
        pred = _predict_multifamily(va_feat, va_keys, bundles, families)
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


def main() -> None:
    overall_start = time.perf_counter()
    parser = argparse.ArgumentParser(
        description="Legacy commit-baseline per-market training with multi-model family ensemble."
    )
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="per_market_interactions_commit_baseline_multimodel")
    parser.add_argument("--exclude-2023", action="store_true")
    parser.add_argument("--exclude-2023-keep-from-month", type=int, default=10)
    parser.add_argument(
        "--train-start-july-2024",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict training rows to delivery_start >= 2024-07-01.",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    families = _parse_model_families(args.models)
    families = _resolve_optional_families(families, allow_missing=args.allow_missing_models)
    print(f"Using model families: {families}")
    print("Using legacy feature profile: strict commit baseline (2c02eb6-equivalent).")

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_path = Path(args.sample_submission)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample = pd.read_csv(sample_path)

    if args.exclude_2023:
        train_df = apply_exclude_2023(train_df, keep_from_month=args.exclude_2023_keep_from_month)
    if args.train_start_july_2024:
        train_df = apply_train_start_cutoff(train_df, start_date="2024-07-01")

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
            families=families,
            seed=args.seed,
        )
        print(f"CV phase elapsed: {_format_duration(time.perf_counter() - cv_phase_start)}")

    feature_phase_start = time.perf_counter()
    train_feat, test_feat = build_feature_table(
        train_df,
        test_df,
        **LEGACY_FEATURE_FLAGS,
    )
    base_drop = {"id", "target", "delivery_start", "delivery_end"}
    candidate_features = [c for c in train_feat.columns if c not in base_drop]
    print(f"Feature prep elapsed: {_format_duration(time.perf_counter() - feature_phase_start)}")

    cat_cols = [
        c
        for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
        if c in candidate_features
    ]

    train_phase_start = time.perf_counter()
    bundles = _train_multifamily_global_local_legacy_2c02(
        train_feat=train_feat,
        feature_cols=candidate_features,
        cat_cols=cat_cols,
        families=families,
        seed=args.seed,
    )
    print(f"Main train elapsed: {_format_duration(time.perf_counter() - train_phase_start)}")

    pred_phase_start = time.perf_counter()
    test_keys = test_feat[["id", "market"]].copy()
    pred = _predict_multifamily(test_feat, test_keys, bundles, families)

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

    latest_path = Path("csv/submission_per_market_interactions_commit_baseline_multimodel.csv")
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

    model_metadata = {
        "model_families": families,
        "feature_profile": "legacy_commit_baseline_2c02eb6",
        "legacy_feature_flags": LEGACY_FEATURE_FLAGS,
        "candidate_features_before_global_pred": candidate_features,
        "cat_cols": cat_cols,
        "cv_rmse": cv_rmse,
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
        "feature_profile": "legacy_commit_baseline_2c02eb6",
        "features": {
            "candidate_features_before_global_pred": candidate_features,
            "cat_cols": cat_cols,
            "legacy_feature_flags": LEGACY_FEATURE_FLAGS,
        },
        "metrics": {
            "cv_rmse": cv_rmse,
        },
        "model_families": families,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Predict+save elapsed: {_format_duration(time.perf_counter() - pred_phase_start)}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved latest copy: {latest_path}")
    print(f"Model families: {families}")
    print(f"Features used: {len(candidate_features)}")
    print(f"Categorical features: {cat_cols}")
    print(f"CV RMSE: {cv_rmse}")
    print(f"Saved metadata: {run_dir / 'model_metadata.json'}")
    print(f"Total wall time: {_format_duration(time.perf_counter() - overall_start)}")


if __name__ == "__main__":
    main()
