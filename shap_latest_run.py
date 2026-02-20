from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from train_per_market_interactions import (
    apply_exclude_2023,
    build_feature_table,
    train_global_and_local_models,
)


@dataclass
class InferredConfig:
    exclude_2023: bool
    exclude_2023_keep_from_month: int
    source: str


def _latest_run_dir(runs_root: Path) -> Path:
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and "per_market_interactions" in p.name]
    if not candidates:
        raise FileNotFoundError("No per_market_interactions run directories found under runs/")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _infer_config_from_run(run_dir: Path, train_df: pd.DataFrame) -> InferredConfig:
    cv_path = run_dir / "cv_results.csv"
    if not cv_path.exists():
        return InferredConfig(
            exclude_2023=False,
            exclude_2023_keep_from_month=10,
            source="cv_results.csv missing; defaulted to no exclusion",
        )

    cv = pd.read_csv(cv_path)
    if cv.empty or not {"val_start", "train_rows"}.issubset(cv.columns):
        return InferredConfig(
            exclude_2023=False,
            exclude_2023_keep_from_month=10,
            source="cv_results.csv missing required columns; defaulted to no exclusion",
        )

    first = cv.sort_values("fold").iloc[0]
    val_start = pd.Timestamp(first["val_start"])
    observed_train_rows = int(first["train_rows"])

    start = pd.to_datetime(train_df["delivery_start"])
    no_excl = int((start < val_start).sum())
    excl_2023 = int(
        (
            (start < val_start)
            & ((start.dt.year != 2023) | (start.dt.month >= 10))
        ).sum()
    )

    if observed_train_rows == excl_2023:
        return InferredConfig(
            exclude_2023=True,
            exclude_2023_keep_from_month=10,
            source="matched cv train_rows against exclude-2023 keep-from-month=10",
        )
    if observed_train_rows == no_excl:
        return InferredConfig(
            exclude_2023=False,
            exclude_2023_keep_from_month=10,
            source="matched cv train_rows against no exclusion",
        )
    return InferredConfig(
        exclude_2023=False,
        exclude_2023_keep_from_month=10,
        source=(
            "could not match cv train_rows to known filter settings; "
            "defaulted to no exclusion"
        ),
    )


def _cat_indices(feature_cols: list[str], cat_cols: list[str]) -> list[int]:
    cat_set = set(cat_cols)
    return [i for i, c in enumerate(feature_cols) if c in cat_set]


def _predict_point(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(model.predict(X))
    if pred.ndim == 2:
        return pred[:, 0].astype(float)
    return pred.astype(float)


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SHAP outputs for latest per_market_interactions run.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--global-sample-size", type=int, default=1500)
    parser.add_argument("--per-market-sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(runs_root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    cfg = _infer_config_from_run(run_dir, train_df)
    if cfg.exclude_2023:
        train_df = apply_exclude_2023(train_df, keep_from_month=cfg.exclude_2023_keep_from_month)

    train_feat, test_feat = build_feature_table(train_df, test_df)
    del test_feat

    base_drop = {"id", "target", "delivery_start", "delivery_end"}
    global_feature_cols = [c for c in train_feat.columns if c not in base_drop]
    cat_cols = [
        c
        for c in ["market", "hour_x_market", "dow_x_market", "month_x_market"]
        if c in global_feature_cols
    ]

    artifacts = train_global_and_local_models(
        train_feat,
        global_feature_cols,
        cat_cols,
        local_residual_modeling=True,
        include_global_pred_in_local=False,
    )

    train_local = train_feat.copy()
    train_local["global_pred_feature"] = _predict_point(artifacts.global_model, train_local[global_feature_cols])

    global_sample = _sample_df(train_feat, args.global_sample_size, args.seed)
    global_pool = Pool(
        data=global_sample[global_feature_cols],
        cat_features=_cat_indices(global_feature_cols, cat_cols),
    )
    global_shap = artifacts.global_model.get_feature_importance(global_pool, type="ShapValues")
    global_shap_values = global_shap[:, :-1]
    global_base_value = global_shap[:, -1]
    global_preds = _predict_point(artifacts.global_model, global_sample[global_feature_cols])

    global_imp = pd.DataFrame(
        {
            "feature": global_feature_cols,
            "mean_abs_shap": np.abs(global_shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    global_imp.to_csv(run_dir / "global_feature_importance_shap.csv", index=False)

    global_sample_out = global_sample[["id", "market", "delivery_start", "target"]].reset_index(drop=True)
    global_sample_out = global_sample_out.assign(base_value=global_base_value, prediction=global_preds)
    shap_cols = pd.DataFrame(
        global_shap_values,
        columns=[f"shap__{feat}" for feat in global_feature_cols],
    )
    global_sample_out = pd.concat([global_sample_out, shap_cols], axis=1)
    global_sample_out.to_csv(run_dir / "global_shap_sample_rows.csv", index=False)

    local_rows: list[dict[str, object]] = []
    local_feature_cols = artifacts.feature_cols
    local_cat_idx = _cat_indices(local_feature_cols, artifacts.cat_cols)

    for market, model in artifacts.local_models.items():
        mdf = train_local.loc[train_local["market"].astype(str) == str(market)].copy()
        if mdf.empty:
            continue
        ms = _sample_df(mdf, args.per_market_sample_size, args.seed)
        pool = Pool(data=ms[local_feature_cols], cat_features=local_cat_idx)
        shap_vals = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        imp = np.abs(shap_vals).mean(axis=0)
        for feat, score in zip(local_feature_cols, imp):
            local_rows.append(
                {
                    "market": str(market),
                    "feature": feat,
                    "mean_abs_shap": float(score),
                }
            )

    pd.DataFrame(local_rows).sort_values(
        ["market", "mean_abs_shap"], ascending=[True, False]
    ).to_csv(run_dir / "local_feature_importance_shap.csv", index=False)

    meta = {
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "rows_train_after_filter": int(len(train_df)),
        "rows_feature_table": int(len(train_feat)),
        "global_feature_count": int(len(global_feature_cols)),
        "global_sample_size_used": int(len(global_sample)),
        "per_market_sample_size_requested": int(args.per_market_sample_size),
        "markets": sorted([str(m) for m in artifacts.local_models.keys()]),
    }
    (run_dir / "shap_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"Run dir: {run_dir}")
    print(f"Inferred config: {cfg}")
    print("Saved:")
    print(f"- {run_dir / 'global_feature_importance_shap.csv'}")
    print(f"- {run_dir / 'local_feature_importance_shap.csv'}")
    print(f"- {run_dir / 'global_shap_sample_rows.csv'}")
    print(f"- {run_dir / 'shap_metadata.json'}")


if __name__ == "__main__":
    main()
