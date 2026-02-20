from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from train_per_market_interactions import apply_exclude_2023, build_feature_table


def _latest_saved_run(runs_root: Path) -> Path:
    candidates = [p for p in runs_root.iterdir() if (p / "model_metadata.json").exists()]
    if not candidates:
        raise FileNotFoundError("No runs with model_metadata.json found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _cat_indices(feature_cols: list[str], cat_cols: list[str]) -> list[int]:
    cset = set(cat_cols)
    return [i for i, c in enumerate(feature_cols) if c in cset]


def _sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SHAP values from saved per_market_interactions models.")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--global-sample-size", type=int, default=1500)
    parser.add_argument("--per-market-sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = Path(args.run_dir) if args.run_dir else _latest_saved_run(runs_root)
    meta_path = run_dir / "model_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")
    meta = json.loads(meta_path.read_text())

    train_args = meta.get("train_args", {})
    train_path = train_args.get("train_path", "data/train.csv")
    test_path = train_args.get("test_path", "data/test_for_participants.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if bool(train_args.get("exclude_2023", False)):
        keep_month = int(train_args.get("exclude_2023_keep_from_month", 10))
        train_df = apply_exclude_2023(train_df, keep_from_month=keep_month)

    train_feat, _ = build_feature_table(train_df, test_df)

    models_dir = run_dir / "models"
    global_model = CatBoostRegressor()
    global_model.load_model(models_dir / meta["global_model"])

    global_feature_cols: list[str] = meta["candidate_features_before_global_pred"]
    local_feature_cols: list[str] = meta["feature_cols"]
    cat_cols: list[str] = meta["cat_cols"]

    train_local = train_feat.copy()
    train_local["global_pred_feature"] = global_model.predict(train_feat[global_feature_cols])

    global_sample = _sample_df(train_feat, args.global_sample_size, args.seed)
    global_pool = Pool(
        data=global_sample[global_feature_cols],
        cat_features=_cat_indices(global_feature_cols, cat_cols),
    )
    shap_global = global_model.get_feature_importance(global_pool, type="ShapValues")
    shap_global_vals = shap_global[:, :-1]
    shap_global_base = shap_global[:, -1]
    global_preds = global_model.predict(global_sample[global_feature_cols])

    global_imp = pd.DataFrame(
        {"feature": global_feature_cols, "mean_abs_shap": np.abs(shap_global_vals).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    global_imp.to_csv(run_dir / "global_feature_importance_shap.csv", index=False)

    global_rows = global_sample[["id", "market", "delivery_start", "target"]].reset_index(drop=True)
    global_rows["base_value"] = shap_global_base
    global_rows["prediction"] = global_preds
    for i, feat in enumerate(global_feature_cols):
        global_rows[f"shap__{feat}"] = shap_global_vals[:, i]
    global_rows.to_csv(run_dir / "global_shap_sample_rows.csv", index=False)

    local_rows: list[dict[str, object]] = []
    local_cat_idx = _cat_indices(local_feature_cols, cat_cols)
    for market, model_file in meta["local_models"].items():
        model = CatBoostRegressor()
        model.load_model(models_dir / model_file)

        mdf = train_local.loc[train_local["market"].astype(str) == str(market)].copy()
        if mdf.empty:
            continue
        ms = _sample_df(mdf, args.per_market_sample_size, args.seed)
        pool = Pool(data=ms[local_feature_cols], cat_features=local_cat_idx)
        shap_local = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
        imp = np.abs(shap_local).mean(axis=0)
        for feat, score in zip(local_feature_cols, imp):
            local_rows.append({"market": str(market), "feature": feat, "mean_abs_shap": float(score)})

    pd.DataFrame(local_rows).sort_values(
        ["market", "mean_abs_shap"], ascending=[True, False]
    ).to_csv(run_dir / "local_feature_importance_shap.csv", index=False)

    out_meta = {
        "run_dir": str(run_dir),
        "global_sample_size_used": int(len(global_sample)),
        "per_market_sample_size_requested": int(args.per_market_sample_size),
        "rows_train_after_filter": int(len(train_df)),
        "global_feature_count": int(len(global_feature_cols)),
        "local_feature_count": int(len(local_feature_cols)),
    }
    (run_dir / "shap_metadata.json").write_text(json.dumps(out_meta, indent=2))

    print(f"Run dir: {run_dir}")
    print("Saved:")
    print(f"- {run_dir / 'global_feature_importance_shap.csv'}")
    print(f"- {run_dir / 'local_feature_importance_shap.csv'}")
    print(f"- {run_dir / 'global_shap_sample_rows.csv'}")
    print(f"- {run_dir / 'shap_metadata.json'}")


if __name__ == "__main__":
    main()
