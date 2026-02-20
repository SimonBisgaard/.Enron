from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from train_per_market_interactions import (
    apply_exclude_2023 as apply_exclude_2023_base,
    build_feature_table as build_feature_table_base,
)
from train_per_market_interactions_hour_experts import (
    build_feature_table as build_feature_table_hour,
)


def _predict_point(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    pred = np.asarray(model.predict(X))
    if pred.ndim == 2:
        return pred[:, 0].astype(float)
    return pred.astype(float)


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


def _call_build_feature_table(
    build_fn,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_args: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig = inspect.signature(build_fn)
    allowed = {k: v for k, v in train_args.items() if k in sig.parameters}
    return build_fn(train_df, test_df, **allowed)


def _ensure_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    missing = [c for c in required_cols if c not in out.columns]
    if not missing:
        return out, missing

    cat_set = set(cat_cols)
    for col in missing:
        if col in cat_set:
            out[col] = "__MISSING__"
        elif col.endswith("_count"):
            out[col] = 0.0
        else:
            out[col] = np.nan
    return out, missing


def _build_features_for_run(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_args: dict[str, object],
    required_cols: list[str],
) -> tuple[pd.DataFrame, str]:
    # Newer runs with robust profiles/hour-expert pipeline require the hour-experts builder.
    needs_hour_builder = (
        bool(train_args.get("use_robust_target_profiles", False))
        or any(c.endswith("_count") and c.startswith("target_profile_") for c in required_cols)
    )

    builders: list[tuple[str, object]] = []
    if needs_hour_builder:
        builders.append(("train_per_market_interactions_hour_experts", build_feature_table_hour))
        builders.append(("train_per_market_interactions", build_feature_table_base))
    else:
        builders.append(("train_per_market_interactions", build_feature_table_base))
        builders.append(("train_per_market_interactions_hour_experts", build_feature_table_hour))

    best_name = ""
    best_df: pd.DataFrame | None = None
    best_overlap = -1

    for name, fn in builders:
        tr_feat, _ = _call_build_feature_table(fn, train_df, test_df, train_args)
        overlap = len([c for c in required_cols if c in tr_feat.columns])
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
            best_df = tr_feat
        if overlap == len(required_cols):
            return tr_feat, name

    if best_df is None:
        raise RuntimeError("Failed to build feature table for SHAP export.")
    return best_df, best_name


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
        train_df = apply_exclude_2023_base(train_df, keep_from_month=keep_month)

    models_dir = run_dir / "models"
    global_model = CatBoostRegressor()
    global_model.load_model(models_dir / meta["global_model"])

    global_feature_cols: list[str] = meta["candidate_features_before_global_pred"]
    local_feature_cols: list[str] = meta["feature_cols"]
    cat_cols: list[str] = meta["cat_cols"]

    required_cols = sorted(set(global_feature_cols) | set(local_feature_cols))
    train_feat, builder_name = _build_features_for_run(train_df, test_df, train_args, required_cols)
    train_feat, missing_any = _ensure_required_columns(train_feat, required_cols, cat_cols)
    if missing_any:
        print(
            "Warning: SHAP feature reconstruction missing columns; "
            f"filled defaults for {len(missing_any)} columns."
        )
        print(f"Filled: {missing_any}")
    print(f"SHAP feature builder: {builder_name}")

    train_local = train_feat.copy()
    train_local["global_pred_feature"] = _predict_point(global_model, train_feat[global_feature_cols])
    train_local, missing_local = _ensure_required_columns(train_local, local_feature_cols, cat_cols)
    if missing_local:
        print(
            "Warning: local feature alignment required additional default-filled columns: "
            f"{missing_local}"
        )

    global_sample = _sample_df(train_feat, args.global_sample_size, args.seed)
    global_pool = Pool(
        data=global_sample[global_feature_cols],
        cat_features=_cat_indices(global_feature_cols, cat_cols),
    )
    shap_global = global_model.get_feature_importance(global_pool, type="ShapValues")
    shap_global_vals = shap_global[:, :-1]
    shap_global_base = shap_global[:, -1]
    global_preds = _predict_point(global_model, global_sample[global_feature_cols])

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
        "feature_builder": builder_name,
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
