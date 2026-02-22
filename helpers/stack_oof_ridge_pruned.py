from __future__ import annotations

import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    test_submission_path: Path
    oof_csv_path: Path | None = None
    oof_zip_path: Path | None = None
    oof_v1_member: str | None = None
    oof_v2_member: str | None = None
    best_config_member: str | None = None


def _parse_lambdas(raw: str) -> list[float]:
    vals = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        raise ValueError("At least one lambda value is required.")
    return vals


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _load_submission(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} submission not found: {path}")
    df = pd.read_csv(path, usecols=["id", "target"])
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["id", "target"]).copy()
    df["id"] = df["id"].astype(int)
    return df.groupby("id", as_index=False)["target"].mean()


def _read_parquet_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"Zip member not found: {member} in {zip_path}")
        payload = zf.read(member)
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise RuntimeError(
            "Failed to read parquet payload. Install `pyarrow` in the active environment, "
            "or run this script with an environment that has parquet support."
        ) from exc


def _read_json_from_zip(zip_path: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(zip_path) as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"Zip member not found: {member} in {zip_path}")
        return json.loads(zf.read(member))


def _weights_from_config(df: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    by_regime = {
        "peak_scarcity": float(cfg.get("w_peak", 0.0)),
        "transition": float(cfg.get("w_transition", 0.0)),
        "normal": float(cfg.get("w_normal", 0.0)),
        "lowprice_midday": float(cfg.get("w_lowprice", 0.0)),
    }
    w = df["primary_regime"].map(by_regime).fillna(by_regime["normal"]).to_numpy(dtype=float)
    if int(float(cfg.get("force_market_a_normal_nonpeak_zero", 0))) == 1:
        force_mask = (
            (df["market"].to_numpy() == "Market A")
            & (df["primary_regime"].to_numpy() == "normal")
            & (~df["hour"].between(17, 20).to_numpy())
        )
        w = np.where(force_mask, 0.0, w)
    return np.clip(w, 0.0, 1.0)


def _pick_pred_col(df: pd.DataFrame) -> str:
    for col in ("pred_final_best_oof", "pred_final_oof", "pred"):
        if col in df.columns:
            return col
    raise ValueError("Could not find OOF prediction column in dataframe.")


def _load_jens_blend_oof(spec: ModelSpec) -> pd.DataFrame:
    assert spec.oof_zip_path is not None
    assert spec.oof_v1_member is not None
    assert spec.oof_v2_member is not None
    assert spec.best_config_member is not None

    oof1 = _read_parquet_from_zip(spec.oof_zip_path, spec.oof_v1_member)
    oof2 = _read_parquet_from_zip(spec.oof_zip_path, spec.oof_v2_member)
    cfg_raw = _read_json_from_zip(spec.oof_zip_path, spec.best_config_member)
    best_cfg = cfg_raw.get("best_config", cfg_raw)
    if not isinstance(best_cfg, dict):
        raise ValueError(f"Invalid best config payload for {spec.name}")

    p1_col = _pick_pred_col(oof1)
    p2_col = _pick_pred_col(oof2)
    req1 = {"id", "y", "market", "ts"}
    req2 = {"id", p2_col}
    miss1 = sorted(req1 - set(oof1.columns))
    miss2 = sorted(req2 - set(oof2.columns))
    if miss1:
        raise ValueError(f"{spec.name} v1 OOF missing columns: {miss1}")
    if miss2:
        raise ValueError(f"{spec.name} v2 OOF missing columns: {miss2}")

    r1 = oof1[["id", "y", "market", "ts", "fold_id", "primary_regime", p1_col]].copy()
    r2 = oof2[["id", p2_col]].copy()
    r1 = r1.rename(columns={p1_col: "p1", "y": "target", "ts": "delivery_start", "fold_id": "fold"})
    r2 = r2.rename(columns={p2_col: "p2"})
    merged = r1.merge(r2, on="id", how="inner")

    merged["delivery_start"] = pd.to_datetime(merged["delivery_start"], errors="coerce")
    if merged["delivery_start"].isna().any():
        raise ValueError(f"{spec.name} contains invalid timestamps in OOF.")
    merged["hour"] = merged["delivery_start"].dt.hour.astype(int)

    w = _weights_from_config(merged, best_cfg)
    pred = (1.0 - w) * merged["p1"].to_numpy(dtype=float) + w * merged["p2"].to_numpy(dtype=float)
    out = merged[["id", "delivery_start", "target", "fold"]].copy()
    out["pred"] = pred
    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    out = out.dropna(subset=["id", "target", "pred", "delivery_start"]).copy()
    out["id"] = out["id"].astype(int)
    return out


def _load_cv_oof(spec: ModelSpec) -> pd.DataFrame:
    assert spec.oof_csv_path is not None
    if not spec.oof_csv_path.exists():
        raise FileNotFoundError(f"OOF csv missing for {spec.name}: {spec.oof_csv_path}")
    df = pd.read_csv(spec.oof_csv_path)
    required = {"id", "delivery_start", "target", "pred"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{spec.name} OOF missing required columns: {missing}")
    out = df[["id", "delivery_start", "target", "pred", "fold"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype("Int64")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out["pred"] = pd.to_numeric(out["pred"], errors="coerce")
    out["delivery_start"] = pd.to_datetime(out["delivery_start"], errors="coerce")
    out = out.dropna(subset=["id", "target", "pred", "delivery_start"]).copy()
    out["id"] = out["id"].astype(int)
    return out


def _fit_nonnegative_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    # Closed-form ridge without intercept: (X^T X + alpha I) w = X^T y
    xtx = x.T @ x
    if alpha > 0:
        xtx = xtx + alpha * np.eye(xtx.shape[0], dtype=float)
    xty = x.T @ y
    try:
        w = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(xtx) @ xty
    w = np.asarray(w, dtype=float).reshape(-1)
    w = np.clip(w, 0.0, None)
    s = float(np.sum(w))
    if s <= 0:
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / s
    return w


def _build_time_folds(ts: pd.Series, n_folds: int, min_train_days: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_folds <= 0:
        return []
    day = pd.to_datetime(ts).dt.floor("D")
    uniq_days = np.array(sorted(day.dropna().unique()))
    if len(uniq_days) < (min_train_days + n_folds):
        return []

    val_days = max(1, len(uniq_days) // (n_folds + 1))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        train_end = min_train_days + k * val_days
        if train_end >= len(uniq_days):
            break
        val_end = min(train_end + val_days, len(uniq_days))
        train_set = set(uniq_days[:train_end])
        val_set = set(uniq_days[train_end:val_end])
        if not val_set:
            continue
        train_idx = np.where(day.isin(train_set).to_numpy())[0]
        val_idx = np.where(day.isin(val_set).to_numpy())[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))
    return folds


def _evaluate_lambdas(
    x: np.ndarray,
    y: np.ndarray,
    ts: pd.Series,
    lambdas: list[float],
    cv_folds: int,
    min_train_days: int,
) -> tuple[float, pd.DataFrame]:
    folds = _build_time_folds(ts, cv_folds, min_train_days)
    rows: list[dict[str, Any]] = []

    if not folds:
        for alpha in lambdas:
            w = _fit_nonnegative_ridge(x, y, alpha=alpha)
            rmse = _rmse(y, x @ w)
            rows.append(
                {
                    "lambda": alpha,
                    "rmse_cv": rmse,
                    "mode": "full_oof_no_time_folds",
                    "n_folds": 0,
                }
            )
        scores = pd.DataFrame(rows).sort_values(["rmse_cv", "lambda"]).reset_index(drop=True)
        return float(scores.iloc[0]["lambda"]), scores

    for alpha in lambdas:
        rmses = []
        ns = []
        for train_idx, val_idx in folds:
            w = _fit_nonnegative_ridge(x[train_idx], y[train_idx], alpha=alpha)
            pred_val = x[val_idx] @ w
            rmses.append(_rmse(y[val_idx], pred_val))
            ns.append(len(val_idx))
        score = float(np.average(rmses, weights=ns))
        rows.append(
            {
                "lambda": alpha,
                "rmse_cv": score,
                "mode": "time_walk_forward",
                "n_folds": len(folds),
            }
        )
    scores = pd.DataFrame(rows).sort_values(["rmse_cv", "lambda"]).reset_index(drop=True)
    return float(scores.iloc[0]["lambda"]), scores


def _prune_by_corr(
    df: pd.DataFrame,
    model_cols: list[str],
    y_col: str,
    threshold: float,
) -> tuple[list[str], list[dict[str, Any]], dict[str, float], pd.DataFrame]:
    active = list(model_cols)
    drops: list[dict[str, Any]] = []

    rmse_map = {m: _rmse(df[y_col].to_numpy(), df[m].to_numpy()) for m in model_cols}

    while True:
        corr = df[active].corr().abs()
        if corr.empty or len(active) <= 1:
            break
        corr_arr = corr.to_numpy().copy()
        np.fill_diagonal(corr_arr, 0.0)
        max_corr = float(corr_arr.max())
        if max_corr <= threshold:
            break
        i, j = np.unravel_index(np.argmax(corr_arr), corr.shape)
        a = corr.index[i]
        b = corr.columns[j]
        if a == b:
            break
        drop = a if rmse_map[a] > rmse_map[b] else b
        keep = b if drop == a else a
        drops.append(
            {
                "dropped": drop,
                "kept": keep,
                "pair_corr": max_corr,
                "rmse_dropped": rmse_map[drop],
                "rmse_kept": rmse_map[keep],
                "reason": f"abs_corr>{threshold}",
            }
        )
        active.remove(drop)

    final_corr = df[active].corr()
    return active, drops, rmse_map, final_corr


def _default_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="20260221-215400_jens",
            kind="jens_blend_zip",
            test_submission_path=Path("runs/20260221-215400_jens/submission.csv"),
            oof_zip_path=Path("Nitor konkurrence - Kopi (3).zip"),
            oof_v1_member="Nitor konkurrence - Kopi (3)/runs/market_layer/20260221_152543_market_layer_v1/oof_market_layer.parquet",
            oof_v2_member="Nitor konkurrence - Kopi (3)/runs/market_layer/20260221_192226_v2_robustxmk/oof_market_layer.parquet",
            best_config_member="Nitor konkurrence - Kopi (3)/runs/v3_blend/20260221_213508_recheck/repro_bundle/artifacts/best_config.json",
        ),
        ModelSpec(
            name="20260222-175700_jens",
            kind="jens_blend_zip",
            test_submission_path=Path("runs/20260222-175700_jens/submission.csv"),
            oof_zip_path=Path("Nitor konkurrence.zip"),
            oof_v1_member="Nitor konkurrence/runs/market_layer_xgb/20260222_150523_submission3_xgb_kickoff_xgb_v1/oof_market_layer.parquet",
            oof_v2_member="Nitor konkurrence/runs/market_layer_xgb/20260222_152031_submission3_xgb_kickoff_xgb_v2/oof_market_layer.parquet",
            best_config_member="Nitor konkurrence/runs/v3_blend_xgb/20260222_160520_submission3_xgb_kickoff_blend/best_config.json",
        ),
        ModelSpec(
            name="20260221-172305_per_market_interactions_2c02eb6_oct2023_cv1_14d",
            kind="cv_oof_csv",
            test_submission_path=Path(
                "runs/20260221-172305_per_market_interactions_2c02eb6_oct2023_cv1_14d/submission.csv"
            ),
            oof_csv_path=Path(
                "runs/20260221-172305_per_market_interactions_2c02eb6_oct2023_cv1_14d/cv_oof.csv"
            ),
        ),
        ModelSpec(
            name="20260221-201434_per_market_interactions_2c02eb6_oct2023_cv1_14d_temp_physics_pruned_nores",
            kind="cv_oof_csv",
            test_submission_path=Path(
                "runs/20260221-201434_per_market_interactions_2c02eb6_oct2023_cv1_14d_temp_physics_pruned_nores/submission.csv"
            ),
            oof_csv_path=Path(
                "runs/20260221-201434_per_market_interactions_2c02eb6_oct2023_cv1_14d_temp_physics_pruned_nores/cv_oof.csv"
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold-safe non-negative ridge stacking on OOF with correlation pruning."
    )
    parser.add_argument("--name", default="final_stack_ridge_pruned")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--latest-copy", default="csv/submission_final_stack_ridge_pruned.csv")
    parser.add_argument("--corr-threshold", type=float, default=0.995)
    parser.add_argument("--lambdas", default="0.0,0.1,1,10,100")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--min-train-days", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_specs = _default_model_specs()
    lambdas = _parse_lambdas(args.lambdas)

    oof_frames: dict[str, pd.DataFrame] = {}
    test_frames: dict[str, pd.DataFrame] = {}
    for spec in model_specs:
        if spec.kind == "jens_blend_zip":
            oof = _load_jens_blend_oof(spec)
        elif spec.kind == "cv_oof_csv":
            oof = _load_cv_oof(spec)
        else:
            raise ValueError(f"Unknown model kind: {spec.kind}")
        oof_frames[spec.name] = oof
        test_frames[spec.name] = _load_submission(spec.test_submission_path, spec.name)

    merged = None
    y_cols = []
    ts_cols = []
    for name, df in oof_frames.items():
        part = df[["id", "target", "delivery_start", "pred"]].copy()
        part = part.rename(
            columns={
                "target": f"target__{name}",
                "delivery_start": f"ts__{name}",
                "pred": name,
            }
        )
        merged = part if merged is None else merged.merge(part, on="id", how="inner")
        y_cols.append(f"target__{name}")
        ts_cols.append(f"ts__{name}")

    if merged is None or merged.empty:
        raise RuntimeError("No overlapping OOF ids across models.")

    y_base = y_cols[0]
    ts_base = ts_cols[0]
    merged["target"] = merged[y_base]
    merged["delivery_start"] = pd.to_datetime(merged[ts_base], errors="coerce")

    y_mismatch = {}
    for c in y_cols[1:]:
        y_mismatch[c] = float(np.nanmax(np.abs(merged[c].to_numpy() - merged[y_base].to_numpy())))
    ts_mismatch = {}
    for c in ts_cols[1:]:
        lhs = pd.to_datetime(merged[c], errors="coerce")
        rhs = pd.to_datetime(merged[ts_base], errors="coerce")
        ts_mismatch[c] = int((lhs != rhs).sum())

    model_cols = [m.name for m in model_specs]
    active_cols, drops, rmse_map, final_corr = _prune_by_corr(
        merged, model_cols=model_cols, y_col="target", threshold=args.corr_threshold
    )

    if len(active_cols) == 0:
        raise RuntimeError("All models were pruned; cannot continue.")

    x = merged[active_cols].to_numpy(dtype=float)
    y = merged["target"].to_numpy(dtype=float)
    best_lambda, lambda_table = _evaluate_lambdas(
        x=x,
        y=y,
        ts=merged["delivery_start"],
        lambdas=lambdas,
        cv_folds=args.cv_folds,
        min_train_days=args.min_train_days,
    )
    weights = _fit_nonnegative_ridge(x, y, alpha=best_lambda)
    oof_pred = x @ weights
    oof_rmse = _rmse(y, oof_pred)

    sample = pd.read_csv(args.sample_submission, usecols=["id"])
    sample["id"] = pd.to_numeric(sample["id"], errors="coerce")
    sample = sample.dropna(subset=["id"]).copy()
    sample["id"] = sample["id"].astype(int)

    test_merged = sample.copy()
    for model_name in active_cols:
        part = test_frames[model_name].rename(columns={"target": model_name})
        test_merged = test_merged.merge(part, on="id", how="left")
    if test_merged[active_cols].isna().any().any():
        missing = test_merged[active_cols].isna().sum()
        raise RuntimeError(f"Missing test predictions after merge: {missing[missing > 0].to_dict()}")

    test_pred = test_merged[active_cols].to_numpy(dtype=float) @ weights
    submission = test_merged[["id"]].copy()
    submission["target"] = test_pred

    print(f"OOF overlap rows: {len(merged)}")
    print(f"Models before prune: {model_cols}")
    print(f"Models after prune: {active_cols}")
    print(f"Dropped models: {[d['dropped'] for d in drops]}")
    print(f"Chosen lambda: {best_lambda}")
    print(f"OOF RMSE (stack): {oof_rmse:.6f}")
    print("Weights:")
    for name, w in zip(active_cols, weights):
        print(f"  - {name}: {w:.6f}")

    if args.dry_run:
        print("Dry run complete. No files written.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{args.name}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    submission_path = run_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    lambda_table.to_csv(run_dir / "lambda_scores.csv", index=False)
    pd.DataFrame(
        {
            "model": active_cols,
            "weight": weights,
            "oof_rmse_model": [rmse_map[m] for m in active_cols],
        }
    ).to_csv(run_dir / "weights.csv", index=False)
    if drops:
        pd.DataFrame(drops).to_csv(run_dir / "pruned_models.csv", index=False)

    oof_diag = merged[["id", "delivery_start", "target"] + model_cols].copy()
    oof_diag["stack_pred"] = oof_pred
    oof_diag.to_csv(run_dir / "oof_stack_predictions.csv", index=False)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "models_requested": model_cols,
        "models_active": active_cols,
        "dropped_models": drops,
        "corr_threshold": args.corr_threshold,
        "lambdas": lambdas,
        "selected_lambda": best_lambda,
        "oof_rows_intersection": int(len(merged)),
        "oof_rmse_stack": oof_rmse,
        "weights": {name: float(w) for name, w in zip(active_cols, weights)},
        "oof_rmse_per_model": rmse_map,
        "target_mismatch_max_abs": y_mismatch,
        "timestamp_mismatch_rows": ts_mismatch,
        "input_specs": [
            {
                "name": s.name,
                "kind": s.kind,
                "test_submission_path": str(s.test_submission_path),
                "oof_csv_path": str(s.oof_csv_path) if s.oof_csv_path is not None else None,
                "oof_zip_path": str(s.oof_zip_path) if s.oof_zip_path is not None else None,
                "oof_v1_member": s.oof_v1_member,
                "oof_v2_member": s.oof_v2_member,
                "best_config_member": s.best_config_member,
            }
            for s in model_specs
        ],
    }
    (run_dir / "ensemble_config.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_corr.to_csv(run_dir / "oof_corr_active_models.csv")

    if args.latest_copy:
        latest = Path(args.latest_copy)
        latest.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(latest, index=False)
        print(f"Saved latest copy: {latest}")

    print(f"Saved submission: {submission_path}")
    print(f"Saved config: {run_dir / 'ensemble_config.json'}")


if __name__ == "__main__":
    main()
