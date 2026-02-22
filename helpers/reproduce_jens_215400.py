from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _load_feature_module(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Feature module not found: {path}")
    spec = importlib.util.spec_from_file_location("jens_market_layer_features_v2", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclass/type-introspection on dynamically loaded modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_test_context(
    *,
    train_path: Path,
    test_path: Path,
    feature_module: Any,
    use_xmk: bool,
) -> pd.DataFrame:
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    if not hasattr(feature_module, "build_train_test_features"):
        raise AttributeError("Feature module must define build_train_test_features(train_df, test_df, ...)")

    build_sig = inspect.signature(feature_module.build_train_test_features)
    if "use_xmk" in build_sig.parameters:
        _, test_feat, _, _ = feature_module.build_train_test_features(train_raw, test_raw, use_xmk=use_xmk)
    else:
        _, test_feat, _, _ = feature_module.build_train_test_features(train_raw, test_raw)

    required = {"id", "market", "primary_regime"}
    missing = sorted(required - set(test_feat.columns))
    if missing:
        raise ValueError(f"Feature builder output missing required test columns: {missing}")

    if "ts" in test_feat.columns:
        ts = pd.to_datetime(test_feat["ts"], errors="coerce")
    else:
        ts = pd.to_datetime(test_raw["delivery_start"], errors="coerce")
        test_feat = test_feat.merge(test_raw[["id", "delivery_start"]], on="id", how="left")

    out = test_feat[["id", "market", "primary_regime"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out = out.dropna(subset=["id"]).copy()
    out["id"] = out["id"].astype(int)
    out["market"] = out["market"].astype(str)
    out["primary_regime"] = out["primary_regime"].astype(str)
    out["ts"] = ts
    if out["ts"].isna().any():
        raise ValueError("Invalid timestamps in test context.")
    out["hour"] = out["ts"].dt.hour.astype(int)
    return out.sort_values("id").reset_index(drop=True)


def _weights_from_config(df: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    config_type = str(cfg.get("config_type", "rule"))
    if config_type == "global":
        w_global = cfg.get("w_global", 0.0)
        w = np.full(len(df), float(0.0 if w_global is None else w_global), dtype=float)
    else:
        by_reg = {
            "peak_scarcity": float(cfg.get("w_peak", 0.0)),
            "transition": float(cfg.get("w_transition", 0.0)),
            "normal": float(cfg.get("w_normal", 0.0)),
            "lowprice_midday": float(cfg.get("w_lowprice", 0.0)),
        }
        w = df["primary_regime"].map(by_reg).fillna(by_reg["normal"]).to_numpy(dtype=float)
        if int(cfg.get("force_market_a_normal_nonpeak_zero", 0)) == 1:
            force_mask = (
                (df["market"].to_numpy() == "Market A")
                & (df["primary_regime"].to_numpy() == "normal")
                & (~df["hour"].between(17, 20).to_numpy())
            )
            w = np.where(force_mask, 0.0, w)
    return np.clip(w, 0.0, 1.0)


def _load_submission(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    required = {"id", "target"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"{label} missing required columns: {missing}")
    out = df[["id", "target"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["id", "target"]).copy()
    out["id"] = out["id"].astype(int)
    return out.sort_values("id").reset_index(drop=True)


def _compare_submissions(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, Any]:
    m = a.merge(b, on="id", suffixes=("_a", "_b"), how="inner")
    if m.empty:
        raise ValueError("No overlapping ids between compared submissions.")
    x = m["target_a"].to_numpy(dtype=float)
    y = m["target_b"].to_numpy(dtype=float)
    d = x - y
    return {
        "rows_compared": int(len(m)),
        "exact_equal": bool(np.allclose(x, y, rtol=0.0, atol=0.0)),
        "corr": float(np.corrcoef(x, y)[0, 1]) if len(m) > 1 else float("nan"),
        "mae_diff": float(np.mean(np.abs(d))),
        "rmse_diff": float(np.sqrt(np.mean(d**2))),
        "max_abs_diff": float(np.max(np.abs(d))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Jens 215400 v3 blend submission with frozen config.")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--feature-module-path", default="jens_repro/market_layer_features_v2.py")
    parser.add_argument("--best-config-path", default="jens_repro/best_config_20260221_213508_recheck.json")
    parser.add_argument("--v1-submission-path", default="jens_repro/submission_v1_recommended_none_alpha0p7.csv")
    parser.add_argument("--v2-submission-path", default="jens_repro/submission_v2_recommended_none_alpha0p7.csv")
    parser.add_argument("--reference-submission-path", default="jens_repro/reference_submission_v3_215400.csv")
    parser.add_argument("--verify-against-run", default="runs/20260221-215400_jens/submission.csv")
    parser.add_argument("--use-xmk", type=int, choices=[0, 1], default=1)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="jens_215400_repro")
    parser.add_argument("--latest-copy", default="csv/submission_jens_215400_repro.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    sample_path = Path(args.sample_submission)
    feature_module_path = Path(args.feature_module_path)
    best_config_path = Path(args.best_config_path)
    v1_sub_path = Path(args.v1_submission_path)
    v2_sub_path = Path(args.v2_submission_path)
    ref_sub_path = Path(args.reference_submission_path)
    verify_run_path = Path(args.verify_against_run) if args.verify_against_run else None

    cfg_payload = json.loads(best_config_path.read_text(encoding="utf-8"))
    best_cfg = cfg_payload.get("best_config", cfg_payload)
    if not isinstance(best_cfg, dict):
        raise ValueError("Best config payload must be an object.")

    feature_module = _load_feature_module(feature_module_path)
    test_ctx = _prepare_test_context(
        train_path=train_path,
        test_path=test_path,
        feature_module=feature_module,
        use_xmk=bool(args.use_xmk),
    )
    v1_sub = _load_submission(v1_sub_path, label="v1 submission")
    v2_sub = _load_submission(v2_sub_path, label="v2 submission")
    sample = pd.read_csv(sample_path, usecols=["id"])
    sample["id"] = pd.to_numeric(sample["id"], errors="coerce")
    sample = sample.dropna(subset=["id"]).copy()
    sample["id"] = sample["id"].astype(int)
    sample = sample.sort_values("id").reset_index(drop=True)

    merged = (
        test_ctx[["id", "market", "primary_regime", "hour"]]
        .merge(v1_sub.rename(columns={"target": "p1"}), on="id", how="inner")
        .merge(v2_sub.rename(columns={"target": "p2"}), on="id", how="inner")
    )
    if len(merged) != len(sample):
        raise ValueError(
            f"Merged rows ({len(merged)}) do not match sample submission rows ({len(sample)})."
        )

    w = _weights_from_config(merged, best_cfg)
    pred = (1.0 - w) * merged["p1"].to_numpy(dtype=float) + w * merged["p2"].to_numpy(dtype=float)
    out = merged[["id"]].copy()
    out["target"] = pred
    out = sample[["id"]].merge(out, on="id", how="left")
    if out["target"].isna().any():
        raise ValueError("Output has NaN targets after id alignment.")

    compare_results: dict[str, Any] = {}
    if ref_sub_path.exists():
        compare_results["vs_reference_bundle_submission"] = _compare_submissions(
            out,
            _load_submission(ref_sub_path, label="reference bundle submission"),
        )
    if verify_run_path is not None and verify_run_path.exists():
        compare_key = f"vs_{verify_run_path.as_posix()}"
        compare_results[compare_key] = _compare_submissions(
            out,
            _load_submission(verify_run_path, label="verification run submission"),
        )

    print("Blend config used:")
    print(json.dumps(best_cfg, indent=2, sort_keys=True))
    for key, value in compare_results.items():
        print(f"{key}: {value}")

    if args.dry_run:
        print("Dry run complete. No files were written.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{args.name}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    out_path = run_dir / "submission.csv"
    out.to_csv(out_path, index=False)

    run_config = {
        "script": Path(__file__).name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_args": vars(args),
        "best_config": best_cfg,
        "inputs": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "sample_submission_path": str(sample_path),
            "feature_module_path": str(feature_module_path),
            "best_config_path": str(best_config_path),
            "v1_submission_path": str(v1_sub_path),
            "v2_submission_path": str(v2_sub_path),
            "reference_submission_path": str(ref_sub_path),
            "verify_against_run": str(verify_run_path) if verify_run_path is not None else None,
        },
        "hashes": {
            "train_sha256": _sha256_file(train_path),
            "test_sha256": _sha256_file(test_path),
            "sample_submission_sha256": _sha256_file(sample_path),
            "feature_module_sha256": _sha256_file(feature_module_path),
            "best_config_sha256": _sha256_file(best_config_path),
            "v1_submission_sha256": _sha256_file(v1_sub_path),
            "v2_submission_sha256": _sha256_file(v2_sub_path),
            "reference_submission_sha256": _sha256_file(ref_sub_path) if ref_sub_path.exists() else None,
        },
        "comparisons": compare_results,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.latest_copy:
        latest = Path(args.latest_copy)
        latest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(latest, index=False)
        print(f"Saved latest copy: {latest}")

    print(f"Saved submission: {out_path}")
    print(f"Saved run config: {run_dir / 'run_config.json'}")


if __name__ == "__main__":
    main()
