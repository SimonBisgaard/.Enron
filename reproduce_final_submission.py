from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LocalTrainSpec:
    key: str
    run_name: str
    cmd_args: list[str]


@dataclass(frozen=True)
class JensRebuildSpec:
    key: str
    run_name: str
    cmd_args: list[str]
    fallback_submission: Path


DEFAULT_WEIGHTS = {
    "jens_215400": 0.35,
    "jens_175700": 0.20,
    "local_172305": 0.25,
    "local_083432": 0.10,
    "local_201434": 0.10,
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_weights(raw: str) -> dict[str, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 5:
        raise ValueError("Expected 5 comma-separated weights in order: j215400,j175700,m172305,m083432,m201434")
    vals = [float(x) for x in parts]
    if any(v < 0 for v in vals):
        raise ValueError("Weights must be non-negative.")
    s = float(sum(vals))
    if s <= 0:
        raise ValueError("Weights must sum to > 0.")
    vals = [v / s for v in vals]
    return {
        "jens_215400": vals[0],
        "jens_175700": vals[1],
        "local_172305": vals[2],
        "local_083432": vals[3],
        "local_201434": vals[4],
    }


def _run(cmd: list[str], cwd: Path) -> None:
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _find_new_run_dir(before: set[Path], runs_dir: Path, suffix: str) -> Path:
    candidates = {p.resolve() for p in runs_dir.glob(f"*_{suffix}") if p.is_dir()}
    new = sorted(candidates - before, key=lambda p: p.stat().st_mtime)
    if new:
        return new[-1]
    raise RuntimeError(
        f"Command completed but no NEW run directory was created for suffix '{suffix}'. "
        "This reproduction mode requires fresh retraining/rebuild artifacts."
    )


def _find_latest_run_dir(runs_dir: Path, suffix: str) -> Path:
    candidates = sorted(
        [p.resolve() for p in runs_dir.glob(f"*_{suffix}") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directory found for suffix '{suffix}' in {runs_dir}")
    return candidates[-1]


def _run_and_capture_run_dir(cmd: list[str], runs_dir: Path, suffix: str, cwd: Path) -> Path:
    before = {p.resolve() for p in runs_dir.glob(f"*_{suffix}") if p.is_dir()}
    _run(cmd, cwd=cwd)
    run_dir = _find_new_run_dir(before, runs_dir, suffix)
    sub = run_dir / "submission.csv"
    if not sub.exists():
        raise FileNotFoundError(f"Expected submission.csv in {run_dir}")
    return run_dir


def _load_submission(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path, usecols=["id", "target"])
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["id", "target"]).copy()
    df["id"] = df["id"].astype(int)
    return df.groupby("id", as_index=False)["target"].mean()


def _build_local_specs() -> list[LocalTrainSpec]:
    py = sys.executable
    return [
        LocalTrainSpec(
            key="local_172305",
            run_name="per_market_interactions_2c02eb6_oct2023_cv1_14d",
            cmd_args=[
                py,
                "train_per_market_interactions_2c02eb6.py",
                "--name",
                "per_market_interactions_2c02eb6_oct2023_cv1_14d",
                "--train-start-oct-2023",
                "--cv",
                "--cv-folds",
                "1",
                "--cv-val-days",
                "14",
                "--cv-step-days",
                "14",
                "--cv-min-train-days",
                "90",
                "--save-shap",
                "--save-models",
                "--save-repro-artifacts",
            ],
        ),
        LocalTrainSpec(
            key="local_201434",
            run_name="per_market_interactions_2c02eb6_oct2023_cv1_14d_temp_physics_pruned_nores",
            cmd_args=[
                py,
                "train_per_market_interactions_2c02eb6.py",
                "--name",
                "per_market_interactions_2c02eb6_oct2023_cv1_14d_temp_physics_pruned_nores",
                "--train-start-oct-2023",
                "--add-temperature-demand-features",
                "--add-physics-regime-features",
                "--drop-redundant-features",
                "--cv",
                "--cv-folds",
                "1",
                "--cv-val-days",
                "14",
                "--cv-step-days",
                "14",
                "--cv-min-train-days",
                "90",
                "--save-shap",
                "--save-models",
                "--save-repro-artifacts",
            ],
        ),
        LocalTrainSpec(
            key="local_083432",
            run_name="per_market_interactions_commit_baseline_multimodel_nocv",
            cmd_args=[
                py,
                "train_per_market_interactions_commit_baseline_multimodel.py",
                "--name",
                "per_market_interactions_commit_baseline_multimodel_nocv",
                "--models",
                "catboost,lightgbm,xgboost,ridge,lasso,rf",
                "--allow-missing-models",
                "--save-models",
                "--seed",
                "42",
            ],
        ),
    ]


def _build_jens_specs() -> list[JensRebuildSpec]:
    py = sys.executable
    return [
        JensRebuildSpec(
            key="jens_215400",
            run_name="rebuild_20260221_215400_jens",
            cmd_args=[
                py,
                "helpers/reproduce_jens_215400.py",
                "--name",
                "rebuild_20260221_215400_jens",
                "--best-config-path",
                "jens_repro/best_config_20260221_213508_recheck.json",
                "--v1-submission-path",
                "jens_repro/submission_v1_recommended_none_alpha0p7.csv",
                "--v2-submission-path",
                "jens_repro/submission_v2_recommended_none_alpha0p7.csv",
                "--reference-submission-path",
                "jens_repro/reference_submission_v3_215400.csv",
                "--verify-against-run",
                "runs/20260221-215400_jens/submission.csv",
            ],
            fallback_submission=Path("runs/20260221-215400_jens/submission.csv"),
        ),
        JensRebuildSpec(
            key="jens_175700",
            run_name="rebuild_20260222_175700_jens",
            cmd_args=[
                py,
                "helpers/reproduce_jens_215400.py",
                "--name",
                "rebuild_20260222_175700_jens",
                "--best-config-path",
                "jens_repro/best_config_20260222_160520_submission3_xgb_kickoff_blend.json",
                "--v1-submission-path",
                "jens_repro/submission_xgb_v1_recommended_none_alpha1.csv",
                "--v2-submission-path",
                "jens_repro/submission_xgb_v2_recommended_none_alpha1.csv",
                "--reference-submission-path",
                "jens_repro/reference_submission_v3_175700.csv",
                "--verify-against-run",
                "runs/20260222-175700_jens/submission.csv",
            ],
            fallback_submission=Path("runs/20260222-175700_jens/submission.csv"),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Single-command repro pipeline: retrain local models, rebuild Jens artifacts, "
            "and generate final weighted ensemble submission."
        )
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--name", default="final_submission_repro_bundle")
    parser.add_argument("--latest-copy", default="csv/submission_final_submission_repro_bundle.csv")
    parser.add_argument(
        "--weights",
        default="0.35,0.20,0.25,0.10,0.10",
        help="Comma-separated weights in order: j215400,j175700,m172305,m083432,m201434",
    )
    parser.add_argument(
        "--train-local-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrain the 3 local models (default: true).",
    )
    parser.add_argument(
        "--rebuild-jens-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild Jens submissions from artifacts via helpers/reproduce_jens_215400.py (default: true).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    weights = _parse_weights(args.weights)
    local_specs = _build_local_specs()
    jens_specs = _build_jens_specs()

    component_submissions: dict[str, Path] = {}
    component_runs: dict[str, Path | None] = {}

    print("Weights:")
    for k in ["jens_215400", "jens_175700", "local_172305", "local_083432", "local_201434"]:
        print(f"- {k}: {weights[k]:.6f}")

    if args.train_local_models:
        print("\n[Step] Retraining local models...")
        for spec in local_specs:
            if args.dry_run:
                print(f"DRY-RUN local: {' '.join(shlex.quote(x) for x in spec.cmd_args)}")
                continue
            run_dir = _run_and_capture_run_dir(spec.cmd_args, runs_dir=runs_dir, suffix=spec.run_name, cwd=root)
            component_runs[spec.key] = run_dir
            component_submissions[spec.key] = run_dir / "submission.csv"
            print(f"Local component ready: {spec.key} -> {component_submissions[spec.key]}")
    else:
        for spec in local_specs:
            run_dir = _find_latest_run_dir(runs_dir, spec.run_name)
            component_runs[spec.key] = run_dir
            component_submissions[spec.key] = run_dir / "submission.csv"
            print(f"Using existing local component: {spec.key} -> {component_submissions[spec.key]}")

    if args.rebuild_jens_models:
        print("\n[Step] Rebuilding Jens models from artifacts...")
        for spec in jens_specs:
            if args.dry_run:
                print(f"DRY-RUN jens: {' '.join(shlex.quote(x) for x in spec.cmd_args)}")
                continue
            run_dir = _run_and_capture_run_dir(spec.cmd_args, runs_dir=runs_dir, suffix=spec.run_name, cwd=root)
            component_runs[spec.key] = run_dir
            component_submissions[spec.key] = run_dir / "submission.csv"
            print(f"Jens component ready: {spec.key} -> {component_submissions[spec.key]}")
    else:
        for spec in jens_specs:
            component_runs[spec.key] = None
            component_submissions[spec.key] = spec.fallback_submission
            print(f"Using fallback Jens component: {spec.key} -> {component_submissions[spec.key]}")

    if args.dry_run:
        print("\nDry run complete. No files written.")
        return

    order = ["jens_215400", "jens_175700", "local_172305", "local_083432", "local_201434"]
    loaded = []
    for key in order:
        loaded.append(_load_submission(component_submissions[key], label=key))

    base_ids = loaded[0]["id"]
    for key, df in zip(order[1:], loaded[1:]):
        if len(df) != len(base_ids) or not df["id"].equals(base_ids):
            raise ValueError(f"ID mismatch in component: {key}")

    pred_stack = np.column_stack([df["target"].to_numpy(dtype=float) for df in loaded])
    w = np.asarray([weights[k] for k in order], dtype=float)
    final_pred = np.sum(pred_stack * w.reshape(1, -1), axis=1)

    sample = pd.read_csv(args.sample_submission, usecols=["id"])
    sample["id"] = pd.to_numeric(sample["id"], errors="coerce")
    sample = sample.dropna(subset=["id"]).copy()
    sample["id"] = sample["id"].astype(int)
    out = sample.merge(pd.DataFrame({"id": base_ids.to_numpy(), "target": final_pred}), on="id", how="left")
    if out["target"].isna().any():
        raise ValueError("NaN targets after merging weighted predictions into sample submission.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{args.name}"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    submission_path = run_dir / "submission.csv"
    out.to_csv(submission_path, index=False)

    latest_copy = Path(args.latest_copy)
    latest_copy.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(latest_copy, index=False)

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "weights": weights,
        "component_run_dirs": {k: (str(v) if v is not None else None) for k, v in component_runs.items()},
        "component_submissions": {k: str(v) for k, v in component_submissions.items()},
        "component_submission_sha256": {k: _sha256_file(v) for k, v in component_submissions.items()},
        "final_submission_sha256": _sha256_file(submission_path),
        "n_rows": int(len(out)),
    }
    (run_dir / "repro_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\n[Done]")
    print(f"Saved final submission: {submission_path}")
    print(f"Saved latest copy: {latest_copy}")
    print(f"Saved repro manifest: {run_dir / 'repro_manifest.json'}")


if __name__ == "__main__":
    main()
