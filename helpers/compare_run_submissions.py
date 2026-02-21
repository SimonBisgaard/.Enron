from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    cv_rmse: float | None
    lb_score: float | None
    submission: pd.DataFrame | None


def _to_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def _load_run(run_dir: Path) -> RunArtifacts:
    run_id = run_dir.name
    metrics_path = run_dir / "metrics.json"
    cv_results_path = run_dir / "cv_results.csv"
    submission_path = run_dir / "submission.csv"

    cv_rmse: float | None = None
    lb_score: float | None = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cv_rmse = _to_float(metrics.get("cv_rmse"))
            lb_score = _to_float(metrics.get("lb_score"))
        except Exception:
            pass
    if cv_rmse is None and cv_results_path.exists():
        try:
            cv_df = pd.read_csv(cv_results_path)
            if "rmse" in cv_df.columns:
                rmse_vals = pd.to_numeric(cv_df["rmse"], errors="coerce")
                valid = rmse_vals.notna()
                if valid.any():
                    if "val_rows" in cv_df.columns:
                        weights = pd.to_numeric(cv_df["val_rows"], errors="coerce").fillna(0.0)
                        w_valid = valid & (weights > 0)
                        if w_valid.any():
                            cv_rmse = float(np.average(rmse_vals[w_valid].to_numpy(dtype=float), weights=weights[w_valid].to_numpy(dtype=float)))
                        else:
                            cv_rmse = float(rmse_vals[valid].mean())
                    else:
                        cv_rmse = float(rmse_vals[valid].mean())
        except Exception:
            pass

    submission_df: pd.DataFrame | None = None
    if submission_path.exists():
        try:
            sub = pd.read_csv(submission_path)
            if {"id", "target"}.issubset(sub.columns):
                submission_df = sub[["id", "target"]].copy()
                submission_df["id"] = pd.to_numeric(submission_df["id"], errors="coerce")
                submission_df["target"] = pd.to_numeric(submission_df["target"], errors="coerce")
                submission_df = submission_df.dropna().sort_values("id").reset_index(drop=True)
        except Exception:
            submission_df = None

    return RunArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        cv_rmse=cv_rmse,
        lb_score=lb_score,
        submission=submission_df,
    )


def _scan_runs(runs_dir: Path) -> list[RunArtifacts]:
    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    run_dirs.sort(key=lambda p: p.name)
    return [_load_run(d) for d in run_dirs]


def _pick_newest(runs: list[RunArtifacts], explicit: str | None) -> RunArtifacts:
    if explicit:
        for r in runs:
            if r.run_id == explicit:
                return r
        raise ValueError(f"Newest run id not found: {explicit}")
    return runs[-1]


def _pick_best_cv(runs: list[RunArtifacts], explicit: str | None) -> RunArtifacts:
    if explicit:
        for r in runs:
            if r.run_id == explicit:
                return r
        raise ValueError(f"Best run id not found: {explicit}")

    candidates = [r for r in runs if r.cv_rmse is not None]
    if not candidates:
        raise ValueError("No runs with valid cv_rmse found.")
    return min(candidates, key=lambda r: float(r.cv_rmse))


def _load_submission_file(path: Path) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if not {"id", "target"}.issubset(sub.columns):
        raise ValueError(f"Submission file missing id/target columns: {path}")
    out = sub[["id", "target"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna().sort_values("id").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Submission file has no valid numeric rows: {path}")
    return out


def _match_run_from_submission_path(runs: list[RunArtifacts], submission_path: Path) -> RunArtifacts | None:
    try:
        ref = submission_path.resolve()
    except Exception:
        ref = submission_path
    for run in runs:
        candidate = (run.run_dir / "submission.csv")
        try:
            cand_resolved = candidate.resolve()
        except Exception:
            cand_resolved = candidate
        if cand_resolved == ref:
            return run
    return None


def _compare_predictions(best_sub: pd.DataFrame, new_sub: pd.DataFrame) -> dict[str, float]:
    merged = best_sub.merge(new_sub, on="id", how="inner", suffixes=("_best", "_new"))
    if merged.empty:
        raise ValueError("No overlapping IDs between submissions.")

    best = merged["target_best"].to_numpy(dtype=float)
    new = merged["target_new"].to_numpy(dtype=float)
    diff = new - best
    abs_diff = np.abs(diff)

    rmse_diff = float(np.sqrt(np.mean(diff**2)))
    mae_diff = float(np.mean(abs_diff))
    corr = float(np.corrcoef(best, new)[0, 1]) if len(best) > 1 else float("nan")
    p95_abs_diff = float(np.percentile(abs_diff, 95))
    p99_abs_diff = float(np.percentile(abs_diff, 99))

    top5_cut = np.percentile(abs_diff, 95)
    top5_mask = abs_diff >= top5_cut
    top5_mean_abs = float(np.mean(abs_diff[top5_mask])) if np.any(top5_mask) else 0.0

    return {
        "overlap_rows": float(len(merged)),
        "prediction_corr": corr,
        "prediction_rmse_diff": rmse_diff,
        "prediction_mae_diff": mae_diff,
        "prediction_mean_diff": float(np.mean(diff)),
        "prediction_abs_diff_p95": p95_abs_diff,
        "prediction_abs_diff_p99": p99_abs_diff,
        "prediction_abs_diff_top5_mean": top5_mean_abs,
    }


def _estimate_public_variance(public_rmse: float, public_n: int) -> dict[str, float]:
    sigma = float(public_rmse / math.sqrt(max(public_n, 1)))
    return {
        "public_n": float(public_n),
        "public_rmse_assumed": float(public_rmse),
        "public_rmse_std_error_1sigma": sigma,
        "public_rmse_noise_band_rough_low": float(public_rmse - sigma),
        "public_rmse_noise_band_rough_high": float(public_rmse + sigma),
        "public_rmse_noise_band_95_low": float(public_rmse - 1.96 * sigma),
        "public_rmse_noise_band_95_high": float(public_rmse + 1.96 * sigma),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare newest run submission vs an explicit reference submission (or best-CV fallback)."
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--newest-run-id", default=None)
    parser.add_argument("--best-run-id", default=None)
    parser.add_argument(
        "--best-submission-path",
        default=None,
        help="Champion submission file to compare against. If omitted, uses best-CV run submission.",
    )
    parser.add_argument("--public-rmse", type=float, default=None, help="If omitted, uses newest lb_score or fallback 25.0")
    parser.add_argument("--public-fraction", type=float, default=0.4)
    parser.add_argument("--test-size", type=int, default=13098)
    parser.add_argument("--json-out", default=None, help="Optional path to save comparison JSON.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    runs = _scan_runs(runs_dir)
    if not runs:
        raise ValueError(f"No run folders found in {runs_dir}")

    newest = _pick_newest(runs, args.newest_run_id)
    best = _pick_best_cv(runs, args.best_run_id)
    reference_run: RunArtifacts = best

    if newest.submission is None:
        raise ValueError(f"Newest run has no valid submission.csv: {newest.run_id}")
    best_submission_ref = Path(args.best_submission_path) if args.best_submission_path else None
    best_submission_df: pd.DataFrame | None = None
    best_submission_label: str = best.run_id

    if best_submission_ref is not None:
        if not best_submission_ref.exists():
            raise FileNotFoundError(f"--best-submission-path not found: {best_submission_ref}")
        best_submission_df = _load_submission_file(best_submission_ref)
        best_submission_label = f"file:{best_submission_ref}"
        matched = _match_run_from_submission_path(runs, best_submission_ref)
        if matched is not None:
            reference_run = matched
            best_submission_label = f"{matched.run_id} ({best_submission_ref})"

    if best_submission_df is None:
        if best.submission is None:
            raise ValueError(f"Best-CV run has no valid submission.csv: {best.run_id}")
        best_submission_df = best.submission

    compare = _compare_predictions(best_submission_df, newest.submission)

    cv_delta = None
    if newest.cv_rmse is not None and reference_run.cv_rmse is not None:
        cv_delta = float(newest.cv_rmse - reference_run.cv_rmse)

    public_rmse = args.public_rmse
    if public_rmse is None:
        public_rmse = newest.lb_score if newest.lb_score is not None else (best.lb_score if best.lb_score is not None else 25.0)
    public_n = int(round(float(args.public_fraction) * int(args.test_size)))
    variance = _estimate_public_variance(float(public_rmse), public_n)

    output = {
        "newest_run": {
            "run_id": newest.run_id,
            "cv_rmse": newest.cv_rmse,
            "lb_score": newest.lb_score,
        },
        "best_cv_run": {
            "run_id": best.run_id,
            "cv_rmse": best.cv_rmse,
            "lb_score": best.lb_score,
        },
        "reference_run": {
            "run_id": reference_run.run_id,
            "cv_rmse": reference_run.cv_rmse,
            "lb_score": reference_run.lb_score,
        },
        "best_submission_reference": best_submission_label,
        "cv_rmse_delta_new_minus_best": cv_delta,
        "submission_comparison": compare,
        "public_score_variance_estimate": variance,
        "interpretation": {
            "noise_threshold_guidance": (
                "Public RMSE changes smaller than about 0.3-0.5 are often noise at this sample size."
            ),
            "tail_focus_guidance": (
                "Use prediction_abs_diff_p95 / top5_mean to check whether changes are concentrated in tail behavior."
            ),
        },
    }

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Newest run: {newest.run_id} | cv_rmse={newest.cv_rmse} | lb={newest.lb_score}")
    print(f"Best CV run: {best.run_id} | cv_rmse={best.cv_rmse} | lb={best.lb_score}")
    print(f"Reference run: {reference_run.run_id} | cv_rmse={reference_run.cv_rmse} | lb={reference_run.lb_score}")
    print(f"Best submission reference: {best_submission_label}")
    print(f"CV delta (new-reference): {cv_delta}")
    print("--- Submission comparison ---")
    for k, v in compare.items():
        print(f"{k}: {v}")
    print("--- Public RMSE variance estimate ---")
    for k, v in variance.items():
        print(f"{k}: {v}")
    if args.json_out:
        print(f"Saved JSON report: {args.json_out}")


if __name__ == "__main__":
    main()
