from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RunInfo:
    run_id: str
    run_dir: Path
    submission_path: Path
    target_transform: str


def _load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    df = pd.read_csv(path)
    if not {"id", "target"}.issubset(df.columns):
        raise ValueError(f"Submission missing required columns id,target: {path}")
    out = df[["id", "target"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["id", "target"]).copy()
    out["id"] = out["id"].astype(int)
    return out.sort_values("id").reset_index(drop=True)


def _read_target_transform(run_dir: Path) -> str:
    meta = run_dir / "model_metadata.json"
    if not meta.exists():
        return "unknown"
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"

    # New format.
    tt = data.get("target_transform")
    if isinstance(tt, dict):
        method = tt.get("method")
        if isinstance(method, str):
            return method
    # Older format in train_args.
    train_args = data.get("train_args", {})
    if isinstance(train_args, dict):
        method = train_args.get("target_transform")
        if isinstance(method, str):
            return method
    return "unknown"


def _scan_runs(runs_dir: Path) -> list[RunInfo]:
    infos: list[RunInfo] = []
    for d in sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        sub = d / "submission.csv"
        if not sub.exists():
            continue
        infos.append(
            RunInfo(
                run_id=d.name,
                run_dir=d,
                submission_path=sub,
                target_transform=_read_target_transform(d),
            )
        )
    return infos


def _find_latest_transformed_run(
    runs: list[RunInfo],
    explicit_run_id: str | None,
) -> RunInfo:
    if explicit_run_id:
        for r in runs:
            if r.run_id == explicit_run_id:
                return r
        raise ValueError(f"Run id not found: {explicit_run_id}")

    transformed = [
        r for r in runs if r.target_transform in {"signed_log", "log_shift", "yeo_johnson"}
    ]
    if not transformed:
        raise ValueError("No transformed runs found with target_transform in {signed_log, log_shift, yeo_johnson}.")
    return transformed[-1]


def _match_run_by_token(runs: list[RunInfo], token: str) -> RunInfo:
    exact = [r for r in runs if r.run_id == token]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"Multiple exact run matches for token={token}")

    partial = [r for r in runs if token in r.run_id]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"No run id contains token={token}")
    raise ValueError(
        f"Token={token} is ambiguous across multiple run ids: {[r.run_id for r in partial]}"
    )


def _find_best_submission_from_experiments(registry_path: Path) -> Path | None:
    if not registry_path.exists():
        return None
    best_row: dict[str, str] | None = None
    best_cv = float("inf")
    with registry_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("cv_rmse", "")
            try:
                cv = float(val)
            except Exception:
                continue
            if not np.isfinite(cv):
                continue
            if cv < best_cv:
                best_cv = cv
                best_row = row
    if best_row is None:
        return None
    sub_path = best_row.get("submission_path", "")
    if not sub_path:
        return None
    p = Path(sub_path)
    return p if p.exists() else None


def _peak_mask(values: np.ndarray, mode: str, upper_q: float, lower_q: float) -> np.ndarray:
    if mode == "upper":
        thr = float(np.quantile(values, upper_q))
        return values >= thr
    if mode == "two_sided":
        hi = float(np.quantile(values, upper_q))
        lo = float(np.quantile(values, lower_q))
        return (values >= hi) | (values <= lo)
    if mode == "abs":
        abs_v = np.abs(values)
        thr = float(np.quantile(abs_v, upper_q))
        return abs_v >= thr
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Blend latest transformed submission with peak rows from best submission.\n"
            "Rows marked as peaks in best submission replace target in latest transformed submission."
        )
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--transformed-run-id",
        default="172917",
        help=(
            "Run id (or unique token) for the transformed/base submission. "
            "Default is 172917."
        ),
    )
    parser.add_argument(
        "--best-run-id",
        default="151106",
        help=(
            "Run id (or unique token) for the best reference submission. "
            "Default is 151106."
        ),
    )
    parser.add_argument(
        "--best-submission-path",
        default="csv/submissio.csv",
        help="Best reference submission file. If missing, falls back to best CV run in csv/experiments.csv.",
    )
    parser.add_argument("--registry", default="csv/experiments.csv")
    parser.add_argument("--mode", choices=["upper", "two_sided", "abs"], default="upper")
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument(
        "--out-path",
        default=None,
        help="Output path. Default: runs/<latest_run>/submission_peakswap_<mode>_qXX.csv",
    )
    args = parser.parse_args()

    if not (0.0 < args.upper_quantile < 1.0):
        raise ValueError("--upper-quantile must be in (0,1)")
    if not (0.0 < args.lower_quantile < 1.0):
        raise ValueError("--lower-quantile must be in (0,1)")
    if args.lower_quantile >= args.upper_quantile:
        raise ValueError("--lower-quantile must be < --upper-quantile")

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    runs = _scan_runs(runs_dir)
    if not runs:
        raise ValueError(f"No run submissions found in {runs_dir}")

    if args.transformed_run_id:
        latest = _match_run_by_token(runs, args.transformed_run_id)
    else:
        latest = _find_latest_transformed_run(runs, explicit_run_id=None)
    latest_sub = _load_submission(latest.submission_path)

    best_ref: Path | None = None
    if args.best_run_id:
        best_run = _match_run_by_token(runs, args.best_run_id)
        best_ref = best_run.submission_path
    else:
        best_ref = Path(args.best_submission_path)
        if not best_ref.exists():
            fallback = _find_best_submission_from_experiments(Path(args.registry))
            if fallback is None:
                raise FileNotFoundError(
                    f"Best submission file not found ({best_ref}) and no fallback from {args.registry}."
                )
            best_ref = fallback
    best_sub = _load_submission(best_ref)

    merged = latest_sub.merge(best_sub, on="id", how="inner", suffixes=("_latest", "_best"))
    if merged.empty:
        raise ValueError("No overlapping IDs between latest transformed and best submission.")

    mask = _peak_mask(
        values=merged["target_best"].to_numpy(dtype=float),
        mode=args.mode,
        upper_q=args.upper_quantile,
        lower_q=args.lower_quantile,
    )
    merged["target_out"] = merged["target_latest"].to_numpy(dtype=float)
    merged.loc[mask, "target_out"] = merged.loc[mask, "target_best"].to_numpy(dtype=float)

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        q_tag = int(round(args.upper_quantile * 1000))
        out_path = latest.run_dir / f"submission_peakswap_{args.mode}_q{q_tag}.csv"
    out = merged[["id", "target_out"]].rename(columns={"target_out": "target"}).sort_values("id")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    replaced = int(mask.sum())
    total = int(len(mask))
    mean_abs_change = float(
        np.mean(np.abs(merged["target_out"].to_numpy(dtype=float) - merged["target_latest"].to_numpy(dtype=float)))
    )
    p95_abs_change = float(
        np.percentile(
            np.abs(merged["target_out"].to_numpy(dtype=float) - merged["target_latest"].to_numpy(dtype=float)),
            95,
        )
    )

    print(f"Latest transformed run: {latest.run_id} (transform={latest.target_transform})")
    print(f"Latest submission: {latest.submission_path}")
    print(f"Best submission ref: {best_ref}")
    print(
        f"Peak mode={args.mode}, upper_q={args.upper_quantile}, lower_q={args.lower_quantile} "
        f"| replaced_rows={replaced}/{total} ({replaced/total:.2%})"
    )
    print(f"mean_abs_change={mean_abs_change:.6f}, p95_abs_change={p95_abs_change:.6f}")
    print(f"Saved blended submission: {out_path}")


if __name__ == "__main__":
    main()
