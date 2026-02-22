from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_hours(raw: str, *, arg_name: str = "--hours") -> list[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{arg_name} must contain at least one hour.")
    out: list[int] = []
    for v in vals:
        h = int(v)
        if h < 0 or h > 23:
            raise ValueError(f"Invalid hour in {arg_name}: {h}")
        out.append(h)
    return sorted(set(out))


def _resolve_run_dir(token: str, runs_dir: Path) -> Path:
    candidate = Path(token)
    if candidate.exists():
        if candidate.is_file():
            if candidate.name != "submission.csv":
                raise ValueError(f"Expected a run directory or submission.csv file, got: {candidate}")
            run_dir = candidate.parent
        else:
            run_dir = candidate
        if (run_dir / "submission.csv").is_file():
            return run_dir
        raise FileNotFoundError(f"Resolved path has no submission.csv: {run_dir}")

    direct = runs_dir / token
    if direct.is_dir() and (direct / "submission.csv").is_file():
        return direct

    matches = [p for p in runs_dir.iterdir() if p.is_dir() and token in p.name]
    if not matches:
        raise FileNotFoundError(f"No run directory matches token '{token}' in {runs_dir}")
    if len(matches) > 1:
        names = ", ".join(sorted(p.name for p in matches))
        raise ValueError(f"Token '{token}' is ambiguous; matched: {names}")
    if not (matches[0] / "submission.csv").is_file():
        raise FileNotFoundError(f"Matched run has no submission.csv: {matches[0]}")
    return matches[0]


def _build_calibration_table(
    oof: pd.DataFrame,
    *,
    hours: list[int],
    apply_market_a_night: bool,
    market_a_label: str,
    market_a_night_hours: list[int],
    min_rows: int,
    alpha: float,
    max_abs_adjustment: float,
) -> pd.DataFrame:
    req = {"market", "delivery_start", "target", "pred"}
    if not req.issubset(oof.columns):
        missing = sorted(req - set(oof.columns))
        raise ValueError(f"cv_oof.csv missing required columns: {missing}")

    work = oof.copy()
    work["delivery_start"] = pd.to_datetime(work["delivery_start"], errors="coerce")
    work["hour"] = work["delivery_start"].dt.hour
    work["market"] = work["market"].astype(str)
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    work["pred"] = pd.to_numeric(work["pred"], errors="coerce")
    work = work.dropna(subset=["hour", "target", "pred"])
    work["hour"] = work["hour"].astype(int)
    keep = work["hour"].isin(hours)
    if apply_market_a_night:
        keep = keep | (
            (work["market"] == str(market_a_label))
            & work["hour"].isin(market_a_night_hours)
        )
    work = work[keep].copy()
    work["err"] = work["pred"] - work["target"]
    work["sq_err"] = work["err"] ** 2

    grouped = (
        work.groupby(["market", "hour"], dropna=False)
        .agg(
            rows=("err", "size"),
            bias=("err", "mean"),
            rmse=("sq_err", lambda s: float(np.sqrt(np.mean(s)))),
        )
        .reset_index()
    )
    grouped = grouped[grouped["rows"] >= int(min_rows)].copy()
    grouped["raw_adjustment"] = -float(alpha) * grouped["bias"].astype(float)
    grouped["adjustment"] = grouped["raw_adjustment"].clip(
        lower=-float(max_abs_adjustment),
        upper=float(max_abs_adjustment),
    )
    grouped["is_global_hour_rule"] = grouped["hour"].isin(hours)
    grouped["is_market_a_night_rule"] = bool(apply_market_a_night) & (
        (grouped["market"] == str(market_a_label))
        & grouped["hour"].isin(market_a_night_hours)
    )
    grouped = grouped.sort_values(["market", "hour"]).reset_index(drop=True)
    return grouped


def _apply_calibration(
    submission: pd.DataFrame,
    test_meta: pd.DataFrame,
    calibration_table: pd.DataFrame,
) -> pd.DataFrame:
    req_sub = {"id", "target"}
    if not req_sub.issubset(submission.columns):
        missing = sorted(req_sub - set(submission.columns))
        raise ValueError(f"submission.csv missing required columns: {missing}")
    req_test = {"id", "market", "delivery_start"}
    if not req_test.issubset(test_meta.columns):
        missing = sorted(req_test - set(test_meta.columns))
        raise ValueError(f"test metadata missing required columns: {missing}")

    work = submission[["id", "target"]].copy()
    work["id"] = pd.to_numeric(work["id"], errors="coerce")
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    if work["id"].isna().any() or work["target"].isna().any():
        raise ValueError("submission.csv has non-numeric id/target values.")
    work["id"] = work["id"].astype(int)

    meta = test_meta[["id", "market", "delivery_start"]].copy()
    meta["id"] = pd.to_numeric(meta["id"], errors="coerce")
    if meta["id"].isna().any():
        raise ValueError("test metadata has non-numeric id values.")
    meta["id"] = meta["id"].astype(int)
    meta["delivery_start"] = pd.to_datetime(meta["delivery_start"], errors="coerce")
    meta["hour"] = meta["delivery_start"].dt.hour
    meta["market"] = meta["market"].astype(str)

    merged = work.merge(meta[["id", "market", "hour"]], on="id", how="left")
    if merged["market"].isna().any() or merged["hour"].isna().any():
        raise ValueError("Unable to join some submission rows to test metadata by id.")
    merged["hour"] = merged["hour"].astype(int)

    cal = calibration_table[["market", "hour", "adjustment"]].copy()
    merged = merged.merge(cal, on=["market", "hour"], how="left")
    merged["adjustment"] = merged["adjustment"].fillna(0.0).astype(float)
    merged["target"] = merged["target"] + merged["adjustment"]
    return merged[["id", "target", "market", "hour", "adjustment"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a run submission using OOF market-hour bias and save a new run under runs/. "
            "By default, calibrates hour 19 globally and Market A night hours."
        )
    )
    parser.add_argument(
        "source_run",
        help="Source run id/token/path containing submission.csv (e.g. 20260222-132125_orthogonal_tuned_2h).",
    )
    parser.add_argument(
        "--calibration-run",
        default=None,
        help="Run id/token/path containing cv_oof.csv for calibration. Defaults to source_run.",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--name", default="submission_market_hour_calibrated")
    parser.add_argument("--hours", default="19", help="Comma-separated hours to calibrate. Default: 19")
    parser.add_argument(
        "--apply-market-a-night",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also calibrate Market A in night hours (default: enabled).",
    )
    parser.add_argument(
        "--market-a-label",
        default="Market A",
        help="Market label used for the night-hour calibration rule (default: 'Market A').",
    )
    parser.add_argument(
        "--market-a-night-hours",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated Market A night hours to calibrate when enabled (default: 0-7).",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Adjustment scale on OOF bias. Default: 0.5")
    parser.add_argument("--min-rows", type=int, default=20, help="Minimum OOF rows per market-hour cell. Default: 20")
    parser.add_argument(
        "--max-abs-adjustment",
        type=float,
        default=30.0,
        help="Absolute cap for each cell adjustment. Default: 30.0",
    )
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument(
        "--latest-copy",
        default="csv/submission_market_hour_calibrated.csv",
        help="Path for a latest copy CSV. Set empty string to disable.",
    )
    args = parser.parse_args()

    if args.min_rows <= 0:
        raise ValueError("--min-rows must be > 0.")
    if args.max_abs_adjustment < 0.0:
        raise ValueError("--max-abs-adjustment must be >= 0.")

    hours = _parse_hours(args.hours, arg_name="--hours")
    market_a_night_hours = _parse_hours(args.market_a_night_hours, arg_name="--market-a-night-hours")
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    source_run_dir = _resolve_run_dir(args.source_run, runs_dir)
    calibration_run_dir = _resolve_run_dir(args.calibration_run, runs_dir) if args.calibration_run else source_run_dir

    submission_path = source_run_dir / "submission.csv"
    cv_oof_path = calibration_run_dir / "cv_oof.csv"
    if not cv_oof_path.is_file():
        raise FileNotFoundError(
            f"Calibration run has no cv_oof.csv: {calibration_run_dir}. "
            "Provide --calibration-run with a run that has cv_oof.csv."
        )

    submission = pd.read_csv(submission_path)
    oof = pd.read_csv(cv_oof_path)
    test_meta = pd.read_csv(args.test_path, usecols=["id", "market", "delivery_start"])
    sample = pd.read_csv(args.sample_submission, usecols=["id"])

    cal_table = _build_calibration_table(
        oof,
        hours=hours,
        apply_market_a_night=bool(args.apply_market_a_night),
        market_a_label=str(args.market_a_label),
        market_a_night_hours=market_a_night_hours,
        min_rows=args.min_rows,
        alpha=args.alpha,
        max_abs_adjustment=args.max_abs_adjustment,
    )
    if cal_table.empty:
        raise ValueError(
            "Calibration table is empty after filters. "
            "Try lower --min-rows, different --hours, or another --calibration-run."
        )

    adjusted = _apply_calibration(submission, test_meta, cal_table)
    out = sample.copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce").astype(int)
    out = out.merge(adjusted[["id", "target"]], on="id", how="left")
    if out["target"].isna().any():
        raise ValueError("Output submission has NaN targets after applying calibration.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{args.name}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    out_path = run_dir / "submission.csv"
    out.to_csv(out_path, index=False)

    cal_table.to_csv(run_dir / "calibration_table.csv", index=False)
    adjusted[["id", "market", "hour", "adjustment"]].to_csv(run_dir / "row_adjustments.csv", index=False)

    stats: dict[str, Any] = {
        "run_id": run_id,
        "source_run_dir": str(source_run_dir),
        "calibration_run_dir": str(calibration_run_dir),
        "submission_path": str(submission_path),
        "cv_oof_path": str(cv_oof_path),
        "hours": hours,
        "apply_market_a_night": bool(args.apply_market_a_night),
        "market_a_label": str(args.market_a_label),
        "market_a_night_hours": market_a_night_hours,
        "alpha": float(args.alpha),
        "min_rows": int(args.min_rows),
        "max_abs_adjustment": float(args.max_abs_adjustment),
        "n_calibrated_cells": int(len(cal_table)),
        "n_calibrated_cells_global_hour_rule": int(cal_table["is_global_hour_rule"].sum()),
        "n_calibrated_cells_market_a_night_rule": int(cal_table["is_market_a_night_rule"].sum()),
        "n_submission_rows": int(len(out)),
        "n_adjusted_rows": int((adjusted["adjustment"].abs() > 1e-12).sum()),
        "n_adjusted_rows_market_a_night": int(
            (
                (adjusted["adjustment"].abs() > 1e-12)
                & (adjusted["market"].astype(str) == str(args.market_a_label))
                & adjusted["hour"].isin(market_a_night_hours)
            ).sum()
        ),
        "mean_adjustment": float(adjusted["adjustment"].mean()),
        "mean_abs_adjustment": float(adjusted["adjustment"].abs().mean()),
        "max_abs_adjustment_applied": float(adjusted["adjustment"].abs().max()),
    }
    (run_dir / "calibration_metadata.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.latest_copy:
        latest_path = Path(args.latest_copy)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(latest_path, index=False)
        print(f"Saved latest copy: {latest_path}")

    print(f"Source run: {source_run_dir}")
    print(f"Calibration run: {calibration_run_dir}")
    print(f"Saved submission: {out_path}")
    print(f"Saved calibration table: {run_dir / 'calibration_table.csv'}")
    print(f"Saved row adjustments: {run_dir / 'row_adjustments.csv'}")
    print(f"Saved metadata: {run_dir / 'calibration_metadata.json'}")
    print(
        "Adjustment summary: "
        f"rows_adjusted={stats['n_adjusted_rows']} "
        f"mean_abs_adjustment={stats['mean_abs_adjustment']:.6f} "
        f"max_abs_adjustment={stats['max_abs_adjustment_applied']:.6f}"
    )


if __name__ == "__main__":
    main()
