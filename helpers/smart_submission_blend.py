from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _resolve_submission_source(token: str, runs_dir: Path) -> Path:
    candidate = Path(token)
    if candidate.is_file():
        return candidate

    direct_run_submission = runs_dir / token / "submission.csv"
    if direct_run_submission.is_file():
        return direct_run_submission

    if candidate.suffix.lower() == ".csv":
        raise FileNotFoundError(f"Submission file not found: {token}")

    matches = [p for p in runs_dir.iterdir() if p.is_dir() and token in p.name]
    if not matches:
        raise FileNotFoundError(f"No run directory matches token '{token}' in {runs_dir}")
    if len(matches) > 1:
        names = ", ".join(sorted(p.name for p in matches))
        raise ValueError(f"Token '{token}' is ambiguous; matched: {names}")

    sub = matches[0] / "submission.csv"
    if not sub.is_file():
        raise FileNotFoundError(f"Matched run has no submission.csv: {matches[0]}")
    return sub


def _load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "target"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df[["id", "target"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["id", "target"]).copy()
    out["id"] = out["id"].astype(int)
    return out.sort_values("id").reset_index(drop=True)


def _parse_weights(raw: str | None, n: int, label: str) -> np.ndarray | None:
    if raw is None:
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != n:
        raise ValueError(f"{label} must provide exactly {n} weights; got {len(parts)}.")
    w = np.asarray([float(x) for x in parts], dtype=float)
    if (w < 0).any():
        raise ValueError(f"{label} weights must be non-negative.")
    if float(w.sum()) <= 0.0:
        raise ValueError(f"{label} weights must sum to a positive value.")
    return w / float(w.sum())


def _weighted_average(preds: list[np.ndarray], weights: np.ndarray | None) -> np.ndarray:
    if not preds:
        raise ValueError("No prediction arrays provided.")
    stack = np.column_stack(preds)
    if weights is None:
        w = np.full(stack.shape[1], 1.0 / float(stack.shape[1]), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
    return np.sum(stack * w, axis=1)


def _parse_peak_hours(raw: str) -> list[int]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--peak-hours must contain at least one hour.")
    out: list[int] = []
    for v in vals:
        h = int(v)
        if h < 0 or h > 23:
            raise ValueError(f"Invalid hour in --peak-hours: {h}")
        out.append(h)
    return sorted(set(out))


def _compute_market_hour_stats(
    train_df: pd.DataFrame,
    *,
    lower_q: float,
    upper_q: float,
) -> dict[str, Any]:
    work = train_df[["market", "delivery_start", "target"]].copy()
    work["delivery_start"] = pd.to_datetime(work["delivery_start"], errors="coerce")
    work["hour"] = work["delivery_start"].dt.hour
    work["market"] = work["market"].astype(str)
    work["target"] = pd.to_numeric(work["target"], errors="coerce")
    work = work.dropna(subset=["target", "hour"])
    if work.empty:
        raise ValueError("Training data has no valid rows for market-hour stats.")

    by_mh = work.groupby(["market", "hour"], dropna=False)["target"]
    by_m = work.groupby("market", dropna=False)["target"]

    mh_low = by_mh.quantile(lower_q).to_dict()
    mh_high = by_mh.quantile(upper_q).to_dict()
    mh_std = by_mh.std().fillna(0.0).to_dict()
    m_low = by_m.quantile(lower_q).to_dict()
    m_high = by_m.quantile(upper_q).to_dict()
    m_std = by_m.std().fillna(0.0).to_dict()

    g_low = float(work["target"].quantile(lower_q))
    g_high = float(work["target"].quantile(upper_q))
    g_std = float(work["target"].std())
    if not np.isfinite(g_std) or g_std <= 0.0:
        g_std = 1.0

    return {
        "market_hour_low": {(str(k[0]), int(k[1])): float(v) for k, v in mh_low.items()},
        "market_hour_high": {(str(k[0]), int(k[1])): float(v) for k, v in mh_high.items()},
        "market_hour_std": {(str(k[0]), int(k[1])): float(v) for k, v in mh_std.items()},
        "market_low": {str(k): float(v) for k, v in m_low.items()},
        "market_high": {str(k): float(v) for k, v in m_high.items()},
        "market_std": {str(k): float(v) for k, v in m_std.items()},
        "global_low": g_low,
        "global_high": g_high,
        "global_std": g_std,
    }


def _row_market_hour_arrays(test_df: pd.DataFrame, stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    market = test_df["market"].astype(str).to_numpy()
    hour = pd.to_datetime(test_df["delivery_start"], errors="coerce").dt.hour.fillna(-1).astype(int).to_numpy()
    keys = list(zip(market, hour))

    low = pd.Series(keys).map(stats["market_hour_low"]).to_numpy(dtype=float)
    high = pd.Series(keys).map(stats["market_hour_high"]).to_numpy(dtype=float)
    std = pd.Series(keys).map(stats["market_hour_std"]).to_numpy(dtype=float)

    m_low = pd.Series(market).map(stats["market_low"]).to_numpy(dtype=float)
    m_high = pd.Series(market).map(stats["market_high"]).to_numpy(dtype=float)
    m_std = pd.Series(market).map(stats["market_std"]).to_numpy(dtype=float)

    low = np.where(np.isnan(low), m_low, low)
    high = np.where(np.isnan(high), m_high, high)
    std = np.where(np.isnan(std), m_std, std)

    low = np.where(np.isnan(low), float(stats["global_low"]), low)
    high = np.where(np.isnan(high), float(stats["global_high"]), high)
    std = np.where(np.isnan(std), float(stats["global_std"]), std)
    std = np.where(std <= 1e-9, float(stats["global_std"]), std)

    bad = high < low
    if np.any(bad):
        tmp = low.copy()
        low[bad] = high[bad]
        high[bad] = tmp[bad]
    return low, high, std


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a robust competition blend from anchor/core/orthogonal submissions.\n"
            "Features: weighted blending, adaptive orthogonal injection, peak-hour boost, "
            "anchor-delta clipping, and market-hour quantile guardrails."
        )
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test_for_participants.csv")
    parser.add_argument("--anchor-source", required=True, help="Anchor submission path or run token.")
    parser.add_argument(
        "--core-sources",
        nargs="+",
        required=True,
        help="Core submissions (paths/tokens) blended with the anchor.",
    )
    parser.add_argument(
        "--ortho-sources",
        nargs="*",
        default=[],
        help="Optional orthogonal submissions for adaptive low-weight injection.",
    )
    parser.add_argument("--core-weights", default=None, help="Comma-separated weights for --core-sources.")
    parser.add_argument("--ortho-weights", default=None, help="Comma-separated weights for --ortho-sources.")
    parser.add_argument("--anchor-weight", type=float, default=0.65)
    parser.add_argument("--ortho-total-weight", type=float, default=0.08)
    parser.add_argument("--max-ortho-row-weight", type=float, default=0.20)
    parser.add_argument("--peak-hours", default="17,18,19,20")
    parser.add_argument("--peak-ortho-boost", type=float, default=1.35)
    parser.add_argument(
        "--disagreement-temperature",
        type=float,
        default=2.0,
        help="Higher means weaker down-weighting when ortho differs from base.",
    )
    parser.add_argument(
        "--delta-clip-mh-std",
        type=float,
        default=1.5,
        help="Clip (blended-anchor) to +/- K * market-hour train std (set <=0 to disable).",
    )
    parser.add_argument("--guardrail-lower-quantile", type=float, default=0.005)
    parser.add_argument("--guardrail-upper-quantile", type=float, default=0.995)
    parser.add_argument("--output", default="csv/submission_smart_blend.csv")
    parser.add_argument("--report-json", default=None, help="Optional diagnostics JSON output path.")
    args = parser.parse_args()

    if not (0.0 <= args.anchor_weight <= 1.0):
        raise ValueError("--anchor-weight must be in [0,1].")
    if not (0.0 <= args.ortho_total_weight <= 1.0):
        raise ValueError("--ortho-total-weight must be in [0,1].")
    if not (0.0 < args.guardrail_lower_quantile < 1.0):
        raise ValueError("--guardrail-lower-quantile must be in (0,1).")
    if not (0.0 < args.guardrail_upper_quantile < 1.0):
        raise ValueError("--guardrail-upper-quantile must be in (0,1).")
    if args.guardrail_lower_quantile >= args.guardrail_upper_quantile:
        raise ValueError("--guardrail-lower-quantile must be < --guardrail-upper-quantile.")
    if args.disagreement_temperature <= 0.0:
        raise ValueError("--disagreement-temperature must be > 0.")

    peak_hours = _parse_peak_hours(args.peak_hours)
    runs_dir = Path(args.runs_dir)
    train_path = Path(args.train_path)
    test_path = Path(args.test_path)
    out_path = Path(args.output)

    test_df = pd.read_csv(test_path)
    test_key = test_df[["id", "market", "delivery_start"]].copy()
    test_key["id"] = pd.to_numeric(test_key["id"], errors="coerce").astype("Int64")
    test_key = test_key.dropna(subset=["id"]).copy()
    test_key["id"] = test_key["id"].astype(int)
    test_key = test_key.sort_values("id").reset_index(drop=True)

    train_df = pd.read_csv(train_path)
    stats = _compute_market_hour_stats(
        train_df,
        lower_q=args.guardrail_lower_quantile,
        upper_q=args.guardrail_upper_quantile,
    )
    low, high, mh_std = _row_market_hour_arrays(test_key, stats)
    peak_mask = pd.to_datetime(test_key["delivery_start"], errors="coerce").dt.hour.isin(peak_hours).to_numpy()

    anchor_path = _resolve_submission_source(args.anchor_source, runs_dir)
    core_paths = [_resolve_submission_source(s, runs_dir) for s in args.core_sources]
    ortho_paths = [_resolve_submission_source(s, runs_dir) for s in args.ortho_sources]

    anchor = _load_submission(anchor_path)
    if not anchor["id"].equals(test_key["id"]):
        raise ValueError(f"ID mismatch between test file and anchor submission: {anchor_path}")
    anchor_pred = anchor["target"].to_numpy(dtype=float)

    core_weights = _parse_weights(args.core_weights, len(core_paths), "--core-weights")
    core_preds: list[np.ndarray] = []
    for p in core_paths:
        sub = _load_submission(p)
        if not sub["id"].equals(test_key["id"]):
            raise ValueError(f"ID mismatch for core submission: {p}")
        core_preds.append(sub["target"].to_numpy(dtype=float))
    core_blend = _weighted_average(core_preds, core_weights)

    base_pred = float(args.anchor_weight) * anchor_pred + (1.0 - float(args.anchor_weight)) * core_blend

    ortho_row_weight = np.zeros(len(base_pred), dtype=float)
    ortho_blend = np.zeros(len(base_pred), dtype=float)
    if ortho_paths and args.ortho_total_weight > 0.0:
        ortho_weights = _parse_weights(args.ortho_weights, len(ortho_paths), "--ortho-weights")
        ortho_preds: list[np.ndarray] = []
        for p in ortho_paths:
            sub = _load_submission(p)
            if not sub["id"].equals(test_key["id"]):
                raise ValueError(f"ID mismatch for ortho submission: {p}")
            ortho_preds.append(sub["target"].to_numpy(dtype=float))
        ortho_blend = _weighted_average(ortho_preds, ortho_weights)

        disagreement_z = np.abs(ortho_blend - base_pred) / (mh_std + 1e-9)
        disagree_gate = np.exp(-disagreement_z / float(args.disagreement_temperature))
        vol_scale = np.clip(mh_std / float(stats["global_std"]), 0.5, 2.0)
        peak_scale = np.where(peak_mask, float(args.peak_ortho_boost), 1.0)

        ortho_row_weight = float(args.ortho_total_weight) * vol_scale * peak_scale * disagree_gate
        ortho_row_weight = np.clip(ortho_row_weight, 0.0, float(args.max_ortho_row_weight))

    blended = (1.0 - ortho_row_weight) * base_pred + ortho_row_weight * ortho_blend

    # Keep blend close to anchor in each market-hour regime.
    anchor_delta_before = blended - anchor_pred
    clipped_by_delta = 0
    if args.delta_clip_mh_std > 0.0:
        cap = float(args.delta_clip_mh_std) * mh_std
        cap = np.maximum(cap, 1e-6)
        delta = np.clip(anchor_delta_before, -cap, cap)
        clipped_by_delta = int(np.count_nonzero(np.abs(delta - anchor_delta_before) > 1e-12))
        blended = anchor_pred + delta

    # Final hard guardrails from train market-hour quantile envelopes.
    guardrail_before = blended.copy()
    blended = np.clip(blended, low, high)
    guardrail_clipped = int(np.count_nonzero(np.abs(blended - guardrail_before) > 1e-12))

    out = pd.DataFrame({"id": test_key["id"], "target": blended})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    diagnostics = {
        "anchor_source": str(anchor_path),
        "core_sources": [str(p) for p in core_paths],
        "ortho_sources": [str(p) for p in ortho_paths],
        "rows": int(len(out)),
        "anchor_weight": float(args.anchor_weight),
        "ortho_total_weight": float(args.ortho_total_weight),
        "ortho_row_weight_mean": float(np.mean(ortho_row_weight)),
        "ortho_row_weight_p95": float(np.quantile(ortho_row_weight, 0.95)),
        "ortho_row_weight_peak_mean": float(np.mean(ortho_row_weight[peak_mask])) if np.any(peak_mask) else 0.0,
        "ortho_row_weight_offpeak_mean": float(np.mean(ortho_row_weight[~peak_mask])) if np.any(~peak_mask) else 0.0,
        "delta_clip_rows": clipped_by_delta,
        "guardrail_clip_rows": guardrail_clipped,
        "prediction_summary": {
            "mean": float(np.mean(blended)),
            "std": float(np.std(blended)),
            "p95": float(np.quantile(blended, 0.95)),
            "p99": float(np.quantile(blended, 0.99)),
            "max": float(np.max(blended)),
            "min": float(np.min(blended)),
        },
        "anchor_distance": {
            "corr": float(np.corrcoef(anchor_pred, blended)[0, 1]),
            "pred_rmse": float(np.sqrt(np.mean((blended - anchor_pred) ** 2))),
            "bias": float(np.mean(blended - anchor_pred)),
        },
    }

    print("Resolved sources:")
    print(f"- anchor: {anchor_path}")
    for i, p in enumerate(core_paths, start=1):
        print(f"- core_{i}: {p}")
    for i, p in enumerate(ortho_paths, start=1):
        print(f"- ortho_{i}: {p}")
    print(
        "Row-level ortho weight: "
        f"mean={diagnostics['ortho_row_weight_mean']:.4f}, "
        f"p95={diagnostics['ortho_row_weight_p95']:.4f}, "
        f"peak_mean={diagnostics['ortho_row_weight_peak_mean']:.4f}, "
        f"offpeak_mean={diagnostics['ortho_row_weight_offpeak_mean']:.4f}"
    )
    print(
        f"Delta-clipped rows={clipped_by_delta}, "
        f"guardrail-clipped rows={guardrail_clipped}"
    )
    print(
        "Prediction summary: "
        f"mean={diagnostics['prediction_summary']['mean']:.4f}, "
        f"std={diagnostics['prediction_summary']['std']:.4f}, "
        f"p99={diagnostics['prediction_summary']['p99']:.4f}, "
        f"max={diagnostics['prediction_summary']['max']:.4f}"
    )
    print(
        "Distance vs anchor: "
        f"corr={diagnostics['anchor_distance']['corr']:.4f}, "
        f"pred_rmse={diagnostics['anchor_distance']['pred_rmse']:.4f}, "
        f"bias={diagnostics['anchor_distance']['bias']:.4f}"
    )
    print(f"Saved blended submission: {out_path}")

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved diagnostics: {report_path}")


if __name__ == "__main__":
    main()
