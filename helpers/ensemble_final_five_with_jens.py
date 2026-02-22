from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCES = [
    "20260221-215400_jens",
    "20260222-175700_jens",
    "20260221-172305_per_market_interactions_2c02eb6_oct2023_cv1_14d",
    "20260222-083432_per_market_interactions_commit_baseline_multimodel_nocv",
    "20260221-201434_per_market_interactions_2c02eb6_oct2023_cv1_14d_temp_physics_pruned_nores",
]
DEFAULT_WEIGHTS = [0.35, 0.20, 0.25, 0.10, 0.10]


def _resolve_submission_source(token: str, runs_dir: Path) -> Path:
    candidate = Path(token)
    if candidate.exists():
        if candidate.is_file():
            return candidate
        sub = candidate / "submission.csv"
        if sub.is_file():
            return sub

    direct = runs_dir / token / "submission.csv"
    if direct.is_file():
        return direct

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


def _parse_weights(raw: str | None, n_sources: int) -> np.ndarray:
    if raw is None:
        if n_sources == len(DEFAULT_WEIGHTS):
            w = np.asarray(DEFAULT_WEIGHTS, dtype=float)
        else:
            w = np.full(n_sources, 1.0 / float(n_sources), dtype=float)
        return w / float(w.sum())

    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if len(vals) != n_sources:
        raise ValueError(f"--weights must have exactly {n_sources} values, got {len(vals)}")
    w = np.asarray([float(x) for x in vals], dtype=float)
    if np.any(w < 0.0):
        raise ValueError("--weights must be non-negative.")
    if float(w.sum()) <= 0.0:
        raise ValueError("--weights must sum to > 0.")
    return w / float(w.sum())


def _load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "target"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"{path} missing required columns: {missing}")
    out = df[["id", "target"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["id", "target"]).copy()
    out["id"] = out["id"].astype(int)
    return out.sort_values("id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the final weighted 5-run ensemble (including Jens runs) and save as a new run folder."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional source run tokens/paths. Default is the selected final 5.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Comma-separated weights aligned with --sources. "
            "If omitted, defaults to 0.35,0.20,0.25,0.10,0.10 for default sources."
        ),
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--sample-submission", default="data/sample_submission.csv")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--name", default="final_ensemble_five_with_jens")
    parser.add_argument("--latest-copy", default="csv/submission_final_ensemble_five_with_jens.csv")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate inputs without writing outputs.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    sources = list(DEFAULT_SOURCES if args.sources is None or len(args.sources) == 0 else args.sources)
    weights = _parse_weights(args.weights, len(sources))
    resolved_sources = [_resolve_submission_source(s, runs_dir) for s in sources]

    loaded = [_load_submission(p) for p in resolved_sources]
    base_ids = loaded[0]["id"]
    for path, df in zip(resolved_sources[1:], loaded[1:]):
        if len(df) != len(base_ids) or not df["id"].equals(base_ids):
            raise ValueError(f"ID mismatch in source {path}")

    pred_stack = np.column_stack([df["target"].to_numpy(dtype=float) for df in loaded])
    blended = np.sum(pred_stack * weights.reshape(1, -1), axis=1)

    sample = pd.read_csv(args.sample_submission, usecols=["id"])
    sample["id"] = pd.to_numeric(sample["id"], errors="coerce")
    sample = sample.dropna(subset=["id"]).copy()
    sample["id"] = sample["id"].astype(int)
    out = sample.merge(pd.DataFrame({"id": base_ids.to_numpy(), "target": blended}), on="id", how="left")
    if out["target"].isna().any():
        raise ValueError("NaN targets after mapping blended predictions to sample_submission ids.")

    print("Resolved sources:")
    for src, path, w in zip(sources, resolved_sources, weights):
        print(f"- {src} -> {path} | weight={w:.6f}")
    print(f"Rows: {len(out)}")
    print(f"Target mean={out['target'].mean():.6f} std={out['target'].std():.6f}")
    if args.dry_run:
        print("Dry run complete. No files were written.")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{args.name}"
    run_dir = Path(args.out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    submission_path = run_dir / "submission.csv"
    out.to_csv(submission_path, index=False)

    meta = {
        "run_id": run_id,
        "script": Path(__file__).name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources_raw": sources,
        "sources_resolved": [str(p) for p in resolved_sources],
        "weights": [float(x) for x in weights.tolist()],
        "n_rows": int(len(out)),
        "target_mean": float(out["target"].mean()),
        "target_std": float(out["target"].std()),
    }
    (run_dir / "ensemble_config.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.latest_copy:
        latest = Path(args.latest_copy)
        latest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(latest, index=False)
        print(f"Saved latest copy: {latest}")

    print(f"Saved submission: {submission_path}")
    print(f"Saved config: {run_dir / 'ensemble_config.json'}")


if __name__ == "__main__":
    main()
