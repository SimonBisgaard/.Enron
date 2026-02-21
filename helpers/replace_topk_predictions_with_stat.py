from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    df = pd.read_csv(path)
    required = {"id", "target"}
    if not required.issubset(df.columns):
        raise ValueError(f"Submission missing required columns {required}: {path}")

    out = df[["id", "target"]].copy()
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["target"] = pd.to_numeric(out["target"], errors="coerce")
    out = out.dropna(subset=["id", "target"]).copy()
    out["id"] = out["id"].astype(int)
    return out.reset_index(drop=True)


def _find_run_submission(runs_dir: Path, token: str) -> tuple[str, Path]:
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and (p / "submission.csv").exists()]
    candidates.sort(key=lambda p: p.name)
    if not candidates:
        raise ValueError(f"No run folders with submission.csv found in: {runs_dir}")

    exact = [p for p in candidates if p.name == token]
    if len(exact) == 1:
        run_dir = exact[0]
        return run_dir.name, run_dir / "submission.csv"
    if len(exact) > 1:
        raise ValueError(f"Multiple exact run matches for token={token}")

    partial = [p for p in candidates if token in p.name]
    if len(partial) == 1:
        run_dir = partial[0]
        return run_dir.name, run_dir / "submission.csv"
    if not partial:
        raise ValueError(f"No run id contains token={token}")
    raise ValueError(f"Token={token} is ambiguous: {[p.name for p in partial]}")


def _topk_indices(values: np.ndarray, top_k: int, mode: str) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if top_k > len(values):
        raise ValueError(f"--top-k={top_k} exceeds number of rows ({len(values)})")

    if mode == "largest":
        return np.argpartition(values, -top_k)[-top_k:]
    if mode == "smallest":
        return np.argpartition(values, top_k)[:top_k]
    if mode == "abs":
        return np.argpartition(np.abs(values), -top_k)[-top_k:]
    raise ValueError(f"Unsupported mode: {mode}")


def _replacement_value(values: np.ndarray, idx: np.ndarray, method: str, const_value: float) -> float:
    mask = np.ones(len(values), dtype=bool)
    mask[idx] = False

    if method == "mean_non_topk":
        if not np.any(mask):
            raise ValueError("No non-top-k rows left for mean_non_topk.")
        return float(np.mean(values[mask]))
    if method == "median_non_topk":
        if not np.any(mask):
            raise ValueError("No non-top-k rows left for median_non_topk.")
        return float(np.median(values[mask]))
    if method == "mean_all":
        return float(np.mean(values))
    if method == "median_all":
        return float(np.median(values))
    if method == "value":
        return float(const_value)
    raise ValueError(f"Unsupported replacement method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace top-k predictions in a submission with a central statistic "
            "(useful for checking whether extreme peaks help or hurt)."
        )
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--run-id",
        default="151106",
        help="Run id token used to locate runs/<run-id>/submission.csv (default: 151106).",
    )
    parser.add_argument(
        "--submission-path",
        default=None,
        help="Optional direct path to submission CSV. Overrides --run-id lookup.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of rows to replace.")
    parser.add_argument(
        "--mode",
        choices=["largest", "smallest", "abs"],
        default="largest",
        help="How to pick top-k rows (default: largest).",
    )
    parser.add_argument(
        "--replace-with",
        choices=["mean_non_topk", "median_non_topk", "mean_all", "median_all", "value"],
        default="mean_non_topk",
        help="Replacement strategy (default: mean_non_topk).",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.0,
        help="Constant replacement value when --replace-with value.",
    )
    parser.add_argument(
        "--out-path",
        default=None,
        help="Output path. Default: beside source submission with descriptive suffix.",
    )
    args = parser.parse_args()

    run_id = "direct_path"
    if args.submission_path:
        src_path = Path(args.submission_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Submission file not found: {src_path}")
    else:
        run_id, src_path = _find_run_submission(Path(args.runs_dir), args.run_id)

    sub = _load_submission(src_path)
    vals = sub["target"].to_numpy(dtype=float)

    idx = _topk_indices(vals, args.top_k, args.mode)
    replacement = _replacement_value(vals, idx, args.replace_with, args.value)

    out = sub.copy()
    out.loc[idx, "target"] = replacement

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        suffix = f"submission_top{args.top_k}_{args.mode}_to_{args.replace_with}.csv"
        out_path = src_path.with_name(suffix)

    out = out.sort_values("id").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    changed = sub.loc[idx, ["id", "target"]].copy()
    changed["new_target"] = replacement
    changed = changed.sort_values("target", ascending=False).reset_index(drop=True)

    print(f"Run/source: {run_id}")
    print(f"Input submission: {src_path}")
    print(f"Rows replaced: {len(changed)}")
    print(f"Selection mode: {args.mode}")
    print(f"Replacement method: {args.replace_with}")
    print(f"Replacement value: {replacement:.6f}")
    print("Changed rows (id, old_target -> new_target):")
    for row in changed.itertuples(index=False):
        print(f"  {int(row.id)}: {float(row.target):.6f} -> {float(row.new_target):.6f}")
    print(f"Saved submission: {out_path}")


if __name__ == "__main__":
    main()
