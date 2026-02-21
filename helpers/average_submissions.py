from __future__ import annotations

import argparse
from pathlib import Path

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


def average_submissions(sources: list[Path]) -> pd.DataFrame:
    base_df: pd.DataFrame | None = None
    preds: list[pd.Series] = []

    for source in sources:
        df = pd.read_csv(source)
        required = {"id", "target"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"{source} is missing required columns: {sorted(missing)}")

        cur = df[["id", "target"]].copy()
        cur["id"] = cur["id"].astype(int)
        cur = cur.sort_values("id").reset_index(drop=True)

        if base_df is None:
            base_df = cur[["id"]].copy()
        else:
            if len(cur) != len(base_df) or not cur["id"].equals(base_df["id"]):
                raise ValueError(f"id mismatch in {source}; all submissions must share identical ids")

        preds.append(cur["target"].astype(float))

    if base_df is None:
        raise ValueError("No sources provided.")

    out = base_df.copy()
    out["target"] = pd.concat(preds, axis=1).mean(axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Average multiple submission files by id.")
    parser.add_argument(
        "sources",
        nargs="+",
        help=(
            "Submission file paths, run ids, or shorthand run tokens "
            "(e.g. runs/20260220-151106_x/submission.csv or 20-151106)."
        ),
    )
    parser.add_argument("--runs-dir", default="runs", help="Runs directory used for token lookup.")
    parser.add_argument("--output", default="csv/submission_average.csv", help="Output CSV path.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    resolved_sources = [_resolve_submission_source(s, runs_dir) for s in args.sources]
    print("Resolved sources:")
    for src in resolved_sources:
        print(f"- {src}")

    avg = average_submissions(resolved_sources)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    avg.to_csv(out_path, index=False)
    print(f"Saved averaged submission: {out_path}")
    print(f"Rows: {len(avg)}")


if __name__ == "__main__":
    main()
