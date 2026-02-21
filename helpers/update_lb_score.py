from pathlib import Path
import argparse
import csv


def update_lb_score(registry_path: Path, run_id: str, lb_score: float) -> None:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    with registry_path.open("r", newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
        fieldnames = rows[0].keys() if rows else [
            "run_id",
            "git_sha",
            "git_branch",
            "dirty_repo",
            "data_hash",
            "config_hash",
            "seed",
            "cv_rmse",
            "lb_score",
            "model_path",
            "submission_path",
            "started_at",
        ]

    updated = False
    for row in rows:
        if row.get("run_id") == run_id:
            row["lb_score"] = str(lb_score)
            updated = True
            break

    if not updated:
        raise ValueError(f"Run ID not found in registry: {run_id}")

    with registry_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated lb_score for run_id={run_id} to {lb_score}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update leaderboard score for an existing run in csv/experiments.csv")
    parser.add_argument("--run-id", required=True, help="Run ID to update")
    parser.add_argument("--lb-score", required=True, type=float, help="Leaderboard score value")
    parser.add_argument(
        "--registry",
        default="csv/experiments.csv",
        help="Path to experiments registry CSV (default: csv/experiments.csv)",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry).resolve()
    update_lb_score(registry_path=registry_path, run_id=args.run_id, lb_score=args.lb_score)


if __name__ == "__main__":
    main()
