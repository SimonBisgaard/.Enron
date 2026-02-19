from pathlib import Path
import argparse
import csv
import subprocess


def _run_git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")


def _find_run(registry: Path, run_id: str) -> dict[str, str]:
    if not registry.exists():
        raise FileNotFoundError(f"Registry file not found: {registry}")

    with registry.open("r", newline="", encoding="utf-8") as file_handle:
        for row in csv.DictReader(file_handle):
            if row.get("run_id") == run_id:
                return row

    raise ValueError(f"Run ID not found: {run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag the commit associated with a run_id from experiments.csv")
    parser.add_argument("--run-id", required=True, help="Run ID present in experiments.csv")
    parser.add_argument("--tag", required=True, help="Tag name, e.g. exp/best-cv-123")
    parser.add_argument("--registry", default="experiments.csv", help="Path to experiments registry CSV")
    parser.add_argument("--push", action="store_true", help="Also push tag to origin")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    registry_path = (repo_root / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry)

    run_row = _find_run(registry_path, args.run_id)
    git_sha = run_row.get("git_sha", "").strip()
    if not git_sha:
        raise ValueError(f"Run {args.run_id} has empty git_sha in registry")

    message = f"run_id={args.run_id}, cv_rmse={run_row.get('cv_rmse', '')}, lb_score={run_row.get('lb_score', '')}"
    _run_git(["tag", "-a", args.tag, git_sha, "-m", message], repo_root)

    if args.push:
        _run_git(["push", "origin", args.tag], repo_root)

    print(f"Created tag {args.tag} -> {git_sha}")
    if args.push:
        print(f"Pushed tag {args.tag} to origin")


if __name__ == "__main__":
    main()
