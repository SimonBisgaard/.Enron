Train: uv run python train.py --config-name per_market_v1
Strict official run (must be clean git): uv run python train.py --official-run
Add LB score later: uv run python helpers/update_lb_score.py --run-id <run_id> --lb-score <score>
Tag best run commit: uv run python helpers/tag_run.py --run-id <run_id> --tag exp/best-cv-xx --push
