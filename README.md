Train: uv run python [train.py](http://_vscodecontentref_/3) --config-name per_market_v1
Strict official run (must be clean git): uv run python [train.py](http://_vscodecontentref_/4) --official-run
Add LB score later: uv run python [update_lb_score.py](http://_vscodecontentref_/5) --run-id <run_id> --lb-score <score>
Tag best run commit: uv run python [tag_run.py](http://_vscodecontentref_/6) --run-id <run_id> --tag exp/best-cv-xx --push