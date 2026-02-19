from pathlib import Path
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_squared_error

LAG_STEPS = [1, 2, 24, 48, 168]
ROLL_WINDOWS = [24, 48, 168]


def _add_cyclical_features(df: pd.DataFrame, column: str, period: int) -> None:
	radians = 2.0 * np.pi * (df[column] / period)
	df[f"{column}_sin"] = np.sin(radians)
	df[f"{column}_cos"] = np.cos(radians)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
	result = df.copy()
	start_ts = pd.to_datetime(result["delivery_start"], errors="coerce")
	end_ts = pd.to_datetime(result["delivery_end"], errors="coerce")

	for col, ts in [("delivery_start", start_ts), ("delivery_end", end_ts)]:
		result[f"{col}_hour"] = ts.dt.hour
		result[f"{col}_day"] = ts.dt.day
		result[f"{col}_month"] = ts.dt.month
		result[f"{col}_week"] = ts.dt.isocalendar().week.astype("float64")
		result[f"{col}_dow"] = ts.dt.dayofweek
		result[f"{col}_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
		_add_cyclical_features(result, f"{col}_hour", 24)
		_add_cyclical_features(result, f"{col}_dow", 7)

	result["delivery_duration_hours"] = (end_ts - start_ts).dt.total_seconds() / 3600.0
	result["delivery_start_ts"] = start_ts.astype("int64") // 10**9

	if {"air_temperature_2m", "dew_point_temperature_2m"}.issubset(result.columns):
		result["temp_dew_spread"] = result["air_temperature_2m"] - result["dew_point_temperature_2m"]

	if {"air_temperature_2m", "apparent_temperature_2m"}.issubset(result.columns):
		result["temp_apparent_spread"] = result["air_temperature_2m"] - result["apparent_temperature_2m"]

	if {"wind_speed_80m", "wind_speed_10m"}.issubset(result.columns):
		result["wind_speed_ratio_80m_10m"] = result["wind_speed_80m"] / (result["wind_speed_10m"].abs() + 1e-3)

	if {
		"global_horizontal_irradiance",
		"diffuse_horizontal_irradiance",
		"direct_normal_irradiance",
	}.issubset(result.columns):
		result["irradiance_total"] = (
			result["global_horizontal_irradiance"]
			+ result["diffuse_horizontal_irradiance"]
			+ result["direct_normal_irradiance"]
		)

	if {"cloud_cover_total", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"}.issubset(result.columns):
		result["cloud_cover_layers_sum"] = (
			result["cloud_cover_low"] + result["cloud_cover_mid"] + result["cloud_cover_high"]
		)
		result["cloud_cover_gap"] = result["cloud_cover_total"] - result["cloud_cover_layers_sum"]

	return result.drop(columns=["delivery_start", "delivery_end"])


def add_market_lag_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	train_with_lags = train_df.copy()
	test_with_lags = test_df.copy()

	train_with_lags = train_with_lags.sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)
	test_with_lags = test_with_lags.sort_values(["market", "delivery_start_ts", "id"]).reset_index(drop=True)

	train_grouped = train_with_lags.groupby("market", sort=False)

	for lag in LAG_STEPS:
		train_with_lags[f"target_lag_{lag}"] = train_grouped["target"].shift(lag)
		test_with_lags[f"target_lag_{lag}"] = np.nan

	target_shifted = train_grouped["target"].shift(1)
	for window in ROLL_WINDOWS:
		train_with_lags[f"target_roll_mean_{window}"] = (
			target_shifted.groupby(train_with_lags["market"]).rolling(window, min_periods=4).mean().reset_index(level=0, drop=True)
		)
		train_with_lags[f"target_roll_std_{window}"] = (
			target_shifted.groupby(train_with_lags["market"]).rolling(window, min_periods=4).std().reset_index(level=0, drop=True)
		)
		test_with_lags[f"target_roll_mean_{window}"] = np.nan
		test_with_lags[f"target_roll_std_{window}"] = np.nan

	if "load_forecast" in train_with_lags.columns:
		train_load_lag_1 = train_grouped["load_forecast"].shift(1)
		train_load_lag_24 = train_grouped["load_forecast"].shift(24)
		train_with_lags["load_forecast_delta_1"] = train_with_lags["load_forecast"] - train_load_lag_1
		train_with_lags["load_forecast_delta_24"] = train_with_lags["load_forecast"] - train_load_lag_24

		test_grouped = test_with_lags.groupby("market", sort=False)
		test_load_lag_1 = test_grouped["load_forecast"].shift(1)
		test_load_lag_24 = test_grouped["load_forecast"].shift(24)
		test_with_lags["load_forecast_delta_1"] = test_with_lags["load_forecast"] - test_load_lag_1
		test_with_lags["load_forecast_delta_24"] = test_with_lags["load_forecast"] - test_load_lag_24

	train_with_lags = train_with_lags.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
	test_with_lags = test_with_lags.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)

	return train_with_lags, test_with_lags


def recursive_market_test_predictions(
	normal_model: CatBoostRegressor,
	spike_model: CatBoostRegressor,
	spike_classifier: CatBoostClassifier | None,
	x_test_market_base: pd.DataFrame,
	history_targets: list[float],
	feature_cols: list[str],
	num_fill_values: dict[str, float],
) -> pd.Series:
	market_preds: list[float] = []

	for row_idx in x_test_market_base.index:
		row_features = x_test_market_base.loc[row_idx, feature_cols].copy()

		for lag in LAG_STEPS:
			lag_value = history_targets[-lag] if len(history_targets) >= lag else np.nan
			row_features[f"target_lag_{lag}"] = lag_value

		for window in ROLL_WINDOWS:
			if len(history_targets) == 0:
				row_features[f"target_roll_mean_{window}"] = np.nan
				row_features[f"target_roll_std_{window}"] = np.nan
				continue

			tail = np.array(history_targets[-window:], dtype=float)
			row_features[f"target_roll_mean_{window}"] = float(np.mean(tail))
			row_features[f"target_roll_std_{window}"] = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0

		row_df = pd.DataFrame([row_features], columns=feature_cols)
		row_idx = row_df.index[0]
		for col, fill_value in num_fill_values.items():
			if pd.isna(row_df.at[row_idx, col]):
				row_df.at[row_idx, col] = fill_value

		normal_pred = float(normal_model.predict(row_df)[0])
		spike_pred = float(spike_model.predict(row_df)[0])

		if spike_classifier is not None:
			p_spike = float(spike_classifier.predict_proba(row_df)[0, 1])
		else:
			p_spike = 0.15

		blend_pred = (1.0 - p_spike) * normal_pred + p_spike * spike_pred
		history_targets.append(blend_pred)
		market_preds.append(blend_pred)

	return pd.Series(market_preds, index=x_test_market_base.index, dtype=float)


def make_timestamp_folds(
	df: pd.DataFrame,
	n_splits: int,
	purge_timestamps: int,
	min_train_timestamps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
	unique_ts = np.array(sorted(df["delivery_start_ts"].unique()))
	if len(unique_ts) < (n_splits + 2):
		raise ValueError("Not enough unique timestamps for requested number of folds.")

	val_size = max(24, len(unique_ts) // (n_splits + 1))
	folds: list[tuple[np.ndarray, np.ndarray]] = []

	for fold in range(n_splits):
		train_end = val_size * (fold + 1)
		if train_end < min_train_timestamps:
			continue

		valid_start = train_end + purge_timestamps
		valid_end = min(valid_start + val_size, len(unique_ts))
		if valid_end <= valid_start:
			break

		train_ts = unique_ts[:train_end]
		valid_ts = unique_ts[valid_start:valid_end]

		train_idx = df.index[df["delivery_start_ts"].isin(train_ts)].to_numpy()
		valid_idx = df.index[df["delivery_start_ts"].isin(valid_ts)].to_numpy()

		if len(train_idx) == 0 or len(valid_idx) == 0:
			continue
		folds.append((train_idx, valid_idx))

	if not folds:
		raise ValueError("No valid timestamp folds were created. Lower min_train_timestamps or purge_timestamps.")

	return folds


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	data_dir = base_dir / "data"
	fast_mode = os.getenv("FAST_MODE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
	mode_label = "FAST" if fast_mode else "FULL"
	print(f"Running in {mode_label} mode")

	train_path = data_dir / "train.csv"
	test_path = data_dir / "test_for_participants.csv"
	sample_submission_path = data_dir / "sample_submission.csv"
	output_path = base_dir / "submission.csv"

	for path in [train_path, test_path, sample_submission_path]:
		if not path.exists():
			raise FileNotFoundError(f"Missing file: {path}")

	train_df = pd.read_csv(train_path)
	test_df = pd.read_csv(test_path)
	sample_submission = pd.read_csv(sample_submission_path)

	train_df = add_time_features(train_df)
	test_df = add_time_features(test_df)

	train_df = train_df.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
	test_df = test_df.sort_values(["delivery_start_ts", "market", "id"]).reset_index(drop=True)
	train_df, test_df = add_market_lag_features(train_df, test_df)

	feature_cols = [
		col for col in train_df.columns if col not in ["id", "target", "market"]
	]
	cat_cols = train_df[feature_cols].select_dtypes(include=["object", "category", "string"]).columns.tolist()
	num_cols = [col for col in feature_cols if col not in cat_cols]

	constant_cols = [col for col in feature_cols if train_df[col].nunique(dropna=False) <= 1]
	if constant_cols:
		print(f"Dropping {len(constant_cols)} constant features")
		train_df = train_df.drop(columns=constant_cols)
		test_df = test_df.drop(columns=constant_cols)
		feature_cols = [col for col in feature_cols if col not in constant_cols]
		cat_cols = [col for col in cat_cols if col not in constant_cols]
		num_cols = [col for col in feature_cols if col not in cat_cols]

	for col in num_cols:
		market_median = train_df.groupby("market")[col].median()
		train_df[col] = train_df[col].fillna(train_df["market"].map(market_median))
		test_df[col] = test_df[col].fillna(test_df["market"].map(market_median))
		global_median = train_df[col].median()
		train_df[col] = train_df[col].fillna(global_median)
		test_df[col] = test_df[col].fillna(global_median)

	for col in cat_cols:
		train_df[col] = train_df[col].fillna("missing").astype(str)
		test_df[col] = test_df[col].fillna("missing").astype(str)

	print(f"Using {len(feature_cols)} total features")
	print(f"Categorical ({len(cat_cols)}): {cat_cols}")
	print(f"Numerical ({len(num_cols)}): {num_cols}")

	x_train_full = train_df[feature_cols].copy()
	y_train_full = train_df["target"].copy()
	x_test_full = test_df[feature_cols].copy()
	num_fill_values = {col: float(train_df[col].median()) for col in num_cols}

	markets = sorted(train_df["market"].dropna().unique())
	train_market_indices = {
		market: train_df.index[train_df["market"] == market].to_numpy() for market in markets
	}
	test_market_indices = {
		market: test_df.index[test_df["market"] == market].to_numpy() for market in markets
	}
	market_train_target_history = {
		market: y_train_full.iloc[train_market_indices[market]].astype(float).tolist() for market in markets
	}
	market_fallback_mean = train_df.groupby("market")["target"].mean().to_dict()
	global_target_mean = float(train_df["target"].mean())

	n_splits = 3 if fast_mode else 5
	purge_timestamps = 24 if fast_mode else 48
	min_train_timestamps = 24 * 7 if fast_mode else 24 * 14
	folds = make_timestamp_folds(
		df=train_df,
		n_splits=n_splits,
		purge_timestamps=purge_timestamps,
		min_train_timestamps=min_train_timestamps,
	)
	print(f"Using {len(folds)} timestamp folds with purge gap = {purge_timestamps} timestamps")

	seeds = [42] if fast_mode else [42, 2024]
	full_model_configs = [
		{
			"name": "rmse_depth7",
			"params": {
				"loss_function": "RMSE",
				"eval_metric": "RMSE",
				"iterations": 2600,
				"learning_rate": 0.03,
				"depth": 7,
				"l2_leaf_reg": 22,
				"bagging_temperature": 0.5,
				"random_strength": 0.9,
				"verbose": 0,
			},
		},
		{
			"name": "mae_depth8",
			"params": {
				"loss_function": "MAE",
				"eval_metric": "RMSE",
				"iterations": 3600,
				"learning_rate": 0.02,
				"depth": 8,
				"l2_leaf_reg": 28,
				"bagging_temperature": 0.7,
				"random_strength": 1.1,
				"verbose": 0,
			},
		},
	]
	fast_model_configs = [
		{
			"name": "rmse_fast",
			"params": {
				"loss_function": "RMSE",
				"eval_metric": "RMSE",
				"iterations": 900,
				"learning_rate": 0.05,
				"depth": 6,
				"l2_leaf_reg": 18,
				"bagging_temperature": 0.5,
				"random_strength": 0.8,
				"verbose": 0,
			},
		}
	]
	model_configs = fast_model_configs if fast_mode else full_model_configs

	oof_sum = np.zeros(len(train_df), dtype=float)
	oof_weight = np.zeros(len(train_df), dtype=float)
	test_sum = np.zeros(len(test_df), dtype=float)
	test_weight = 0.0
	fold_rmses: list[float] = []
	model_level_scores: list[tuple[str, int, float]] = []

	for config in model_configs:
		config_name = config["name"]
		config_params = config["params"]
		for seed in seeds:
			print(f"\n=== Config {config_name} | seed {seed} ===")
			seed_fold_scores: list[float] = []
			for fold_num, (train_idx_all, valid_idx_all) in enumerate(folds, start=1):
				valid_pred_series = pd.Series(index=valid_idx_all, dtype=float)
				test_pred_fold = pd.Series(index=test_df.index, dtype=float)

				for market in markets:
					market_train_idx = np.intersect1d(train_idx_all, train_market_indices[market], assume_unique=False)
					market_valid_idx = np.intersect1d(valid_idx_all, train_market_indices[market], assume_unique=False)
					market_test_idx = test_market_indices.get(market, np.array([], dtype=int))

					if len(market_train_idx) < 200 or len(market_valid_idx) == 0:
						continue

					x_train = x_train_full.iloc[market_train_idx]
					y_train = y_train_full.iloc[market_train_idx]
					x_valid = x_train_full.iloc[market_valid_idx]
					x_test_market = x_test_full.iloc[market_test_idx]

					normal_model = CatBoostRegressor(
						random_seed=seed,
						**config_params,
					)
					normal_model.fit(
						x_train,
						y_train,
						cat_features=cat_cols,
						use_best_model=False,
					)

					spike_threshold = float(np.quantile(np.abs(y_train.to_numpy()), 0.9))
					spike_mask = (np.abs(y_train.to_numpy()) >= spike_threshold).astype(int)
					spike_weights = np.where(spike_mask == 1, 4.0, 1.0)

					spike_model = CatBoostRegressor(
						random_seed=seed + 13,
						**config_params,
					)
					spike_model.fit(
						x_train,
						y_train,
						cat_features=cat_cols,
						sample_weight=spike_weights,
						use_best_model=False,
					)

					spike_classifier: CatBoostClassifier | None = None
					if spike_mask.sum() > 20 and spike_mask.sum() < len(spike_mask) - 20:
						spike_classifier = CatBoostClassifier(
							loss_function="Logloss",
							iterations=300 if fast_mode else 700,
							learning_rate=0.04,
							depth=6,
							l2_leaf_reg=16,
							random_seed=seed + 101,
							verbose=0,
						)
						spike_classifier.fit(
							x_train,
							spike_mask,
							cat_features=cat_cols,
						)
						p_spike_valid = spike_classifier.predict_proba(x_valid)[:, 1]
					else:
						p_spike_valid = np.full(len(x_valid), 0.15)

					normal_valid = normal_model.predict(x_valid)
					spike_valid = spike_model.predict(x_valid)
					blend_valid = (1.0 - p_spike_valid) * normal_valid + p_spike_valid * spike_valid

					valid_pred_series.loc[market_valid_idx] = blend_valid

					if len(market_test_idx) > 0:
						history_targets = market_train_target_history[market].copy()
						recursive_preds = recursive_market_test_predictions(
							normal_model=normal_model,
							spike_model=spike_model,
							spike_classifier=spike_classifier,
							x_test_market_base=x_test_market,
							history_targets=history_targets,
							feature_cols=feature_cols,
							num_fill_values=num_fill_values,
						)
						test_pred_fold.loc[market_test_idx] = recursive_preds.values

				valid_pred_series = valid_pred_series.dropna()
				if valid_pred_series.empty:
					continue

				y_valid = y_train_full.loc[valid_pred_series.index]
				fold_rmse = mean_squared_error(y_valid, valid_pred_series.values) ** 0.5
				fold_weight = 1.0 / (fold_rmse + 1e-6)

				oof_sum[valid_pred_series.index.to_numpy()] += valid_pred_series.values * fold_weight
				oof_weight[valid_pred_series.index.to_numpy()] += fold_weight

				if test_pred_fold.isna().any():
					missing_idx = test_pred_fold.index[test_pred_fold.isna()].to_numpy()
					fallback = test_df.loc[missing_idx, "market"].map(market_fallback_mean).fillna(global_target_mean)
					test_pred_fold.loc[missing_idx] = fallback.values

				test_sum += test_pred_fold.values * fold_weight
				test_weight += fold_weight

				fold_rmses.append(fold_rmse)
				seed_fold_scores.append(fold_rmse)
				print(
					f"Config {config_name} | Seed {seed} | Fold {fold_num}/{len(folds)} RMSE: {fold_rmse:.6f}"
				)

			if seed_fold_scores:
				seed_mean_rmse = float(np.mean(seed_fold_scores))
				model_level_scores.append((config_name, seed, seed_mean_rmse))
				print(f"Config {config_name} | Seed {seed} mean RMSE: {seed_mean_rmse:.6f}")

	valid_mask = oof_weight > 0
	if not np.any(valid_mask):
		raise ValueError("No OOF predictions available.")
	oof_pred = np.zeros_like(oof_sum)
	oof_pred[valid_mask] = oof_sum[valid_mask] / oof_weight[valid_mask]

	if test_weight <= 0:
		raise ValueError("No test prediction weights accumulated.")
	test_pred = test_sum / test_weight

	cv_rmse = mean_squared_error(y_train_full[valid_mask], oof_pred[valid_mask]) ** 0.5
	recent_fold_mean = float(np.mean(fold_rmses[-2:])) if len(fold_rmses) >= 2 else float(np.mean(fold_rmses))

	print(f"\nCV RMSE (OOF): {cv_rmse:.6f}")
	print(f"Fold RMSE mean: {np.mean(fold_rmses):.6f} | std: {np.std(fold_rmses):.6f}")
	print(f"Recent fold RMSE mean (last 2): {recent_fold_mean:.6f}")
	print(f"OOF coverage: {valid_mask.mean() * 100:.2f}%")

	if model_level_scores:
		print("Model-level mean RMSE:")
		for config_name, seed, score in sorted(model_level_scores, key=lambda x: x[2]):
			print(f"  {config_name} | seed {seed}: {score:.6f}")

	test_id_to_pred = pd.Series(test_pred, index=test_df["id"].values)
	submission = sample_submission[["id"]].copy()
	submission["target"] = submission["id"].map(test_id_to_pred)

	if submission["target"].isna().any():
		raise ValueError("Some submission IDs did not receive predictions.")

	submission.to_csv(output_path, index=False)
	print(f"Saved upload-ready submission: {output_path}")
	print(submission.head())


if __name__ == "__main__":
	main()
