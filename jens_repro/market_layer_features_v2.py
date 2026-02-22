from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6
Z_MIN_STD = 1e-3
Z_CLIP = 5.0

XMK_GLOBAL_BASE_VARS: tuple[str, ...] = (
    "residual_load",
    "oversupply_share",
    "solar_share",
    "wind_share",
    "load_forecast",
    "wind_forecast",
    "solar_forecast",
)

METEO_CONVERSION_BASE_FEATURES: tuple[str, ...] = (
    "ws80_cu",
    "gustiness_10m",
    "gust_ratio_10m",
    "ws_ratio_80_10",
    "wind_dir_sin",
    "wind_dir_cos",
    "clear_sky_proxy",
    "diffuse_share",
    "cloud_low_share",
    "cloud_x_solar_day",
    "temp_dew_spread",
    "temp_apparent_gap",
    "heating_degree",
    "cooling_degree",
    "storm_risk",
    "storm_flag",
)

SLOW_WINDOWS: tuple[int, ...] = (24, 168, 720)


@dataclass
class RegimeThresholds:
    lowprice_residual_q25_by_market: dict[str, float]
    lowprice_solar_q75_by_market: dict[str, float]
    transition_ramp_q85_by_market: dict[str, float]
    transition_std_q85_by_market: dict[str, float]
    peak_residual_q80_by_market: dict[str, float]
    peak_windshare_q20_by_market: dict[str, float]
    peak_stress_q80_by_market: dict[str, float]
    apply_peak_stress_filter_by_market: dict[str, bool]
    storm_mode: str
    storm_cin_q20_by_market: dict[str, float]
    storm_cin_q20_global: float
    storm_risk_q90_global: float
    global_defaults: dict[str, float]
    available_meteo_features: list[str] = field(default_factory=list)
    local_xmk_base_vars: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lowprice_residual_q25_by_market": self.lowprice_residual_q25_by_market,
            "lowprice_solar_q75_by_market": self.lowprice_solar_q75_by_market,
            "transition_ramp_q85_by_market": self.transition_ramp_q85_by_market,
            "transition_std_q85_by_market": self.transition_std_q85_by_market,
            "peak_residual_q80_by_market": self.peak_residual_q80_by_market,
            "peak_windshare_q20_by_market": self.peak_windshare_q20_by_market,
            "peak_stress_q80_by_market": self.peak_stress_q80_by_market,
            "apply_peak_stress_filter_by_market": self.apply_peak_stress_filter_by_market,
            "storm_mode": self.storm_mode,
            "storm_cin_q20_by_market": self.storm_cin_q20_by_market,
            "storm_cin_q20_global": self.storm_cin_q20_global,
            "storm_risk_q90_global": self.storm_risk_q90_global,
            "global_defaults": self.global_defaults,
            "available_meteo_features": self.available_meteo_features,
            "local_xmk_base_vars": self.local_xmk_base_vars,
        }


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.astype(float) / np.maximum(den.astype(float), EPS)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["delivery_start"], errors="coerce")
    out["hour"] = out["ts"].dt.hour
    out["dow"] = out["ts"].dt.dayofweek
    out["month"] = out["ts"].dt.month
    out["weekofyear"] = out["ts"].dt.isocalendar().week.astype(int)

    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["is_midday"] = out["hour"].isin([10, 11, 12, 13, 14, 15, 16]).astype(int)
    out["is_peak_17_20"] = out["hour"].isin([17, 18, 19, 20]).astype(int)
    out["is_night"] = (out["solar_forecast"] <= 0).astype(int)
    out["is_winter"] = out["month"].isin([11, 12, 1, 2]).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    out["market_hour"] = out["market"].astype(str) + "_" + out["hour"].astype(str)
    out["market_dow"] = out["market"].astype(str) + "_" + out["dow"].astype(str)
    out["market_month"] = out["market"].astype(str) + "_" + out["month"].astype(str)
    return out


def _add_system_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom = np.maximum(out["load_forecast"].to_numpy(dtype=float), EPS)

    out["residual_load"] = out["load_forecast"] - out["wind_forecast"] - out["solar_forecast"]
    out["oversupply"] = (out["wind_forecast"] + out["solar_forecast"]) - out["load_forecast"]
    out["oversupply_pos"] = np.clip(out["oversupply"], a_min=0.0, a_max=None)
    out["oversupply_share"] = out["oversupply_pos"] / denom
    out["wind_share"] = out["wind_forecast"] / denom
    out["solar_share"] = out["solar_forecast"] / denom
    out["net_load_share"] = out["residual_load"] / denom
    out["stress_ratio"] = out["load_forecast"] / (out["wind_forecast"] + 1000.0)
    out["residual_load_sq"] = out["residual_load"] ** 2
    out["stress_ratio_sq"] = out["stress_ratio"] ** 2

    out["residual_load_pos"] = np.clip(out["residual_load"], a_min=0.0, a_max=None)
    out["residual_load_neg"] = np.clip(-out["residual_load"], a_min=0.0, a_max=None)
    out["oversupply_share_sq"] = out["oversupply_share"] ** 2
    out["solar_share_x_is_midday"] = out["solar_share"] * out["is_midday"]
    out["stress_x_is_peak"] = out["stress_ratio"] * out["is_peak_17_20"]
    return out


def _add_transition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["market", "ts"], kind="mergesort")

    grp = out.groupby("market", sort=False)
    out["residual_load_lag_1"] = grp["residual_load"].shift(1)
    out["residual_load_diff_1"] = out["residual_load"] - out["residual_load_lag_1"]
    out["abs_residual_ramp_1"] = np.abs(out["residual_load_diff_1"])
    out["residual_load_diff_6"] = out["residual_load"] - grp["residual_load"].shift(6)
    out["abs_residual_ramp_6"] = np.abs(out["residual_load_diff_6"])

    lagged = grp["residual_load"].shift(1)
    out["residual_load_roll_std_6"] = (
        lagged.groupby(out["market"], sort=False).rolling(window=6, min_periods=6).std().reset_index(level=0, drop=True)
    )
    return out.sort_index()


def _add_slow_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["market", "ts"], kind="mergesort")
    lag1 = out.groupby("market", sort=False)["residual_load"].shift(1)

    for w in SLOW_WINDOWS:
        minp = max(6, w // 4)
        out[f"residual_load_roll_mean_{w}"] = (
            lag1.groupby(out["market"], sort=False).rolling(window=w, min_periods=minp).mean().reset_index(level=0, drop=True)
        )
        out[f"residual_load_roll_std_{w}"] = (
            lag1.groupby(out["market"], sort=False).rolling(window=w, min_periods=minp).std().reset_index(level=0, drop=True)
        )
    return out.sort_index()


def _add_meteo_conversions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = set(out.columns)

    if "wind_speed_80m" in cols:
        out["ws80_cu"] = out["wind_speed_80m"] ** 3
    if {"wind_gust_speed_10m", "wind_speed_10m"}.issubset(cols):
        out["gustiness_10m"] = out["wind_gust_speed_10m"] - out["wind_speed_10m"]
        out["gust_ratio_10m"] = _safe_div(out["wind_gust_speed_10m"], out["wind_speed_10m"])
    if {"wind_speed_80m", "wind_speed_10m"}.issubset(cols):
        out["ws_ratio_80_10"] = _safe_div(out["wind_speed_80m"], out["wind_speed_10m"])
    if "wind_direction_80m" in cols:
        rad = np.deg2rad(out["wind_direction_80m"].astype(float))
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)

    if {"direct_normal_irradiance", "global_horizontal_irradiance"}.issubset(cols):
        out["clear_sky_proxy"] = _safe_div(out["direct_normal_irradiance"], out["global_horizontal_irradiance"])
    if {"diffuse_horizontal_irradiance", "global_horizontal_irradiance"}.issubset(cols):
        out["diffuse_share"] = _safe_div(out["diffuse_horizontal_irradiance"], out["global_horizontal_irradiance"])
    if {"cloud_cover_low", "cloud_cover_total"}.issubset(cols):
        out["cloud_low_share"] = _safe_div(out["cloud_cover_low"], out["cloud_cover_total"])
    if "cloud_cover_total" in cols:
        out["cloud_x_solar_day"] = out["cloud_cover_total"] * out["solar_forecast"] * out["is_midday"]

    if {"air_temperature_2m", "dew_point_temperature_2m"}.issubset(cols):
        out["temp_dew_spread"] = out["air_temperature_2m"] - out["dew_point_temperature_2m"]
    if {"apparent_temperature_2m", "air_temperature_2m"}.issubset(cols):
        out["temp_apparent_gap"] = out["apparent_temperature_2m"] - out["air_temperature_2m"]
    if "air_temperature_2m" in cols:
        out["heating_degree"] = np.clip(18.0 - out["air_temperature_2m"], a_min=0.0, a_max=None)
        out["cooling_degree"] = np.clip(out["air_temperature_2m"] - 22.0, a_min=0.0, a_max=None)

    return out


def _apply_storm_features(df: pd.DataFrame, thresholds: RegimeThresholds) -> pd.DataFrame:
    out = df.copy()
    cols = set(out.columns)

    if "convective_available_potential_energy" not in cols:
        out["storm_risk"] = np.nan
        out["storm_flag"] = np.nan
        return out

    storm_energy = np.log1p(np.clip(out["convective_available_potential_energy"].astype(float), a_min=0.0, a_max=None))
    mode = thresholds.storm_mode

    if mode == "cin" and "convective_inhibition" in cols:
        q_cin = out["market"].map(thresholds.storm_cin_q20_by_market).fillna(thresholds.storm_cin_q20_global)
        indicator = (out["convective_inhibition"] <= q_cin).astype(float)
    elif mode == "lifted" and "lifted_index" in cols:
        indicator = (out["lifted_index"] <= -2.0).astype(float)
    else:
        indicator = pd.Series(np.zeros(len(out), dtype=float), index=out.index)

    out["storm_risk"] = storm_energy * indicator
    if np.isfinite(thresholds.storm_risk_q90_global):
        out["storm_flag"] = (out["storm_risk"] > thresholds.storm_risk_q90_global).astype(int)
    else:
        out["storm_flag"] = np.nan
    return out


def _add_cross_market_features(
    df: pd.DataFrame,
    local_base_vars: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    grouped = out.groupby("ts", sort=False)
    new_cols: dict[str, pd.Series] = {}

    def _robust_z(diff: pd.Series, std: pd.Series) -> pd.Series:
        cond = std.abs() > Z_MIN_STD
        z = np.where(cond.to_numpy(), (diff / (std + EPS)).to_numpy(), 0.0)
        return pd.Series(np.clip(z, -Z_CLIP, Z_CLIP), index=diff.index)

    def _robust_rank(var: str, std: pd.Series) -> pd.Series:
        rank = grouped[var].rank(pct=True, method="average")
        return rank.where(std.abs() > Z_MIN_STD, 0.5)

    for var in XMK_GLOBAL_BASE_VARS:
        if var not in out.columns:
            continue
        mean = grouped[var].transform("mean")
        std = grouped[var].transform("std")
        diff = out[var] - mean
        new_cols[f"{var}_xmk_mean"] = mean
        new_cols[f"{var}_xmk_std"] = std
        new_cols[f"{var}_xmk_diff"] = diff
        new_cols[f"{var}_xmk_z"] = _robust_z(diff, std)

    if "residual_load" in out.columns:
        std = grouped["residual_load"].transform("std")
        new_cols["residual_load_xmk_rank_pct"] = _robust_rank("residual_load", std)
    if "oversupply_share" in out.columns:
        std = grouped["oversupply_share"].transform("std")
        new_cols["oversupply_share_xmk_rank_pct"] = _robust_rank("oversupply_share", std)
    if "solar_share" in out.columns:
        std = grouped["solar_share"].transform("std")
        new_cols["solar_share_xmk_rank_pct"] = _robust_rank("solar_share", std)
    if "wind_share" in out.columns:
        std = grouped["wind_share"].transform("std")
        new_cols["wind_share_xmk_rank_pct"] = _robust_rank("wind_share", std)

    local_used: list[str] = []
    for var in local_base_vars:
        if var not in out.columns:
            continue
        local_used.append(var)
        mean = grouped[var].transform("mean")
        std = grouped[var].transform("std")
        diff = out[var] - mean
        new_cols[f"{var}_xmk_mean"] = mean
        new_cols[f"{var}_xmk_std"] = std
        new_cols[f"{var}_xmk_diff"] = diff
        new_cols[f"{var}_xmk_z"] = _robust_z(diff, std)
        new_cols[f"{var}_xmk_rank_pct"] = _robust_rank(var, std)

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out, sorted(local_used)


def make_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _add_time_features(df)
    out = _add_system_state_features(out)
    out = _add_transition_features(out)
    out = _add_slow_state_features(out)
    out = _add_meteo_conversions(out)
    return out


def fit_regime_thresholds(train_features: pd.DataFrame) -> RegimeThresholds:
    train = train_features.copy()
    cols = set(train.columns)

    midday = train[train["is_midday"] == 1].copy()
    peak = train[train["is_peak_17_20"] == 1].copy()

    lowprice_resid = midday.groupby("market")["residual_load"].quantile(0.25).to_dict()
    lowprice_solar = midday.groupby("market")["solar_share"].quantile(0.75).to_dict()

    transition_ramp = train.groupby("market")["abs_residual_ramp_1"].quantile(0.85).to_dict()
    transition_std = train.groupby("market")["residual_load_roll_std_6"].quantile(0.85).to_dict()

    peak_resid = peak.groupby("market")["residual_load"].quantile(0.80).to_dict()
    peak_windshare = peak.groupby("market")["wind_share"].quantile(0.20).to_dict()
    peak_stress = peak.groupby("market")["stress_ratio"].quantile(0.80).to_dict()

    global_defaults = {
        "lowprice_residual_q25": float(midday["residual_load"].quantile(0.25)) if len(midday) else float("nan"),
        "lowprice_solar_q75": float(midday["solar_share"].quantile(0.75)) if len(midday) else float("nan"),
        "transition_ramp_q85": float(train["abs_residual_ramp_1"].quantile(0.85)),
        "transition_std_q85": float(train["residual_load_roll_std_6"].quantile(0.85)),
        "peak_residual_q80": float(peak["residual_load"].quantile(0.80)) if len(peak) else float("nan"),
        "peak_windshare_q20": float(peak["wind_share"].quantile(0.20)) if len(peak) else float("nan"),
        "peak_stress_q80": float(peak["stress_ratio"].quantile(0.80)) if len(peak) else float("nan"),
    }

    peak_base = (
        (train["is_peak_17_20"] == 1)
        & (train["residual_load"] >= train["market"].map(peak_resid).fillna(global_defaults["peak_residual_q80"]))
        & (train["wind_share"] <= train["market"].map(peak_windshare).fillna(global_defaults["peak_windshare_q20"]))
    )
    peak_rows_count = peak.groupby("market")["id"].count().to_dict()
    peak_hits_count = train[peak_base].groupby("market")["id"].count().to_dict()
    apply_stress_filter: dict[str, bool] = {}
    for market, n_peak in peak_rows_count.items():
        share = float(peak_hits_count.get(market, 0)) / float(n_peak) if n_peak > 0 else 0.0
        apply_stress_filter[market] = bool(share > 0.10)

    storm_mode = "none"
    storm_cin_q20_by_market: dict[str, float] = {}
    storm_cin_q20_global = float("nan")
    storm_risk_q90_global = float("nan")

    if "convective_available_potential_energy" in cols:
        storm_energy = np.log1p(np.clip(train["convective_available_potential_energy"].astype(float), a_min=0.0, a_max=None))
        if "convective_inhibition" in cols:
            storm_mode = "cin"
            storm_cin_q20_by_market = train.groupby("market")["convective_inhibition"].quantile(0.20).to_dict()
            storm_cin_q20_global = float(train["convective_inhibition"].quantile(0.20))
            q_cin = train["market"].map(storm_cin_q20_by_market).fillna(storm_cin_q20_global)
            indicator = (train["convective_inhibition"] <= q_cin).astype(float)
        elif "lifted_index" in cols:
            storm_mode = "lifted"
            indicator = (train["lifted_index"] <= -2.0).astype(float)
        else:
            indicator = pd.Series(np.zeros(len(train), dtype=float), index=train.index)
        storm_risk = storm_energy * indicator
        if len(storm_risk):
            storm_risk_q90_global = float(storm_risk.quantile(0.90))

    return RegimeThresholds(
        lowprice_residual_q25_by_market={k: float(v) for k, v in lowprice_resid.items()},
        lowprice_solar_q75_by_market={k: float(v) for k, v in lowprice_solar.items()},
        transition_ramp_q85_by_market={k: float(v) for k, v in transition_ramp.items()},
        transition_std_q85_by_market={k: float(v) for k, v in transition_std.items()},
        peak_residual_q80_by_market={k: float(v) for k, v in peak_resid.items()},
        peak_windshare_q20_by_market={k: float(v) for k, v in peak_windshare.items()},
        peak_stress_q80_by_market={k: float(v) for k, v in peak_stress.items()},
        apply_peak_stress_filter_by_market=apply_stress_filter,
        storm_mode=storm_mode,
        storm_cin_q20_by_market={k: float(v) for k, v in storm_cin_q20_by_market.items()},
        storm_cin_q20_global=storm_cin_q20_global,
        storm_risk_q90_global=storm_risk_q90_global,
        global_defaults=global_defaults,
    )


def apply_regimes(df: pd.DataFrame, thresholds: RegimeThresholds) -> pd.DataFrame:
    out = df.copy()

    q_resid_low = out["market"].map(thresholds.lowprice_residual_q25_by_market).fillna(
        thresholds.global_defaults["lowprice_residual_q25"]
    )
    q_solar_high = out["market"].map(thresholds.lowprice_solar_q75_by_market).fillna(
        thresholds.global_defaults["lowprice_solar_q75"]
    )
    q_ramp = out["market"].map(thresholds.transition_ramp_q85_by_market).fillna(
        thresholds.global_defaults["transition_ramp_q85"]
    )
    q_std = out["market"].map(thresholds.transition_std_q85_by_market).fillna(
        thresholds.global_defaults["transition_std_q85"]
    )
    q_peak_resid = out["market"].map(thresholds.peak_residual_q80_by_market).fillna(
        thresholds.global_defaults["peak_residual_q80"]
    )
    q_peak_wind = out["market"].map(thresholds.peak_windshare_q20_by_market).fillna(
        thresholds.global_defaults["peak_windshare_q20"]
    )
    q_peak_stress = out["market"].map(thresholds.peak_stress_q80_by_market).fillna(
        thresholds.global_defaults["peak_stress_q80"]
    )
    apply_peak_stress = out["market"].map(thresholds.apply_peak_stress_filter_by_market).fillna(False)

    lowprice_rule_a = (
        (out["is_midday"] == 1)
        & (out["residual_load"] <= q_resid_low)
        & (out["solar_share"] >= q_solar_high)
    )
    residual_rank = out["residual_load_xmk_rank_pct"] if "residual_load_xmk_rank_pct" in out.columns else pd.Series(0.5, index=out.index)
    solar_rank = out["solar_share_xmk_rank_pct"] if "solar_share_xmk_rank_pct" in out.columns else pd.Series(0.5, index=out.index)

    lowprice_rule_b = (
        (out["is_midday"] == 1)
        & (residual_rank <= 0.25)
        & (solar_rank >= 0.75)
    )
    out["lowprice_midday"] = lowprice_rule_a | lowprice_rule_b

    peak_base = (
        (out["is_peak_17_20"] == 1)
        & (out["residual_load"] >= q_peak_resid)
        & (out["wind_share"] <= q_peak_wind)
    )
    out["peak_scarcity"] = peak_base & ((~apply_peak_stress) | (out["stress_ratio"] >= q_peak_stress))
    out["peak_scarcity"] = out["peak_scarcity"].fillna(False)

    out["transition"] = (
        (out["abs_residual_ramp_1"] >= q_ramp)
        | (out["residual_load_roll_std_6"] >= q_std)
    )
    out["transition"] = out["transition"].fillna(False)

    out["normal"] = ~(out["lowprice_midday"] | out["peak_scarcity"] | out["transition"])
    out["primary_regime"] = np.select(
        [out["lowprice_midday"], out["peak_scarcity"], out["transition"]],
        ["lowprice_midday", "peak_scarcity", "transition"],
        default="normal",
    )
    return out


def build_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    use_xmk: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, RegimeThresholds, dict[str, Any]]:
    train_feat = make_base_features(train_df)
    test_feat = make_base_features(test_df)
    thresholds = fit_regime_thresholds(train_feat)

    train_feat = _apply_storm_features(train_feat, thresholds)
    test_feat = _apply_storm_features(test_feat, thresholds)

    local_used: list[str] = []
    if use_xmk:
        local_xmk_candidates = [
            v
            for v in METEO_CONVERSION_BASE_FEATURES
            if (v in train_feat.columns) or (v in test_feat.columns)
        ]
        train_feat, local_used_train = _add_cross_market_features(train_feat, local_xmk_candidates)
        test_feat, local_used_test = _add_cross_market_features(test_feat, local_xmk_candidates)
        local_used = sorted(set(local_used_train).union(local_used_test))
    else:
        for col in [
            "residual_load_xmk_rank_pct",
            "oversupply_share_xmk_rank_pct",
            "solar_share_xmk_rank_pct",
            "wind_share_xmk_rank_pct",
        ]:
            train_feat[col] = 0.5
            test_feat[col] = 0.5

    meteo_activated = sorted([v for v in METEO_CONVERSION_BASE_FEATURES if v in train_feat.columns or v in test_feat.columns])
    thresholds.local_xmk_base_vars = local_used
    thresholds.available_meteo_features = meteo_activated

    train_feat = apply_regimes(train_feat, thresholds)
    test_feat = apply_regimes(test_feat, thresholds)

    metadata = {
        "meteo_features_activated": meteo_activated,
        "local_xmk_base_vars_used": local_used,
        "use_xmk": bool(use_xmk),
    }
    return train_feat, test_feat, thresholds, metadata


def build_gate(primary_regime: pd.Series, gate_mode: str) -> pd.Series:
    if gate_mode == "none":
        weights = np.ones(len(primary_regime), dtype=float)
    elif gate_mode == "rule":
        weights = np.where(
            primary_regime == "transition",
            1.0,
            np.where(
                primary_regime == "peak_scarcity",
                1.0,
                np.where(primary_regime == "lowprice_midday", 0.5, 0.3),
            ),
        )
    elif gate_mode == "conservative":
        weights = np.where(
            primary_regime == "transition",
            1.0,
            np.where(
                primary_regime == "peak_scarcity",
                1.0,
                np.where(primary_regime == "lowprice_midday", 0.2, 0.1),
            ),
        )
    elif gate_mode == "mid":
        weights = np.where(
            primary_regime == "transition",
            1.0,
            np.where(
                primary_regime == "peak_scarcity",
                1.0,
                np.where(primary_regime == "lowprice_midday", 0.35, 0.2),
            ),
        )
    else:
        raise ValueError(f"Unsupported gate_mode: {gate_mode}")
    return pd.Series(weights.astype(float), index=primary_regime.index)


def get_feature_columns(use_xmk: bool = True) -> list[str]:
    cols: list[str] = [
        "market",
        "primary_regime",
        "market_hour",
        "market_dow",
        "market_month",
        "hour",
        "dow",
        "month",
        "weekofyear",
        "is_weekend",
        "is_midday",
        "is_peak_17_20",
        "is_night",
        "is_winter",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "load_forecast",
        "wind_forecast",
        "solar_forecast",
        "residual_load",
        "oversupply",
        "oversupply_pos",
        "oversupply_share",
        "wind_share",
        "solar_share",
        "net_load_share",
        "stress_ratio",
        "residual_load_sq",
        "stress_ratio_sq",
        "residual_load_pos",
        "residual_load_neg",
        "oversupply_share_sq",
        "solar_share_x_is_midday",
        "stress_x_is_peak",
        "residual_load_lag_1",
        "residual_load_diff_1",
        "abs_residual_ramp_1",
        "residual_load_diff_6",
        "abs_residual_ramp_6",
        "residual_load_roll_std_6",
    ]

    for w in SLOW_WINDOWS:
        cols.extend([f"residual_load_roll_mean_{w}", f"residual_load_roll_std_{w}"])

    cols.extend(list(METEO_CONVERSION_BASE_FEATURES))

    if use_xmk:
        for var in XMK_GLOBAL_BASE_VARS:
            cols.extend(
                [
                    f"{var}_xmk_mean",
                    f"{var}_xmk_std",
                    f"{var}_xmk_diff",
                    f"{var}_xmk_z",
                ]
            )

        cols.extend(
            [
                "residual_load_xmk_rank_pct",
                "oversupply_share_xmk_rank_pct",
                "solar_share_xmk_rank_pct",
                "wind_share_xmk_rank_pct",
            ]
        )

        for var in METEO_CONVERSION_BASE_FEATURES:
            cols.extend(
                [
                    f"{var}_xmk_mean",
                    f"{var}_xmk_std",
                    f"{var}_xmk_diff",
                    f"{var}_xmk_z",
                    f"{var}_xmk_rank_pct",
                ]
            )

    cols.extend(
        [
            "lowprice_midday",
            "peak_scarcity",
            "transition",
            "normal",
            "pred_base",
        ]
    )

    # Preserve order while removing duplicates.
    seen = set()
    deduped: list[str] = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        deduped.append(c)
    return deduped


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: tuple[str, ...] = ("market", "primary_regime", "market_hour", "market_dow", "market_month"),
) -> pd.DataFrame:
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            out[col] = np.nan
    x = out[feature_cols].copy()
    for col in x.columns:
        if x[col].dtype == bool:
            x[col] = x[col].astype(float)
    for col in categorical_cols:
        if col in x.columns:
            x[col] = x[col].astype("category")
    return x
