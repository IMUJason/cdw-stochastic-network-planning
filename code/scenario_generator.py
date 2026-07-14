#!/usr/bin/env python3
"""Generate mean-preserving correlated demand scenarios for the CDW study."""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


TARGET_YEAR = 2030
FAMILY_WEIGHTS = {"baseline": 0.50, "strong_management": 0.25, "high_growth": 0.25}
REGION_CORRELATION = np.array(
    [
        [1.0, 0.65, 0.45],
        [0.65, 1.0, 0.50],
        [0.45, 0.50, 1.0],
    ],
    dtype=float,
)
REGION_SIGMA = {"baseline": 0.10, "strong_management": 0.08, "high_growth": 0.13}
IDIOSYNCRATIC_SIGMA = {"baseline": 0.05, "strong_management": 0.04, "high_growth": 0.07}
REGION_INDEX = {"east": 0, "central": 1, "west": 2}
GENERATOR_VERSION = "mean_preserving_correlated_lognormal_v1"


def family_counts(scenario_count: int, weights: Mapping[str, float] = FAMILY_WEIGHTS) -> dict[str, int]:
    """Return exact integer family counts for a requested scenario-set size."""
    raw = {family: scenario_count * float(weight) for family, weight in weights.items()}
    if any(not np.isclose(value, round(value)) for value in raw.values()):
        raise ValueError("Scenario count must preserve the specified family weights exactly")
    counts = {family: int(round(value)) for family, value in raw.items()}
    if sum(counts.values()) != scenario_count or any(value <= 0 for value in counts.values()):
        raise ValueError("Invalid family counts")
    return counts


def generate_scenarios(
    projections: pd.DataFrame,
    cities: pd.DataFrame,
    scenario_count: int,
    seed: int,
    *,
    scenario_prefix: str = "S",
    target_year: int = TARGET_YEAR,
    weights: Mapping[str, float] = FAMILY_WEIGHTS,
    correlation: np.ndarray = REGION_CORRELATION,
    region_sigma: Mapping[str, float] = REGION_SIGMA,
    idiosyncratic_sigma: Mapping[str, float] = IDIOSYNCRATIC_SIGMA,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Draw a sample-average approximation from the disclosed demand generator.

    The lognormal correction subtracts one half of the marginal log variance,
    so each city's shock multiplier has population mean one within its demand
    family. Finite scenario sets may still deviate from that mean, as expected
    for independently generated sample-average approximations.
    """
    counts = family_counts(scenario_count, weights)
    correlation = np.asarray(correlation, dtype=float)
    if correlation.shape != (3, 3) or not np.allclose(correlation, correlation.T):
        raise ValueError("The macro-region correlation matrix must be symmetric 3 x 3")
    chol = np.linalg.cholesky(correlation)

    city_ref = cities[["city_code_final", "city_name", "region"]].copy()
    city_ref["city_code_final"] = city_ref["city_code_final"].astype(int)
    if not city_ref["region"].isin(REGION_INDEX).all():
        unknown = sorted(city_ref.loc[~city_ref["region"].isin(REGION_INDEX), "region"].astype(str).unique())
        raise ValueError(f"Unknown macro-region labels: {unknown}")

    target = projections[projections["year"].astype(int).eq(int(target_year))].copy()
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    probabilities: list[float] = []
    scenario_number = 1

    for family in weights:
        base = target[target["scenario"].astype(str).eq(family)][
            ["city_code_final", "cdw_pred_final"]
        ].copy()
        base["city_code_final"] = base["city_code_final"].astype(int)
        merged = city_ref.merge(base, on="city_code_final", how="left", validate="one_to_one")
        if merged["cdw_pred_final"].isna().any():
            missing = merged.loc[merged["cdw_pred_final"].isna(), "city_code_final"].tolist()
            raise ValueError(f"Missing {target_year} projections for {family}: {missing[:5]}")

        sigma_macro = float(region_sigma[family])
        sigma_local = float(idiosyncratic_sigma[family])
        mean_correction = 0.5 * (sigma_macro**2 + sigma_local**2)
        region_ids = merged["region"].map(REGION_INDEX).to_numpy(int)
        base_demand = merged["cdw_pred_final"].to_numpy(float)
        probability = float(weights[family]) / counts[family]

        for _ in range(counts[family]):
            scenario_id = f"{scenario_prefix}{scenario_number:04d}"
            scenario_number += 1
            macro_shock = chol @ rng.normal(size=3)
            local_shock = rng.normal(size=len(merged))
            log_multiplier = (
                sigma_macro * macro_shock[region_ids]
                + sigma_local * local_shock
                - mean_correction
            )
            demand = base_demand * np.exp(log_multiplier)
            probabilities.append(probability)
            for row, value in zip(merged.itertuples(index=False), demand):
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "base_scenario": family,
                        "year": int(target_year),
                        "city_code_final": int(row.city_code_final),
                        "city_name": row.city_name,
                        "region": row.region,
                        "demand_ton": float(value),
                        "probability": probability,
                        "scenario_set_seed": int(seed),
                        "generator_version": GENERATOR_VERSION,
                    }
                )

    scenario = pd.DataFrame(rows)
    probs = np.asarray(probabilities, dtype=float)
    probs /= probs.sum()
    if scenario["scenario_id"].nunique() != scenario_count:
        raise RuntimeError("Generated scenario count mismatch")
    if not np.isclose(probs.sum(), 1.0):
        raise RuntimeError("Generated probabilities do not sum to one")
    if scenario.groupby("scenario_id")["city_code_final"].nunique().ne(len(city_ref)).any():
        raise RuntimeError("Generated scenario city coverage mismatch")
    return scenario, probs


def demand_matrix(cities: pd.DataFrame, scenario: pd.DataFrame, probabilities: Sequence[float]) -> tuple[np.ndarray, list[str]]:
    """Convert a long scenario table to an ordered scenario-by-city matrix."""
    city_to_idx = {int(value): i for i, value in enumerate(cities["city_code_final"].astype(int))}
    names = sorted(scenario["scenario_id"].astype(str).unique())
    if len(names) != len(probabilities):
        raise ValueError("Scenario probability length mismatch")
    matrix = np.zeros((len(names), len(cities)), dtype=float)
    for s, name in enumerate(names):
        subset = scenario[scenario["scenario_id"].astype(str).eq(name)]
        for row in subset[["city_code_final", "demand_ton"]].itertuples(index=False):
            matrix[s, city_to_idx[int(row.city_code_final)]] = float(row.demand_ton)
    if np.any(matrix <= 0):
        raise ValueError("Every scenario-city demand must be positive")
    return matrix, names
