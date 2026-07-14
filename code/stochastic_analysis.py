#!/usr/bin/env python3
"""Shared exact evaluation helpers for stochastic-value and stability analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from benders_solver import BendersConfig, _build_subproblem_matrices, build_index, prepare_edge_arrays, solve_subproblem
from run_rigorous_revision import PolicySpec
from scenario_generator import demand_matrix


@dataclass(frozen=True)
class EvaluationContext:
    city: pd.DataFrame
    facility: pd.DataFrame
    edge_arrays: Dict[str, np.ndarray]
    matrices: Dict[str, object]
    processing_cost: np.ndarray
    processing_emission: np.ndarray
    landfill_cost: np.ndarray
    fixed_cost: np.ndarray
    capacity_cost: np.ndarray
    config: BendersConfig


def build_evaluation_context(bundle: Dict[str, object], spec: PolicySpec) -> EvaluationContext:
    """Apply a policy's parameter scales and prepare reusable recourse matrices."""
    city = bundle["city"].copy()
    facility = bundle["facility"].copy()
    edge = bundle["edge"].copy()
    facility["fixed_cost_yuan"] *= spec.fixed_cost_scale
    facility["capacity_cost_yuan_per_ton"] *= spec.capex_scale
    facility["processing_cost_yuan_per_t"] *= spec.operating_cost_scale
    edge["unit_transport_cost"] *= spec.operating_cost_scale
    city["landfill_cost_yuan_per_t"] *= spec.operating_cost_scale

    city_to_idx, facility_to_idx = build_index(city, facility)
    edge_arrays = prepare_edge_arrays(edge, city_to_idx, facility_to_idx)
    matrices = _build_subproblem_matrices(
        len(city),
        len(facility),
        edge_arrays["city_idx"],
        edge_arrays["fac_idx"],
    )
    config = BendersConfig(
        coverage_ratio=spec.coverage_ratio,
        min_open_facilities=24,
        max_open_facilities=spec.max_open,
        shortage_penalty=spec.base_shortage_penalty,
        landfill_penalty=spec.landfill_penalty,
        carbon_price_per_tco2=spec.carbon_price,
        shortage_emission_tco2_per_t=spec.shortage_emission,
    )
    return EvaluationContext(
        city=city,
        facility=facility,
        edge_arrays=edge_arrays,
        matrices=matrices,
        processing_cost=facility["processing_cost_yuan_per_t"].to_numpy(float),
        processing_emission=facility["processing_emission_tco2_per_t"].to_numpy(float),
        landfill_cost=city["landfill_cost_yuan_per_t"].to_numpy(float),
        fixed_cost=facility["fixed_cost_yuan"].to_numpy(float),
        capacity_cost=facility["capacity_cost_yuan_per_ton"].to_numpy(float),
        config=config,
    )


def read_design(path: Path, facilities: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Read a saved design and align it to the facility-table order."""
    observed = pd.read_csv(path).set_index("site_id")
    aligned = observed.reindex(facilities["site_id"])
    if aligned[["open", "installed_capacity_ton"]].isna().any().any():
        raise ValueError(f"Incomplete facility design: {path}")
    return aligned["open"].to_numpy(float), aligned["installed_capacity_ton"].to_numpy(float)


def evaluate_design(
    context: EvaluationContext,
    scenario: pd.DataFrame,
    probabilities: np.ndarray,
    open_decision: np.ndarray,
    capacity: np.ndarray,
) -> dict[str, object]:
    """Independently evaluate a fixed first-stage design over a scenario set."""
    probabilities = np.asarray(probabilities, dtype=float)
    matrix, names = demand_matrix(context.city, scenario, probabilities)
    first_stage = float(context.fixed_cost @ open_decision + context.capacity_cost @ capacity)
    recourse = np.zeros(len(probabilities), dtype=float)
    shortage = np.zeros(len(probabilities), dtype=float)
    emission = np.zeros(len(probabilities), dtype=float)
    transport = np.zeros(len(probabilities), dtype=float)
    processing = np.zeros(len(probabilities), dtype=float)
    shortage_cost = np.zeros(len(probabilities), dtype=float)
    carbon_cost = np.zeros(len(probabilities), dtype=float)

    for index, demand in enumerate(matrix):
        result = solve_subproblem(
            demand,
            capacity,
            context.edge_arrays,
            context.processing_cost,
            context.processing_emission,
            context.landfill_cost,
            context.config,
            context.matrices,
        )
        if not bool(result["success"]):
            raise RuntimeError(f"Recourse evaluation failed for {names[index]}: {result['message']}")
        recourse[index] = float(result["objective"])
        shortage[index] = float(np.asarray(result["z"], dtype=float).sum())
        emission[index] = float(result["total_emission_tco2"])
        transport[index] = float(result["transport_cost"])
        processing[index] = float(result["processing_cost"])
        shortage_cost[index] = float(result["shortage_cost"])
        carbon_cost[index] = float(result["carbon_cost"])

    total = first_stage + recourse
    return {
        "first_stage_cost_yuan": first_stage,
        "expected_recourse_cost_yuan": float(probabilities @ recourse),
        "expected_total_cost_yuan": float(probabilities @ total),
        "expected_shortage_ton": float(probabilities @ shortage),
        "expected_emission_tco2": float(probabilities @ emission),
        "expected_transport_cost_yuan": float(probabilities @ transport),
        "expected_processing_cost_yuan": float(probabilities @ processing),
        "expected_shortage_cost_yuan": float(probabilities @ shortage_cost),
        "expected_carbon_cost_yuan": float(probabilities @ carbon_cost),
        "scenario_total_cost_yuan": total,
        "scenario_shortage_ton": shortage,
        "scenario_emission_tco2": emission,
        "scenario_names": names,
    }


def jaccard(open_a: np.ndarray, open_b: np.ndarray) -> float:
    a = np.asarray(open_a) > 0.5
    b = np.asarray(open_b) > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0
