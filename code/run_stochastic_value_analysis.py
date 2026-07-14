#!/usr/bin/env python3
"""Compute exact RP, EEV, VSS, wait-and-see, and EVPI benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from release_layout import DATA_DIR, OUT_DIR, VERSION
from run_rigorous_revision import (
    LOG_DIR,
    SOLUTION_DIR,
    TMP_DIR,
    PolicySpec,
    build_revision_inputs,
    solve_single_period,
)
from scenario_generator import demand_matrix
from stochastic_analysis import build_evaluation_context, evaluate_design, jaccard, read_design


def expected_demand_scenario(bundle: dict[str, object]) -> pd.DataFrame:
    matrix, _ = demand_matrix(bundle["city"], bundle["scenario"], bundle["probs"])
    expected = np.asarray(bundle["probs"], dtype=float) @ matrix
    city = bundle["city"][["city_code_final", "city_name", "region"]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "scenario_id": "EV0001",
            "base_scenario": "probability_weighted_expected_demand",
            "year": 2030,
            "city_code_final": city["city_code_final"].astype(int),
            "city_name": city["city_name"],
            "region": city["region"],
            "demand_ton": expected,
            "probability": 1.0,
            "data_version": VERSION,
        }
    )


def main() -> None:
    for directory in (DATA_DIR, OUT_DIR, LOG_DIR, SOLUTION_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    bundle = build_revision_inputs()
    baseline = PolicySpec("baseline", "primary")
    context = build_evaluation_context(bundle, baseline)

    primary_path = OUT_DIR / "single_period_exact_results.csv"
    if not primary_path.exists():
        raise FileNotFoundError("Run run_rigorous_revision.py before stochastic-value analysis")
    primary = pd.read_csv(primary_path).set_index("name")
    rp_reported = float(primary.loc["baseline", "reevaluated_objective_yuan"])
    rp_x, rp_capacity = read_design(
        SOLUTION_DIR / "single_baseline_facility_solution.csv",
        context.facility,
    )
    rp_evaluation = evaluate_design(
        context,
        bundle["scenario"],
        bundle["probs"],
        rp_x,
        rp_capacity,
    )
    rp = float(rp_evaluation["expected_total_cost_yuan"])
    if not np.isclose(rp, rp_reported, rtol=3e-13, atol=1e-4):
        raise RuntimeError("RP fixed-design evaluation does not reproduce the primary exact result")

    expected_scenario = expected_demand_scenario(bundle)
    expected_scenario.to_csv(DATA_DIR / "stochastic_value_expected_demand.csv", index=False)
    ev_bundle = {**bundle, "scenario": expected_scenario, "probs": np.array([1.0])}
    ev_spec = PolicySpec("ev_expected_demand", "stochastic_value")
    ev_result = solve_single_period(ev_bundle, ev_spec)
    if ev_result.get("solver_status") != "optimal":
        raise RuntimeError(f"Expected-value problem was not solved to optimality: {ev_result}")
    ev_x, ev_capacity = read_design(
        SOLUTION_DIR / "single_ev_expected_demand_facility_solution.csv",
        context.facility,
    )
    eev_evaluation = evaluate_design(
        context,
        bundle["scenario"],
        bundle["probs"],
        ev_x,
        ev_capacity,
    )
    eev = float(eev_evaluation["expected_total_cost_yuan"])

    scenario_names = sorted(bundle["scenario"]["scenario_id"].astype(str).unique())
    ws_rows: list[dict[str, object]] = []
    for index, (scenario_name, probability) in enumerate(zip(scenario_names, bundle["probs"]), start=1):
        print(f"[wait-and-see] {index}/{len(scenario_names)} {scenario_name}", flush=True)
        subset = bundle["scenario"][bundle["scenario"]["scenario_id"].astype(str).eq(scenario_name)].copy()
        ws_bundle = {**bundle, "scenario": subset, "probs": np.array([1.0])}
        ws_spec = PolicySpec(f"ws_{index:04d}", "wait_and_see")
        result = solve_single_period(ws_bundle, ws_spec)
        if result.get("solver_status") != "optimal":
            raise RuntimeError(f"Wait-and-see problem {scenario_name} was not solved to optimality")
        ws_rows.append(
            {
                "scenario_id": scenario_name,
                "scenario_probability": float(probability),
                "objective_yuan": float(result["reevaluated_objective_yuan"]),
                "open_facilities": int(result["open_facilities"]),
                "installed_capacity_ton": float(result["installed_capacity_ton"]),
                "expected_shortage_ton": float(result["expected_shortage_ton"]),
                "expected_emission_tco2": float(result["expected_emission_tco2"]),
                "solver_status": result["solver_status"],
                "solver_gap": float(result["solver_gap"]),
                "runtime_sec": float(result["runtime_sec"]),
                "objective_crosscheck_relative_error": float(result["objective_crosscheck_relative_error"]),
                "data_version": VERSION,
            }
        )

    ws_frame = pd.DataFrame(ws_rows)
    ws_frame.to_csv(OUT_DIR / "wait_and_see_exact_results.csv", index=False)
    ws = float(ws_frame["scenario_probability"] @ ws_frame["objective_yuan"])
    vss = eev - rp
    evpi = rp - ws
    tolerance = 1e-7 * max(abs(rp), 1.0)
    if vss < -tolerance or evpi < -tolerance:
        raise RuntimeError(f"Stochastic-value inequalities failed: VSS={vss}, EVPI={evpi}")

    union_capacity = np.maximum(rp_capacity, ev_capacity).sum()
    capacity_overlap = float(np.minimum(rp_capacity, ev_capacity).sum() / union_capacity) if union_capacity else 1.0
    summary = pd.DataFrame(
        [
            {
                "rp_yuan": rp,
                "ev_problem_yuan": float(ev_result["reevaluated_objective_yuan"]),
                "eev_yuan": eev,
                "ws_yuan": ws,
                "vss_yuan": vss,
                "vss_pct_of_rp": 100.0 * vss / rp,
                "evpi_yuan": evpi,
                "evpi_pct_of_rp": 100.0 * evpi / rp,
                "rp_open_facilities": int((rp_x > 0.5).sum()),
                "ev_open_facilities": int((ev_x > 0.5).sum()),
                "open_site_jaccard_rp_vs_ev": jaccard(rp_x, ev_x),
                "capacity_overlap_rp_vs_ev": capacity_overlap,
                "rp_expected_shortage_ton": float(rp_evaluation["expected_shortage_ton"]),
                "eev_expected_shortage_ton": float(eev_evaluation["expected_shortage_ton"]),
                "rp_expected_emission_tco2": float(rp_evaluation["expected_emission_tco2"]),
                "eev_expected_emission_tco2": float(eev_evaluation["expected_emission_tco2"]),
                "rp_objective_crosscheck_relative_error": (rp - rp_reported) / max(abs(rp_reported), 1.0),
                "data_version": VERSION,
            }
        ]
    )
    summary.to_csv(OUT_DIR / "stochastic_value_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
