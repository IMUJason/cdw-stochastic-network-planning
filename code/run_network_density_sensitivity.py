#!/usr/bin/env python3
"""Exact sensitivity of policy results to the retained road arcs per city."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from run_rigorous_revision import (
    DATA_DIR,
    LOG_DIR,
    OUT_DIR,
    SOLUTION_DIR,
    SOURCE_PATHS,
    TMP_DIR,
    VERSION,
    PolicySpec,
    solve_single_period,
    write_manifest,
)


NETWORK_K = (8, 12, 20, 30)
POLICIES = (
    PolicySpec("baseline", "network_density"),
    PolicySpec("carbon_tax_120", "network_density", carbon_price=120.0),
    PolicySpec("landfill_penalty_260", "network_density", landfill_penalty=260.0),
    PolicySpec("combined_120_260", "network_density", carbon_price=120.0, landfill_penalty=260.0),
)


def build_edges(city: pd.DataFrame, facility: pd.DataFrame, k: int) -> pd.DataFrame:
    transport = pd.read_csv(SOURCE_PATHS["transport_matrix"])
    regional = pd.read_csv(SOURCE_PATHS["regional_operating_parameters"])
    regional_by_province = regional.drop_duplicates("province").set_index("province")
    city_codes = set(city.city_code_final.astype(int))
    site_ids = set(facility.site_id.astype(int))
    edge = transport[
        transport.city_code.astype(int).isin(city_codes)
        & transport.site_id.astype(int).isin(site_ids)
    ].copy()
    province = city.assign(code=city.city_code_final.astype(int)).set_index("code").province
    edge["origin_province"] = edge.city_code.astype(int).map(province)
    edge["unit_transport_cost"] = (
        edge.highway_distance_km.astype(float)
        * edge.origin_province.map(regional_by_province.transport_cost_yuan_per_tkm).astype(float)
    )
    edge["transport_emission_tco2_per_t"] = edge.highway_distance_km.astype(float) * 0.12 / 1000.0
    edge["best_mode"] = "road"
    edge["network_rule"] = f"{k} lowest road-cost candidate facilities per matched city"
    edge["data_version"] = VERSION
    edge = edge.sort_values(["city_code", "unit_transport_cost"]).groupby("city_code", as_index=False).head(k)
    counts = edge.groupby("city_code").size()
    if len(counts) != len(city) or not (counts == k).all():
        raise RuntimeError(f"Network k={k} does not contain exactly {k} arcs per city")
    return edge


def main() -> None:
    for directory in (OUT_DIR, LOG_DIR, SOLUTION_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    city = pd.read_csv(DATA_DIR / "city_list_complete_case.csv")
    facility = pd.read_csv(DATA_DIR / "facility_list_transparent_proxy.csv")
    scenario = pd.read_csv(DATA_DIR / "stochastic_scenarios_2030.csv")
    probs = np.load(DATA_DIR / "scenario_probabilities_2030.npy")
    rows = []
    for k in NETWORK_K:
        edge = build_edges(city, facility, k)
        bundle = {"city": city, "facility": facility, "edge": edge, "scenario": scenario, "probs": probs}
        for base_spec in POLICIES:
            spec = replace(
                base_spec,
                name=f"network_k{k:02d}_{base_spec.name}",
                parameter_value=float(k),
            )
            print(f"[network] k={k} policy={base_spec.name}", flush=True)
            row = solve_single_period(bundle, spec)
            row["network_k"] = k
            row["policy"] = base_spec.name
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT_DIR / "network_density_sensitivity_checkpoint.csv", index=False)
    result = pd.DataFrame(rows)
    result["cost_change_pct_vs_k_baseline"] = np.nan
    result["emission_reduction_pct_vs_k_baseline"] = np.nan
    result["shortage_reduction_pct_vs_k_baseline"] = np.nan
    for k, frame in result.groupby("network_k"):
        baseline = frame[frame.policy == "baseline"].iloc[0]
        idx = frame.index
        result.loc[idx, "cost_change_pct_vs_k_baseline"] = (
            result.loc[idx, "reevaluated_objective_yuan"] / baseline.reevaluated_objective_yuan - 1
        ) * 100
        result.loc[idx, "emission_reduction_pct_vs_k_baseline"] = (
            1 - result.loc[idx, "expected_emission_tco2"] / baseline.expected_emission_tco2
        ) * 100
        result.loc[idx, "shortage_reduction_pct_vs_k_baseline"] = (
            1 - result.loc[idx, "expected_shortage_ton"] / baseline.expected_shortage_ton
        ) * 100
    result.to_csv(OUT_DIR / "network_density_sensitivity.csv", index=False)
    (OUT_DIR / "network_density_sensitivity_checkpoint.csv").unlink(missing_ok=True)
    write_manifest(
        {
            "city": city,
            "facility": facility,
            "edge": pd.read_csv(DATA_DIR / "road_transport_edges.csv"),
            "probs": probs,
        }
    )
    print(result[["network_k", "policy", "solver_status", "reevaluated_objective_yuan", "open_facilities", "expected_shortage_ton"]].to_string(index=False))


if __name__ == "__main__":
    main()
