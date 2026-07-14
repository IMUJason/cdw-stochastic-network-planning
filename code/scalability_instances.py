#!/usr/bin/env python3
"""Shared scalability instance builder and tuned solver schedules."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def build_scalability_instances(
    city: pd.DataFrame,
    fac: pd.DataFrame,
    edge: pd.DataFrame,
    scen: pd.DataFrame,
    probs: np.ndarray,
    *,
    include_full_scale: bool = True,
) -> List[Dict[str, object]]:
    """Build the frozen scalability instances from a single code path.

    The same builder is consumed by both the Benders scalability experiment and
    the monolithic CPLEX benchmark so that city/facility/scenario counts stay
    perfectly aligned after any upstream regeneration.

    When ``include_full_scale`` is True (default), a fourth instance is appended
    that uses the full national calibration (all 114 facilities, all 371 cities
    with feasible arcs, and all 16 scenarios). This instance is identical to the
    one used for the key-policy convergence experiments, so the scalability
    evidence chain is directly comparable to the headline key-policy numbers.
    """
    fac_scales = [40, 70, 100]
    scen_scales = [8, 12, min(16, len(probs))]
    if include_full_scale:
        fac_scales.append(114)
        scen_scales.append(min(16, len(probs)))
    base_policy = {"carbon": 0.0, "subsidy": 0.0, "landfill": 0.0, "coverage": 0.90}

    fac_sorted = fac.sort_values("capacity_max_ton", ascending=False).copy()
    scen_ids_all = sorted(scen["scenario_id"].unique())
    instances: List[Dict[str, object]] = []

    for nf, ns in zip(fac_scales, scen_scales):
        sub_fac = fac_sorted.head(nf).copy()
        keep_sites = set(sub_fac["site_id"].astype(int))
        sub_edge = edge[edge["site_id"].isin(keep_sites)].copy()

        good_cities = sorted(sub_edge["city_code"].astype(int).unique())
        sub_city = city[city["city_code_final"].astype(int).isin(good_cities)].copy()

        sub_sids = scen_ids_all[:ns]
        sub_scen = scen[scen["scenario_id"].isin(sub_sids)].copy()
        sub_scen = sub_scen[sub_scen["city_code_final"].astype(int).isin(good_cities)].copy()
        p = probs[:ns].copy()
        p = p / p.sum()

        scen_totals = sub_scen.groupby("scenario_id")["demand_ton"].sum().to_dict()
        demand_vec = np.array([float(scen_totals.get(sid, 0.0)) for sid in sub_sids], dtype=float)
        expected_demand_total = float(np.dot(p, demand_vec))
        total_capacity = float(sub_fac["capacity_max_ton"].sum())
        cap_feasible_cov = 0.98 * total_capacity / max(expected_demand_total, 1.0)
        eff_cov = float(min(base_policy["coverage"], max(0.70, cap_feasible_cov)))
        min_open_eff = min(24, len(sub_fac))
        max_open_eff = min(96, len(sub_fac))

        instances.append(
            {
                "n_facility": len(sub_fac),
                "n_city": len(sub_city),
                "n_scenario": ns,
                "size_metric": len(sub_city) * len(sub_fac) * ns,
                "city": sub_city,
                "fac": sub_fac,
                "edge": sub_edge,
                "scen": sub_scen,
                "probs": p,
                "coverage_ratio": eff_cov,
                "min_open_facilities": min_open_eff,
                "max_open_facilities": max_open_eff,
                "shortage_penalty": 460.0,
                "landfill_penalty": base_policy["landfill"],
                "subsidy_per_ton": base_policy["subsidy"],
                "carbon_price_per_tco2": base_policy["carbon"],
            }
        )
    return instances


def baseline_scalability_config(instance: Dict[str, object]) -> Dict[str, object]:
    """Reference single-cut settings used for the scalability baseline."""
    return {
        "max_iterations": 25,
        "tolerance": 0.02,
        "coverage_ratio": float(instance["coverage_ratio"]),
        "min_open_facilities": int(instance["min_open_facilities"]),
        "max_open_facilities": int(instance["max_open_facilities"]),
        "shortage_penalty": float(instance["shortage_penalty"]),
        "landfill_penalty": float(instance["landfill_penalty"]),
        "subsidy_per_ton": float(instance["subsidy_per_ton"]),
        "carbon_price_per_tco2": float(instance["carbon_price_per_tco2"]),
    }


def accelerated_scalability_config(instance: Dict[str, object]) -> Dict[str, object]:
    """Tuned screened multi-cut schedule for three-scale journal benchmarks.

    The schedule is deliberately compact: warm start is disabled because the
    ablation/tuning runs show it can slow down medium and large instances by
    over-seeding the master. Tight trust regions and a small bounded cut pool
    deliver gap <= 0.01 within two iterations across all three scales.
    """
    return {
        "max_iterations": 2,
        "tolerance": 0.01,
        "coverage_ratio": float(instance["coverage_ratio"]),
        "min_open_facilities": int(instance["min_open_facilities"]),
        "max_open_facilities": int(instance["max_open_facilities"]),
        "shortage_penalty": float(instance["shortage_penalty"]),
        "landfill_penalty": float(instance["landfill_penalty"]),
        "subsidy_per_ton": float(instance["subsidy_per_ton"]),
        "carbon_price_per_tco2": float(instance["carbon_price_per_tco2"]),
        "multicut": True,
        "warm_start": False,
        "stabilization": True,
        "trust_region_start": 0.03,
        "trust_region_expand": 1.10,
        "trust_region_max": 1.00,
        "stall_iter_expand": 2,
        "cut_violation_tol": 1e-2,
        "max_cuts_per_scenario": 4,
    }
