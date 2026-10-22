#!/usr/bin/env python3
"""Run the second-revision verification experiments for the CDW study.

These are the exact-verification solves reported in the manuscript as the
homogeneous-emission equivalence check (Section 5.2) and the charge-versus-
base-cost identity check (Section 5.3), together with the treated-emission
intensity statistics (design average versus switched-ton estimates).

Run the core workflow first so that the primary results and facility
solutions exist:

    PYTHONPATH=code python code/run_rigorous_revision.py

The intensity statistics additionally read the extended carbon sweep:

    PYTHONPATH=code python code/run_policy_extensions.py --only dose

Then run one family, or all of them:

    PYTHONPATH=code python code/run_revision2_verifications.py --only homogeneous
    PYTHONPATH=code python code/run_revision2_verifications.py --only identity
    PYTHONPATH=code python code/run_revision2_verifications.py --only intensities
    PYTHONPATH=code python code/run_revision2_verifications.py --only all

Experiment families
-------------------
homogeneous  Under a uniform emission field (every arc's transport-plus-
             processing intensity set to the design average e_bar), the
             charge case and the carbon case at the exact iso-leverage price
             tau* = lambda / (e_short - e_bar) must share an optimal
             solution: the demand-balance identity makes their objectives
             differ only by tau* * e_bar * E[demand]. Verified by solving
             both cases and comparing open sets, capacities, shortage, and
             the objective difference against the predicted constant.
identity     In h_i the base shortage cost and the out-of-network charge
             enter additively, so the charge case at base b is the same
             mathematical program as the uncharged case at base b + 260.
             Verified by solving baseline at 600 and 720 CNY/t and the
             charge case at base 340, and comparing against the primary
             charge case (base 460).
intensities  Design-average and switched-ton treated-emission intensities
             computed from existing solutions (no new solves): the average
             under the frozen baseline, and finite-difference estimates of
             the tons that enter treatment between the baseline and the
             carbon-price optima (tau = 120 and tau = 890).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from release_layout import OUT_DIR
from run_rigorous_revision import (
    PolicySpec,
    SOLUTION_DIR,
    build_revision_inputs,
    solve_single_period,
)
from stochastic_analysis import read_design

SHORTAGE_EMISSION_TCO2_PER_T = 0.35
CHARGE_LEVEL_CNY_PER_T = 260.0


def _require_primary_results() -> pd.DataFrame:
    path = OUT_DIR / "single_period_exact_results.csv"
    if not path.exists():
        raise SystemExit(
            "Primary results are missing. Run the core workflow first:\n"
            "    PYTHONPATH=code python code/run_rigorous_revision.py"
        )
    return pd.read_csv(path)


def _primary_row(df: pd.DataFrame, name: str) -> pd.Series:
    row = df[(df["group"] == "primary") & (df["name"] == name)]
    if row.empty:
        raise SystemExit(f"Primary row '{name}' not found in single_period_exact_results.csv")
    return row.iloc[0]


def _design_average_intensity(bundle: dict, baseline_row: pd.Series) -> tuple[float, float, float]:
    """Design-average treated-flow intensity under the frozen baseline.

    e_bar = (E[emissions] - e_short * E[shortage]) / E[treated], with the
    expectation over the planning scenarios. Returns (e_bar, expected total
    demand, iso-leverage carbon price).
    """
    scenario: pd.DataFrame = bundle["scenario"]
    probs: np.ndarray = bundle["probs"]
    group_col = "scenario_id" if "scenario_id" in scenario.columns else scenario.columns[0]
    per_scen = scenario.groupby(group_col)["demand_ton"].sum()
    total_demand = float(probs @ per_scen.to_numpy(float))
    emission = float(baseline_row["expected_emission_tco2"])
    shortage = float(baseline_row["expected_shortage_ton"])
    treated = max(total_demand - shortage, 1e-9)
    e_bar = (emission - SHORTAGE_EMISSION_TCO2_PER_T * shortage) / treated
    iso_price = CHARGE_LEVEL_CNY_PER_T / (SHORTAGE_EMISSION_TCO2_PER_T - e_bar)
    return e_bar, total_demand, iso_price


def _open_set_and_capacity(name: str, facility_df: pd.DataFrame):
    sol = SOLUTION_DIR / f"single_{name}_facility_solution.csv"
    if not sol.exists():
        raise SystemExit(f"Facility solution '{sol}' is missing; run the core workflow first.")
    return read_design(sol, facility_df)


def _switched_ton_intensity(base: pd.Series, case: pd.Series) -> float:
    """Average emission intensity of the tons entering treatment.

    Finite-difference estimate between two optima:
    (d_emission - e_short * d_shortage) / d_treated, where treated = demand -
    shortage, so d_treated = -d_shortage.
    """
    d_shortage = float(case["expected_shortage_ton"]) - float(base["expected_shortage_ton"])
    d_emission = float(case["expected_emission_tco2"]) - float(base["expected_emission_tco2"])
    d_treated = -d_shortage
    return (d_emission - SHORTAGE_EMISSION_TCO2_PER_T * d_shortage) / d_treated


def run_homogeneous(bundle: dict, baseline_row: pd.Series) -> dict:
    fac = bundle["facility"].copy()
    edge = bundle["edge"].copy()
    e_bar, total_demand, iso_price = _design_average_intensity(bundle, baseline_row)
    predicted_constant = iso_price * e_bar * total_demand
    print(f"[homogeneous] e_bar={e_bar:.6f} tCO2/t; tau*={iso_price:.4f}; "
          f"E[demand]={total_demand:.0f} t; predicted constant={predicted_constant:.6e}")

    fac["processing_emission_tco2_per_t"] = e_bar
    edge["transport_emission_tco2_per_t"] = 0.0
    uniform_bundle = {**bundle, "facility": fac, "edge": edge}

    specs = [
        PolicySpec("r2_charge260_uniform", "r2_homogeneous", landfill_penalty=CHARGE_LEVEL_CNY_PER_T),
        PolicySpec("r2_carbon_iso_uniform", "r2_homogeneous", carbon_price=iso_price),
    ]
    rows = []
    for spec in specs:
        print(f"[solve] {spec.name}", flush=True)
        rows.append(solve_single_period(uniform_bundle, spec, time_limit=900))
    pd.DataFrame(rows).to_csv(OUT_DIR / "revision2_homogeneous_results.csv", index=False)

    by_name = {r["name"]: r for r in rows}
    open_charge, cap_charge = _open_set_and_capacity("r2_charge260_uniform", bundle["facility"])
    open_carbon, cap_carbon = _open_set_and_capacity("r2_carbon_iso_uniform", bundle["facility"])
    union = float(((open_charge > 0.5) | (open_carbon > 0.5)).sum())
    inter = float(((open_charge > 0.5) & (open_carbon > 0.5)).sum())
    objective_diff = float(by_name["r2_carbon_iso_uniform"]["cplex_objective_yuan"]
                           - by_name["r2_charge260_uniform"]["cplex_objective_yuan"])
    result = {
        "e_bar_tco2_per_t": e_bar,
        "iso_leverage_price_cny_per_tco2": iso_price,
        "expected_total_demand_ton": total_demand,
        "jaccard_open_sets": inter / union if union else 1.0,
        "max_capacity_difference_ton": float(np.max(np.abs(cap_charge - cap_carbon))),
        "shortage_difference_ton": float(by_name["r2_carbon_iso_uniform"]["expected_shortage_ton"]
                                         - by_name["r2_charge260_uniform"]["expected_shortage_ton"]),
        "objective_difference_yuan": objective_diff,
        "predicted_constant_yuan": predicted_constant,
        "objective_difference_relative_error": abs(objective_diff - predicted_constant) / predicted_constant,
    }
    print("[homogeneous]", json.dumps(result, indent=2))
    return result


def run_identity(bundle: dict, primary: pd.DataFrame) -> dict:
    specs = [
        PolicySpec("r2_base600", "r2_identity", base_shortage_penalty=600.0),
        PolicySpec("r2_base720", "r2_identity", base_shortage_penalty=720.0),
        PolicySpec("r2_charge_base340", "r2_identity", base_shortage_penalty=340.0,
                   landfill_penalty=CHARGE_LEVEL_CNY_PER_T),
    ]
    rows = []
    for spec in specs:
        print(f"[solve] {spec.name}", flush=True)
        rows.append(solve_single_period(bundle, spec, time_limit=900))
    pd.DataFrame(rows).to_csv(OUT_DIR / "revision2_identity_results.csv", index=False)

    by_name = {r["name"]: r for r in rows}
    charge_primary = _primary_row(primary, "landfill_penalty_260")

    def compare(a_name: str, a_row: pd.Series, b_row: pd.Series) -> dict:
        open_a, cap_a = _open_set_and_capacity(a_name, bundle["facility"])
        open_b, cap_b = _open_set_and_capacity(str(b_row["name"]), bundle["facility"])
        return {
            "open_sites_identical": bool(np.array_equal(open_a > 0.5, open_b > 0.5)),
            "max_capacity_difference_ton": float(np.max(np.abs(cap_a - cap_b))),
            "shortage_absolute_difference_ton": abs(float(a_row["expected_shortage_ton"])
                                                    - float(b_row["expected_shortage_ton"])),
            "objective_absolute_difference_yuan": abs(float(a_row["cplex_objective_yuan"])
                                                      - float(b_row["cplex_objective_yuan"])),
        }

    result = {
        "baseline_600_equals_charge_base340": compare("r2_base600", by_name["r2_base600"],
                                                      by_name["r2_charge_base340"]),
        "baseline_720_equals_charge_base460_primary": compare("r2_base720", by_name["r2_base720"],
                                                              charge_primary),
    }
    print("[identity]", json.dumps(result, indent=2))
    return result


def run_intensities(bundle: dict, primary: pd.DataFrame, e_bar: float) -> dict:
    edge = bundle["edge"]
    fac = bundle["facility"]
    fac_pos = pd.Series(np.arange(len(fac)), index=fac["site_id"].astype(int))
    per_arc = (edge["transport_emission_tco2_per_t"].to_numpy(float)
               + fac["processing_emission_tco2_per_t"].to_numpy(float)[
                   edge["site_id"].astype(int).map(fac_pos).to_numpy(int)])
    baseline = _primary_row(primary, "baseline")
    carbon120 = _primary_row(primary, "carbon_tax_120")

    stats = {
        "design_average_baseline_e_bar": e_bar,
        "design_average_carbon120": _switched_free_average(bundle, carbon120),
        "switched_ton_intensity_baseline_to_carbon120": _switched_ton_intensity(baseline, carbon120),
    }
    dose_path = OUT_DIR / "policy_extension_dose_response.csv"
    if dose_path.exists():
        dose = pd.read_csv(dose_path)
        row890 = dose[dose["name"].astype(str).str.contains("890", regex=False)]
        if not row890.empty:
            stats["switched_ton_intensity_baseline_to_carbon890"] = _switched_ton_intensity(
                baseline, row890.iloc[0])
    else:
        print("[intensities] dose sweep not found; run run_policy_extensions.py --only dose "
              "for the tau=890 estimate")
    stats["arc_intensity_min"] = float(per_arc.min())
    stats["arc_intensity_median"] = float(np.median(per_arc))
    stats["arc_intensity_max"] = float(per_arc.max())
    print("[intensities]", json.dumps(stats, indent=2))
    return stats


def _switched_free_average(bundle: dict, row: pd.Series) -> float:
    """Design-average intensity recomputed under another design's solution."""
    emission = float(row["expected_emission_tco2"])
    shortage = float(row["expected_shortage_ton"])
    scenario: pd.DataFrame = bundle["scenario"]
    probs: np.ndarray = bundle["probs"]
    group_col = "scenario_id" if "scenario_id" in scenario.columns else scenario.columns[0]
    per_scen = scenario.groupby(group_col)["demand_ton"].sum()
    total_demand = float(probs @ per_scen.to_numpy(float))
    return (emission - SHORTAGE_EMISSION_TCO2_PER_T * shortage) / max(total_demand - shortage, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["homogeneous", "identity", "intensities", "all"],
                    default="all")
    args = ap.parse_args()

    bundle = build_revision_inputs()
    primary = _require_primary_results()
    baseline_row = _primary_row(primary, "baseline")
    e_bar, _, _ = _design_average_intensity(bundle, baseline_row)

    summary: dict = {"data_version": str(OUT_DIR)}
    if args.only in ("all", "homogeneous"):
        summary["homogeneous_emission_equivalence"] = run_homogeneous(bundle, baseline_row)
    if args.only in ("all", "identity"):
        summary["charge_base_cost_identity"] = run_identity(bundle, primary)
    if args.only in ("all", "intensities"):
        summary["treated_intensity_stats"] = run_intensities(bundle, primary, e_bar)

    (OUT_DIR / "revision2_verification_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {OUT_DIR / 'revision2_verification_summary.json'}")


if __name__ == "__main__":
    main()
