#!/usr/bin/env python3
"""Run the policy-dose and robustness extension experiments for the CDW study.

These are the analyses reported in the manuscript as the channel-leverage
comparison, the base-shortage-cost threshold analysis, the out-of-sample policy
ranking, the scenario-specification sensitivity, the demand-coefficient grid,
and the MILP size/certification summary.

Run the core workflow first so that the primary results and facility solutions
exist:

    PYTHONPATH=code python code/run_rigorous_revision.py

Then run one family, or all of them:

    PYTHONPATH=code python code/run_policy_extensions.py --only dose
    PYTHONPATH=code python code/run_policy_extensions.py --only shortage
    PYTHONPATH=code python code/run_policy_extensions.py --only ranking
    PYTHONPATH=code python code/run_policy_extensions.py --only specification
    PYTHONPATH=code python code/run_policy_extensions.py --only coefficients
    PYTHONPATH=code python code/run_policy_extensions.py --only statistics
    PYTHONPATH=code python code/run_policy_extensions.py --only all

Experiment families
-------------------
dose           Extended carbon-price sweep toward the iso-leverage price, plus
               the shortage-margin leverage of each instrument at each dose.
shortage       Base-shortage-cost fine grid solved under the baseline and the
               out-of-network-charge cases (threshold analysis).
ranking        The four primary policy designs re-evaluated on the common
               500-scenario holdout (out-of-sample policy ordering).
specification  Scenario-weight and macro-region-correlation sensitivity, each
               retrained with independent seeds and evaluated on the holdout.
coefficients   Two-dimensional sweep of the demand coefficients, applied as a
               structure-weighted demand scaling (see the documented constant
               DEMOLITION_TO_CONSTRUCTION_AREA_RATIO).
statistics     MILP size and solver certifications, plus Monte Carlo standard
               errors of the holdout service summaries.
"""

from __future__ import annotations

import argparse
import json
import math
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
from scenario_generator import generate_scenarios
from stochastic_analysis import build_evaluation_context, evaluate_design, read_design


SHORTAGE_EMISSION_TCO2_PER_T = 0.35
CHARGE_LEVEL_CNY_PER_T = 260.0
HOLDOUT_SIZE = 500
HOLDOUT_SEED = 90_260_713
SPECIFICATION_SEED_BASE = 50_260_713

# The upstream demand projection infers demolition area as a fixed ratio of
# completed new-construction area within each year (0.80 in the reference year
# 2022). With coefficients 0.12 and 2.19 t/m2, the new-construction share of
# estimated demand is therefore 0.12 / (0.12 + 2.19 * 0.80) = 0.0641, and a
# coefficient pair (alpha, delta) scales every city's demand by the
# structure-weighted multiplier r * alpha + (1 - r) * delta.
DEMOLITION_TO_CONSTRUCTION_AREA_RATIO = 0.80
DEMAND_COEFFICIENT_CONSTRUCTION = 0.12
DEMAND_COEFFICIENT_DEMOLITION = 2.19
NEW_CONSTRUCTION_COST_SHARE = DEMAND_COEFFICIENT_CONSTRUCTION / (
    DEMAND_COEFFICIENT_CONSTRUCTION
    + DEMAND_COEFFICIENT_DEMOLITION * DEMOLITION_TO_CONSTRUCTION_AREA_RATIO
)

PRIMARY_POLICIES = (
    ("baseline", "Baseline", PolicySpec("baseline", "extension_ranking")),
    ("carbon_tax_120", "Carbon price 120", PolicySpec("carbon_120", "extension_ranking", carbon_price=120.0)),
    (
        "landfill_penalty_260",
        "Out-of-network charge 260",
        PolicySpec("charge_260", "extension_ranking", landfill_penalty=CHARGE_LEVEL_CNY_PER_T),
    ),
    (
        "combined_120_260",
        "Combined 120 + 260",
        PolicySpec("combined", "extension_ranking", carbon_price=120.0, landfill_penalty=CHARGE_LEVEL_CNY_PER_T),
    ),
)


def _require_primary_results() -> pd.DataFrame:
    path = OUT_DIR / "single_period_exact_results.csv"
    if not path.exists():
        raise SystemExit(
            f"Primary results not found at {path}. Run "
            "`PYTHONPATH=code python code/run_rigorous_revision.py` first."
        )
    return pd.read_csv(path)


def _ensure_dirs() -> None:
    for directory in (DATA_DIR, OUT_DIR, LOG_DIR, SOLUTION_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _load_holdout(bundle: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Rebuild the common 500-scenario holdout used by the stability workflow."""
    path = DATA_DIR / f"scenario_stability_holdout_{HOLDOUT_SIZE}.csv"
    prob_path = DATA_DIR / f"scenario_stability_holdout_{HOLDOUT_SIZE}_probabilities.npy"
    if path.exists() and prob_path.exists():
        return pd.read_csv(path), np.load(prob_path)
    holdout, holdout_probabilities = generate_scenarios(
        bundle["pred"],
        bundle["city"],
        scenario_count=HOLDOUT_SIZE,
        seed=HOLDOUT_SEED,
        scenario_prefix="H",
    )
    holdout["data_version"] = VERSION
    holdout.to_csv(path, index=False)
    np.save(prob_path, holdout_probabilities)
    return holdout, holdout_probabilities


def _shortage_margin_leverage(bundle: dict, baseline_row: pd.Series) -> tuple[float, float]:
    """Treated-flow emission intensity and the iso-leverage carbon price.

    The out-of-network charge raises the unit shortage cost by its full face
    value. A carbon price raises it by tau * (e_short - e_bar), where e_bar is
    the emission intensity of the marginal treated ton under the baseline
    design (transport plus processing).
    """
    expected_emission = float(baseline_row["expected_emission_tco2"])
    expected_shortage = float(baseline_row["expected_shortage_ton"])
    scenario = bundle["scenario"]
    probs = np.asarray(bundle["probs"], dtype=float)
    per_scenario = scenario.groupby("scenario_id")["demand_ton"].sum().to_numpy(float)
    total_demand = float(probs @ per_scenario)
    treated = max(total_demand - expected_shortage, 1e-9)
    e_bar = (expected_emission - SHORTAGE_EMISSION_TCO2_PER_T * expected_shortage) / treated
    iso_leverage_price = CHARGE_LEVEL_CNY_PER_T / (SHORTAGE_EMISSION_TCO2_PER_T - e_bar)
    return e_bar, iso_leverage_price


def run_dose(bundle: dict, baseline_row: pd.Series) -> pd.DataFrame:
    """Extended carbon sweep plus the per-dose shortage-margin leverage table."""
    e_bar, iso_price = _shortage_margin_leverage(bundle, baseline_row)
    print(f"[dose] treated emission intensity e_bar={e_bar:.4f} tCO2/t; iso-leverage tau={iso_price:.0f}")

    rows = []
    carbon_points = sorted({300.0, 500.0, 700.0, float(round(iso_price / 10.0) * 10)})
    for value in carbon_points:
        spec = PolicySpec(f"carbon_{int(value)}", "extension_dose", carbon_price=value, parameter_value=value)
        print(f"[dose] carbon {value:.0f} CNY/tCO2")
        rows.append(solve_single_period(bundle, spec, time_limit=600))
    df = pd.DataFrame(rows)
    leverage = SHORTAGE_EMISSION_TCO2_PER_T - e_bar
    df["shortage_margin_leverage_cny_per_t"] = df["parameter_value"].astype(float) * leverage
    baseline_objective = float(baseline_row["reevaluated_objective_yuan"])
    baseline_shortage = float(baseline_row["expected_shortage_ton"])
    df["cost_change_pct_vs_primary_baseline"] = (df["reevaluated_objective_yuan"] / baseline_objective - 1.0) * 100.0
    df["shortage_reduction_pct_vs_primary_baseline"] = (1.0 - df["expected_shortage_ton"] / baseline_shortage) * 100.0
    df.to_csv(OUT_DIR / "policy_extension_dose_response.csv", index=False)
    summary = {
        "treated_emission_intensity_tco2_per_t": e_bar,
        "iso_leverage_carbon_price_cny_per_tco2": iso_price,
        "shortage_margin_leverage_per_unit_carbon_price": leverage,
    }
    (OUT_DIR / "policy_extension_dose_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return df


def run_shortage(bundle: dict, baseline_row: pd.Series) -> pd.DataFrame:
    """Base-shortage-cost fine grid under the baseline and charge cases."""
    rows = []
    for value in (340.0, 400.0, 520.0, 580.0):
        for tag, charge in (("base", 0.0), ("charge", CHARGE_LEVEL_CNY_PER_T)):
            spec = PolicySpec(
                f"shortage_{int(value)}_{tag}",
                "extension_shortage",
                base_shortage_penalty=value,
                landfill_penalty=charge,
                parameter_value=value,
            )
            print(f"[shortage] base {value:.0f} CNY/t, {tag} case")
            rows.append(solve_single_period(bundle, spec, time_limit=600))
    df = pd.DataFrame(rows)
    baseline_objective = float(baseline_row["reevaluated_objective_yuan"])
    baseline_shortage = float(baseline_row["expected_shortage_ton"])
    df["cost_change_pct_vs_primary_baseline"] = (df["reevaluated_objective_yuan"] / baseline_objective - 1.0) * 100.0
    df["shortage_reduction_pct_vs_primary_baseline"] = (1.0 - df["expected_shortage_ton"] / baseline_shortage) * 100.0
    df.to_csv(OUT_DIR / "policy_extension_shortage_grid.csv", index=False)
    return df


def run_ranking(bundle: dict, baseline_row: pd.Series) -> pd.DataFrame:
    """Evaluate the four primary designs on the common holdout."""
    holdout, holdout_probabilities = _load_holdout(bundle)
    rows = []
    for solution_name, label, spec in PRIMARY_POLICIES:
        solution_path = SOLUTION_DIR / f"single_{solution_name}_facility_solution.csv"
        if not solution_path.exists():
            raise SystemExit(
                f"Facility solution not found at {solution_path}. Run the core "
                "workflow first so the primary designs exist."
            )
        open_decision, capacity = read_design(solution_path, bundle["facility"])
        context = build_evaluation_context(bundle, spec)
        evaluation = evaluate_design(context, holdout, holdout_probabilities, open_decision, capacity)
        rows.append(
            {
                "policy": label,
                "holdout_expected_objective_yuan": float(evaluation["expected_total_cost_yuan"]),
                "holdout_expected_shortage_ton": float(evaluation["expected_shortage_ton"]),
                "holdout_expected_emission_tco2": float(evaluation["expected_emission_tco2"]),
                "open_facilities": int((open_decision > 0.5).sum()),
                "holdout_size": HOLDOUT_SIZE,
            }
        )
        print(f"[ranking] {label}: holdout shortage {rows[-1]['holdout_expected_shortage_ton']:.0f} t")
    df = pd.DataFrame(rows)
    baseline_objective = float(baseline_row["reevaluated_objective_yuan"])
    baseline_shortage = float(baseline_row["expected_shortage_ton"])
    base_row = df[df["policy"] == "Baseline"].iloc[0]
    df["holdout_objective_increase_pct_vs_baseline"] = (
        df["holdout_expected_objective_yuan"] / float(base_row["holdout_expected_objective_yuan"]) - 1.0
    ) * 100.0
    df["holdout_shortage_reduction_pct_vs_baseline"] = (
        1.0 - df["holdout_expected_shortage_ton"] / float(base_row["holdout_expected_shortage_ton"])
    ) * 100.0
    df.to_csv(OUT_DIR / "policy_extension_holdout_ranking.csv", index=False)
    return df


def run_specification(bundle: dict, replicates: int = 5) -> pd.DataFrame:
    """Scenario-weight and macro-region-correlation sensitivity."""
    holdout, holdout_probabilities = _load_holdout(bundle)
    weight_variants = {
        "near_uniform": {"baseline": 6 / 16, "strong_management": 5 / 16, "high_growth": 5 / 16},
        "baseline_heavy": {"baseline": 12 / 16, "strong_management": 2 / 16, "high_growth": 2 / 16},
    }
    correlation_base = np.array([[1.0, 0.65, 0.45], [0.65, 1.0, 0.50], [0.45, 0.50, 1.0]])
    off_diagonal = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    variants = [
        ("weight_near_uniform", {"weights": weight_variants["near_uniform"]}),
        ("weight_baseline_heavy", {"weights": weight_variants["baseline_heavy"]}),
        ("correlation_minus_015", {"correlation": correlation_base - 0.15 * off_diagonal}),
        ("correlation_plus_015", {"correlation": correlation_base + 0.15 * off_diagonal}),
    ]
    rows = []
    for variant_index, (variant, kwargs) in enumerate(variants):
        for replicate in range(1, replicates + 1):
            seed = SPECIFICATION_SEED_BASE + variant_index * 100 + replicate
            print(f"[specification] {variant} replicate {replicate} seed={seed}")
            scenario, probabilities = generate_scenarios(
                bundle["pred"], bundle["city"], scenario_count=16, seed=seed, scenario_prefix="V", **kwargs
            )
            training_bundle = {**bundle, "scenario": scenario, "probs": probabilities}
            spec = PolicySpec(f"specification_{variant}_r{replicate:02d}", "extension_specification")
            result = solve_single_period(training_bundle, spec, time_limit=600)
            open_decision, capacity = read_design(
                SOLUTION_DIR / f"single_{spec.name}_facility_solution.csv", bundle["facility"]
            )
            context = build_evaluation_context(bundle, PolicySpec(f"{spec.name}_eval", "extension_specification_eval"))
            evaluation = evaluate_design(context, holdout, holdout_probabilities, open_decision, capacity)
            rows.append(
                {
                    "variant": variant,
                    "replicate": replicate,
                    "training_seed": seed,
                    "open_facilities": int(result["open_facilities"]),
                    "holdout_expected_objective_yuan": float(evaluation["expected_total_cost_yuan"]),
                    "holdout_expected_shortage_ton": float(evaluation["expected_shortage_ton"]),
                    "solver_status": result["solver_status"],
                    "solver_gap": float(result["solver_gap"]),
                }
            )
            pd.DataFrame(rows).to_csv(OUT_DIR / "policy_extension_specification.csv", index=False)
    return pd.DataFrame(rows)


def run_coefficients(bundle: dict, baseline_row: pd.Series) -> pd.DataFrame:
    """Two-dimensional demand-coefficient sweep as a structure-weighted scaling."""
    print(
        f"[coefficients] new-construction cost share r={NEW_CONSTRUCTION_COST_SHARE:.4f} "
        f"under demolition/new-construction area ratio {DEMOLITION_TO_CONSTRUCTION_AREA_RATIO:.2f}"
    )
    rows = []
    for alpha in (0.75, 1.00, 1.25):
        for delta in (0.75, 1.00, 1.25):
            multiplier = NEW_CONSTRUCTION_COST_SHARE * alpha + (1.0 - NEW_CONSTRUCTION_COST_SHARE) * delta
            spec = PolicySpec(
                f"coefficient_a{alpha:.2f}_d{delta:.2f}",
                "extension_coefficients",
                demand_scale=multiplier,
                parameter_value=multiplier,
            )
            print(f"[coefficients] alpha={alpha:.2f} delta={delta:.2f} demand multiplier={multiplier:.4f}")
            rows.append(solve_single_period(bundle, spec, time_limit=900))
    df = pd.DataFrame(rows)
    df["coefficient_alpha_new_construction"] = [0.75, 0.75, 0.75, 1.00, 1.00, 1.00, 1.25, 1.25, 1.25]
    df["coefficient_delta_demolition"] = [0.75, 1.00, 1.25] * 3
    baseline_objective = float(baseline_row["reevaluated_objective_yuan"])
    baseline_shortage = float(baseline_row["expected_shortage_ton"])
    df["cost_change_pct_vs_primary_baseline"] = (df["reevaluated_objective_yuan"] / baseline_objective - 1.0) * 100.0
    df["shortage_reduction_pct_vs_primary_baseline"] = (1.0 - df["expected_shortage_ton"] / baseline_shortage) * 100.0
    df.to_csv(OUT_DIR / "policy_extension_coefficient_grid.csv", index=False)
    return df


def run_statistics(baseline_row: pd.Series) -> dict:
    """MILP size and certification summary plus Monte Carlo errors."""
    single = _require_primary_results()
    summary: dict = {
        "single_period_runs": int(len(single)),
        "all_optimal": bool((single["solver_status"] == "optimal").all()),
        "max_solver_gap": float(single["solver_gap"].max()),
        "max_reconstruction_relative_error": float(single["objective_crosscheck_relative_error"].max()),
        "runtime_sec_min": float(single["runtime_sec"].min()),
        "runtime_sec_median": float(single["runtime_sec"].median()),
        "runtime_sec_max": float(single["runtime_sec"].max()),
    }

    # Deterministic-equivalent size for the single-period model. Variable counts
    # follow from the model structure: |A| arcs x |S| scenario flows, |I| cities
    # x |S| scenario shortages, and |J| continuous capacities.
    city = pd.read_csv(DATA_DIR / "city_list_complete_case.csv")
    facility = pd.read_csv(DATA_DIR / "facility_list_transparent_proxy.csv")
    edge = pd.read_csv(DATA_DIR / "road_transport_edges.csv")
    scenario = pd.read_csv(DATA_DIR / "stochastic_scenarios_2030.csv")
    n_scenarios = int(scenario["scenario_id"].nunique())
    summary["deterministic_equivalent_size"] = {
        "constraints_from_lp_artifact": 34687,
        "continuous_variables": int(len(edge) * n_scenarios + len(city) * n_scenarios + len(facility)),
        "binary_variables": int(len(facility)),
        "arcs": int(len(edge)),
        "cities": int(len(city)),
        "scenarios": n_scenarios,
    }

    stability_path = OUT_DIR / "scenario_stability_exact_results.csv"
    if stability_path.exists():
        stability = pd.read_csv(stability_path)
        summary["stability_runs"] = int(len(stability))
        summary["stability_all_optimal"] = bool((stability["solver_status"] == "optimal").all())
        mc = {}
        for count, group in stability.groupby("scenario_count"):
            values = group["holdout_expected_shortage_ton"].to_numpy(float)
            mc[str(int(count))] = {
                "n_runs": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "monte_carlo_standard_error_of_mean": float(values.std(ddof=1) / math.sqrt(len(values))),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
            }
        summary["holdout_shortage_monte_carlo_by_training_size"] = mc

    (OUT_DIR / "policy_extension_statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:1500])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        choices=["dose", "shortage", "ranking", "specification", "coefficients", "statistics", "all"],
        default="all",
    )
    parser.add_argument("--replicates", type=int, default=5, help="independent seeds per specification variant")
    args = parser.parse_args()

    _ensure_dirs()
    bundle = build_revision_inputs()

    primary = _require_primary_results()
    baseline_row = primary[(primary["group"] == "primary") & (primary["name"] == "baseline")].iloc[0]

    if args.only in ("all", "dose"):
        run_dose(bundle, baseline_row)
    if args.only in ("all", "shortage"):
        run_shortage(bundle, baseline_row)
    if args.only in ("all", "ranking"):
        run_ranking(bundle, baseline_row)
    if args.only in ("all", "specification"):
        run_specification(bundle, replicates=args.replicates)
    if args.only in ("all", "coefficients"):
        run_coefficients(bundle, baseline_row)
    if args.only in ("all", "statistics"):
        run_statistics(baseline_row)


if __name__ == "__main__":
    main()
