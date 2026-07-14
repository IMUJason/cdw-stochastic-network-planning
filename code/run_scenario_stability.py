#!/usr/bin/env python3
"""Run exact scenario-count and independent-training-set stability experiments."""

from __future__ import annotations

import argparse
from itertools import combinations

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
from scenario_generator import GENERATOR_VERSION, generate_scenarios
from stochastic_analysis import build_evaluation_context, evaluate_design, jaccard, read_design


DEFAULT_COUNTS = (8, 16, 32, 64)
DEFAULT_REPLICATES = 10
DEFAULT_HOLDOUT_SIZE = 500
HOLDOUT_SEED = 90_260_713
TRAINING_SEED_BASE = 30_260_713


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=list(DEFAULT_COUNTS))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--holdout-size", type=int, default=DEFAULT_HOLDOUT_SIZE)
    return parser.parse_args()


def training_seed(scenario_count: int, replicate: int) -> int:
    return TRAINING_SEED_BASE + int(scenario_count) * 1_000 + int(replicate)


def empirical_summary(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    values = frame[metric].to_numpy(float)
    return {
        f"{metric}_mean": float(values.mean()),
        f"{metric}_median": float(np.median(values)),
        f"{metric}_p10": float(np.quantile(values, 0.10)),
        f"{metric}_p90": float(np.quantile(values, 0.90)),
        f"{metric}_min": float(values.min()),
        f"{metric}_max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    counts = tuple(sorted(set(args.counts)))
    if args.replicates < 2:
        raise ValueError("At least two training replications are required")
    for directory in (DATA_DIR, OUT_DIR, LOG_DIR, SOLUTION_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    bundle = build_revision_inputs()
    baseline = PolicySpec("baseline", "primary")
    context = build_evaluation_context(bundle, baseline)
    holdout, holdout_probabilities = generate_scenarios(
        bundle["pred"],
        bundle["city"],
        scenario_count=args.holdout_size,
        seed=HOLDOUT_SEED,
        scenario_prefix="H",
    )
    holdout["data_version"] = VERSION
    holdout.to_csv(DATA_DIR / f"scenario_stability_holdout_{args.holdout_size}.csv", index=False)
    np.save(DATA_DIR / f"scenario_stability_holdout_{args.holdout_size}_probabilities.npy", holdout_probabilities)

    seed_rows = [
        {
            "scenario_count": count,
            "replicate": replicate,
            "training_seed": training_seed(count, replicate),
            "holdout_size": args.holdout_size,
            "holdout_seed": HOLDOUT_SEED,
            "generator_version": GENERATOR_VERSION,
            "data_version": VERSION,
        }
        for count in counts
        for replicate in range(1, args.replicates + 1)
    ]
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(OUT_DIR / "scenario_stability_seeds.csv", index=False)

    checkpoint_path = OUT_DIR / "scenario_stability_checkpoint.csv"
    completed = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    rows = completed.to_dict("records") if not completed.empty else []
    completed_keys = (
        set(zip(completed.scenario_count.astype(int), completed.replicate.astype(int)))
        if not completed.empty
        else set()
    )

    for count in counts:
        for replicate in range(1, args.replicates + 1):
            if (count, replicate) in completed_keys:
                continue
            seed = training_seed(count, replicate)
            print(f"[scenario stability] n={count} replicate={replicate}/{args.replicates} seed={seed}", flush=True)
            scenario, probabilities = generate_scenarios(
                bundle["pred"],
                bundle["city"],
                scenario_count=count,
                seed=seed,
                scenario_prefix="T",
            )
            scenario["data_version"] = VERSION
            training_bundle = {**bundle, "scenario": scenario, "probs": probabilities}
            name = f"stability_n{count:03d}_r{replicate:02d}"
            spec = PolicySpec(name, "scenario_stability")
            saved_solution = SOLUTION_DIR / f"single_{name}_facility_solution.csv"
            result = solve_single_period(
                training_bundle,
                spec,
                time_limit=1200 if count >= 64 else 300,
                mip_start_solution_csv=saved_solution if saved_solution.exists() else None,
            )
            if result.get("solver_status") != "optimal":
                raise RuntimeError(f"Training problem {name} was not solved to optimality")
            open_decision, capacity = read_design(
                SOLUTION_DIR / f"single_{name}_facility_solution.csv",
                context.facility,
            )
            evaluation = evaluate_design(
                context,
                holdout,
                holdout_probabilities,
                open_decision,
                capacity,
            )
            rows.append(
                {
                    "scenario_count": count,
                    "replicate": replicate,
                    "training_seed": seed,
                    "training_objective_yuan": float(result["reevaluated_objective_yuan"]),
                    "holdout_expected_objective_yuan": float(evaluation["expected_total_cost_yuan"]),
                    "holdout_expected_shortage_ton": float(evaluation["expected_shortage_ton"]),
                    "holdout_expected_emission_tco2": float(evaluation["expected_emission_tco2"]),
                    "open_facilities": int((open_decision > 0.5).sum()),
                    "installed_capacity_ton": float(capacity.sum()),
                    "solver_status": result["solver_status"],
                    "solver_gap": float(result["solver_gap"]),
                    "runtime_sec": float(result["runtime_sec"]),
                    "objective_crosscheck_relative_error": float(result["objective_crosscheck_relative_error"]),
                    "holdout_size": args.holdout_size,
                    "holdout_seed": HOLDOUT_SEED,
                    "data_version": VERSION,
                }
            )
            pd.DataFrame(rows).sort_values(["scenario_count", "replicate"]).to_csv(checkpoint_path, index=False)

    results = pd.DataFrame(rows).sort_values(["scenario_count", "replicate"]).reset_index(drop=True)
    results["holdout_regret_pct_vs_best_observed_design"] = (
        results["holdout_expected_objective_yuan"] / results["holdout_expected_objective_yuan"].min() - 1.0
    ) * 100.0
    results.to_csv(OUT_DIR / "scenario_stability_exact_results.csv", index=False)

    primary_x, _ = read_design(SOLUTION_DIR / "single_baseline_facility_solution.csv", context.facility)
    frequency_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for count, frame in results.groupby("scenario_count", sort=True):
        designs: list[np.ndarray] = []
        capacities: list[np.ndarray] = []
        for replicate in frame["replicate"].astype(int):
            name = f"stability_n{int(count):03d}_r{replicate:02d}"
            open_decision, capacity = read_design(
                SOLUTION_DIR / f"single_{name}_facility_solution.csv",
                context.facility,
            )
            designs.append(open_decision)
            capacities.append(capacity)
        design_matrix = np.vstack(designs)
        capacity_matrix = np.vstack(capacities)
        pairwise = [jaccard(designs[first], designs[second]) for first, second in combinations(range(len(designs)), 2)]
        primary_similarity = [jaccard(design, primary_x) for design in designs]

        row: dict[str, object] = {
            "scenario_count": int(count),
            "training_replications": len(frame),
            "holdout_size": args.holdout_size,
            "pairwise_site_jaccard_mean": float(np.mean(pairwise)),
            "pairwise_site_jaccard_min": float(np.min(pairwise)),
            "site_jaccard_vs_primary_mean": float(np.mean(primary_similarity)),
            "site_jaccard_vs_primary_min": float(np.min(primary_similarity)),
            "data_version": VERSION,
        }
        for metric in (
            "holdout_expected_objective_yuan",
            "holdout_expected_shortage_ton",
            "holdout_expected_emission_tco2",
            "open_facilities",
            "installed_capacity_ton",
            "holdout_regret_pct_vs_best_observed_design",
        ):
            row.update(empirical_summary(frame, metric))
        summary_rows.append(row)

        for facility_index, facility in context.facility.reset_index(drop=True).iterrows():
            frequency_rows.append(
                {
                    "scenario_count": int(count),
                    "site_id": int(facility["site_id"]),
                    "facility_city": facility["facility_city"],
                    "province": facility["province"],
                    "selection_frequency": float(design_matrix[:, facility_index].mean()),
                    "mean_installed_capacity_ton": float(capacity_matrix[:, facility_index].mean()),
                    "p10_installed_capacity_ton": float(np.quantile(capacity_matrix[:, facility_index], 0.10)),
                    "p90_installed_capacity_ton": float(np.quantile(capacity_matrix[:, facility_index], 0.90)),
                    "training_replications": len(frame),
                    "data_version": VERSION,
                }
            )

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "scenario_stability_summary.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(OUT_DIR / "scenario_site_selection_frequency.csv", index=False)
    checkpoint_path.unlink(missing_ok=True)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
