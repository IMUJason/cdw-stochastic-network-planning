#!/usr/bin/env python3
"""Accelerated multi-cut Benders solver for the CDW network model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from benders_solver import build_index, prepare_edge_arrays, solve_subproblem, _build_subproblem_matrices


@dataclass
class AcceleratedBendersConfig:
    max_iterations: int = 80
    tolerance: float = 0.01
    coverage_ratio: float = 0.90
    min_open_facilities: int = 24
    max_open_facilities: int = 92
    shortage_penalty: float = 460.0
    landfill_penalty: float = 0.0
    subsidy_per_ton: float = 0.0
    carbon_price_per_tco2: float = 0.0
    shortage_emission_tco2_per_t: float = 0.35

    # Acceleration options
    multicut: bool = True
    warm_start: bool = True
    stabilization: bool = True
    trust_region_start: float = 0.12
    trust_region_expand: float = 1.25
    trust_region_max: float = 1.00
    stall_iter_expand: int = 3
    cut_violation_tol: float = 1e-5
    max_cuts_per_scenario: int = 40


@dataclass
class AcceleratedSolverResult:
    success: bool
    message: str
    iterations: int
    lower_bound: float
    upper_bound: float
    best_x: np.ndarray
    best_cap: np.ndarray
    best_theta_s: np.ndarray
    solver_log: pd.DataFrame
    expected_flow: np.ndarray
    expected_shortage: np.ndarray
    summary_metrics: Dict[str, float]


def _evaluate_solution(
    cap: np.ndarray,
    demand_matrix: np.ndarray,
    scenario_probs: np.ndarray,
    edge_arrays: Dict[str, np.ndarray],
    proc_cost: np.ndarray,
    proc_emission: np.ndarray,
    city_landfill_cost: np.ndarray,
    cfg: AcceleratedBendersConfig,
    mats: Tuple[np.ndarray, np.ndarray],
) -> Dict[str, object]:
    n_scen = len(scenario_probs)
    n_edge = len(edge_arrays["city_idx"])
    n_city = demand_matrix.shape[1]

    recourse = np.zeros(n_scen, dtype=float)
    theta_rhs = np.zeros(n_scen, dtype=float)
    cut_a = np.zeros(n_scen, dtype=float)
    cut_b = np.zeros((n_scen, cap.size), dtype=float)

    flow_acc = np.zeros(n_edge, dtype=float)
    shortage_acc = np.zeros(n_city, dtype=float)

    transport_cost = 0.0
    processing_cost = 0.0
    shortage_cost = 0.0
    subsidy_benefit = 0.0
    carbon_cost = 0.0
    total_emissions = 0.0

    for s in range(n_scen):
        sub = solve_subproblem(
            demand=demand_matrix[s],
            cap=cap,
            edge_arrays=edge_arrays,
            facility_proc_cost=proc_cost,
            facility_proc_emission=proc_emission,
            city_landfill_cost=city_landfill_cost,
            cfg=cfg,
            prebuilt_mats=mats,
        )
        if not bool(sub["success"]):
            return {
                "success": False,
                "message": str(sub["message"]),
            }

        p = float(scenario_probs[s])
        recourse[s] = float(sub["objective"])

        pi = np.asarray(sub["pi"], dtype=float)
        mu = np.asarray(sub["mu"], dtype=float)
        a = float(np.dot(pi, demand_matrix[s]))
        b = mu.astype(float)

        cut_a[s] = a
        cut_b[s, :] = b
        theta_rhs[s] = a + float(np.dot(b, cap))

        flow_acc += p * np.asarray(sub["y"], dtype=float)
        shortage_acc += p * np.asarray(sub["z"], dtype=float)

        transport_cost += p * float(sub["transport_cost"])
        processing_cost += p * float(sub["processing_cost"])
        shortage_cost += p * float(sub["shortage_cost"])
        subsidy_benefit += p * float(sub["subsidy_benefit"])
        carbon_cost += p * float(sub["carbon_cost"])
        total_emissions += p * float(sub["total_emission_tco2"])

    return {
        "success": True,
        "recourse": recourse,
        "expected_recourse": float(np.dot(scenario_probs, recourse)),
        "cut_a": cut_a,
        "cut_b": cut_b,
        "theta_rhs": theta_rhs,
        "expected_flow": flow_acc,
        "expected_shortage": shortage_acc,
        "expected_transport_cost_yuan": transport_cost,
        "expected_processing_cost_yuan": processing_cost,
        "expected_shortage_cost_yuan": shortage_cost,
        "expected_subsidy_benefit_yuan": subsidy_benefit,
        "expected_carbon_cost_yuan": carbon_cost,
        "expected_total_emission_tco2": total_emissions,
    }


def _initial_heuristic(
    expected_demand: np.ndarray,
    fixed: np.ndarray,
    capex: np.ndarray,
    cap_min: np.ndarray,
    cap_max: np.ndarray,
    proc_cost: np.ndarray,
    edge_arrays: Dict[str, np.ndarray],
    cfg: AcceleratedBendersConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    n_fac = cap_max.size
    target = cfg.coverage_ratio * float(expected_demand.sum())

    fac_mean_transport = np.zeros(n_fac, dtype=float)
    for j in range(n_fac):
        mask = edge_arrays["fac_idx"] == j
        if np.any(mask):
            fac_mean_transport[j] = float(np.mean(edge_arrays["unit_transport_cost"][mask]))
        else:
            fac_mean_transport[j] = float(np.mean(edge_arrays["unit_transport_cost"]))

    score = fixed / np.maximum(cap_max, 1.0) + 0.15 * capex + proc_cost + fac_mean_transport
    order = np.argsort(score)

    x = np.zeros(n_fac, dtype=float)
    cap = np.zeros(n_fac, dtype=float)

    cum_cap = 0.0
    chosen = []
    for j in order:
        chosen.append(j)
        x[j] = 1.0
        cap[j] = cap_max[j]
        cum_cap += cap_max[j]
        if cum_cap >= target and len(chosen) >= cfg.min_open_facilities:
            break

    # Guarantee min_open and max_open bounds.
    if len(chosen) < cfg.min_open_facilities:
        for j in order:
            if j in chosen:
                continue
            chosen.append(j)
            x[j] = 1.0
            cap[j] = cap_max[j]
            if len(chosen) >= cfg.min_open_facilities:
                break

    if len(chosen) > cfg.max_open_facilities:
        chosen = chosen[: cfg.max_open_facilities]
        x[:] = 0.0
        cap[:] = 0.0
        for j in chosen:
            x[j] = 1.0
            cap[j] = cap_max[j]

    # Refit capacity toward target, keeping feasible lower bounds.
    active = np.where(x > 0.5)[0]
    cap[:] = 0.0
    remain = target
    for j in active:
        alloc = min(cap_max[j], max(cap_min[j], remain / max(1, len(active))))
        cap[j] = alloc
        remain -= alloc

    if cap.sum() < target:
        for j in active:
            extra = min(cap_max[j] - cap[j], target - cap.sum())
            if extra > 0:
                cap[j] += extra
            if cap.sum() >= target:
                break

    cap = np.maximum(cap, cap_min * x)
    cap = np.minimum(cap, cap_max * x)

    return x, cap


def _add_cut(
    cut_pool: Dict[int, List[Dict[str, object]]],
    scenario: int,
    a: float,
    b: np.ndarray,
    violation: float,
    it: int,
    max_cuts: int,
) -> bool:
    if not np.isfinite(a) or not np.all(np.isfinite(b)):
        return False

    existing = cut_pool[scenario]
    for c in existing:
        if abs(float(c["a"]) - a) <= 1e-6 and np.allclose(np.asarray(c["b"], dtype=float), b, rtol=1e-5, atol=1e-8):
            c["last_violation"] = max(float(c["last_violation"]), float(violation))
            c["iter"] = it
            return False

    existing.append(
        {
            "a": float(a),
            "b": b.astype(float).copy(),
            "last_violation": float(violation),
            "iter": int(it),
        }
    )

    if len(existing) > max_cuts:
        existing.sort(key=lambda x: (float(x["last_violation"]), int(x["iter"])), reverse=True)
        del existing[max_cuts:]

    return True


def _build_master_problem(
    fixed: np.ndarray,
    capex: np.ndarray,
    cap_min: np.ndarray,
    cap_max: np.ndarray,
    scenario_probs: np.ndarray,
    target_capacity: float,
    cfg: AcceleratedBendersConfig,
    cut_pool: Dict[int, List[Dict[str, object]]],
    trust_center: np.ndarray | None,
    trust_radius: float,
):
    n_fac = cap_max.size
    n_scen = scenario_probs.size
    n_var = 2 * n_fac + n_scen
    theta_offset = 2 * n_fac

    c_obj = np.concatenate([fixed, capex, scenario_probs])
    integrality = np.concatenate([np.ones(n_fac, dtype=int), np.zeros(n_fac + n_scen, dtype=int)])

    lb = np.concatenate([np.zeros(n_fac), np.zeros(n_fac), np.zeros(n_scen)])
    ub = np.concatenate([np.ones(n_fac), cap_max, np.full(n_scen, np.inf)])
    bounds = Bounds(lb=lb, ub=ub)

    constraints: List[LinearConstraint] = []

    # cap <= cap_max * x
    mat1 = np.zeros((n_fac, n_var), dtype=float)
    for j in range(n_fac):
        mat1[j, j] = -cap_max[j]
        mat1[j, n_fac + j] = 1.0
    constraints.append(LinearConstraint(mat1, -np.inf, 0.0))

    # cap >= cap_min * x
    mat2 = np.zeros((n_fac, n_var), dtype=float)
    for j in range(n_fac):
        mat2[j, j] = cap_min[j]
        mat2[j, n_fac + j] = -1.0
    constraints.append(LinearConstraint(mat2, -np.inf, 0.0))

    # Coverage
    mat3 = np.zeros((1, n_var), dtype=float)
    mat3[0, n_fac : 2 * n_fac] = 1.0
    constraints.append(LinearConstraint(mat3, target_capacity, np.inf))

    # Number of facilities
    mat4 = np.zeros((1, n_var), dtype=float)
    mat4[0, :n_fac] = 1.0
    constraints.append(LinearConstraint(mat4, cfg.min_open_facilities, cfg.max_open_facilities))

    # Stabilization box around trust center.
    if cfg.stabilization and trust_center is not None:
        tr_up = np.zeros((n_fac, n_var), dtype=float)
        tr_lo = np.zeros((n_fac, n_var), dtype=float)
        upper = trust_center + trust_radius * cap_max
        lower = np.maximum(0.0, trust_center - trust_radius * cap_max)
        for j in range(n_fac):
            tr_up[j, n_fac + j] = 1.0
            tr_lo[j, n_fac + j] = 1.0
        constraints.append(LinearConstraint(tr_up, -np.inf, upper))
        constraints.append(LinearConstraint(tr_lo, lower, np.inf))

    # Benders cuts.
    cut_rows = []
    cut_lb = []
    for s, cuts in cut_pool.items():
        for c in cuts:
            row = np.zeros(n_var, dtype=float)
            row[n_fac : 2 * n_fac] = -np.asarray(c["b"], dtype=float)
            row[theta_offset + s] = 1.0
            cut_rows.append(row)
            cut_lb.append(float(c["a"]))

    if cut_rows:
        mat_cut = np.vstack(cut_rows)
        constraints.append(LinearConstraint(mat_cut, np.array(cut_lb, dtype=float), np.full(len(cut_lb), np.inf)))

    return c_obj, integrality, bounds, constraints


def solve_benders_accelerated(
    city_df: pd.DataFrame,
    facility_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    scenario_probs: np.ndarray,
    cfg: AcceleratedBendersConfig,
) -> AcceleratedSolverResult:
    t0 = time.time()

    city_to_idx, fac_to_idx = build_index(city_df, facility_df)
    edge_arrays = prepare_edge_arrays(edge_df, city_to_idx, fac_to_idx)

    n_city = len(city_df)
    n_fac = len(facility_df)
    n_scen = scenario_probs.size

    scenario_names = sorted(scenario_df["scenario_id"].astype(str).unique())
    if len(scenario_names) != n_scen:
        raise ValueError("Scenario count mismatch between scenario_df and scenario_probs.")
    demand_matrix = np.zeros((n_scen, n_city), dtype=float)
    for s_idx, s_name in enumerate(scenario_names):
        sub = scenario_df[scenario_df["scenario_id"] == s_name][["city_code_final", "demand_ton"]]
        for _, row in sub.iterrows():
            i = city_to_idx.get(int(row["city_code_final"]))
            if i is not None:
                demand_matrix[s_idx, i] = float(row["demand_ton"])

    expected_demand = scenario_probs @ demand_matrix

    cap_max = facility_df["capacity_max_ton"].to_numpy(dtype=float)
    cap_min = facility_df["capacity_min_ton"].to_numpy(dtype=float)
    fixed = facility_df["fixed_cost_yuan"].to_numpy(dtype=float)
    capex = facility_df["capacity_cost_yuan_per_ton"].to_numpy(dtype=float)
    proc_cost = facility_df["processing_cost_yuan_per_t"].to_numpy(dtype=float)
    proc_emission = facility_df["processing_emission_tco2_per_t"].to_numpy(dtype=float)
    city_landfill_cost = city_df["landfill_cost_yuan_per_t"].to_numpy(dtype=float)

    mats = _build_subproblem_matrices(n_city, n_fac, edge_arrays["city_idx"], edge_arrays["fac_idx"])
    target_capacity = float(cfg.coverage_ratio * expected_demand.sum())

    cut_pool: Dict[int, List[Dict[str, object]]] = {s: [] for s in range(n_scen)}
    log_rows: List[Dict[str, float]] = []

    best_ub = np.inf
    best_x = np.zeros(n_fac)
    best_cap = np.zeros(n_fac)
    best_theta_s = np.zeros(n_scen)
    best_flow = np.zeros(len(edge_arrays["city_idx"]))
    best_shortage = np.zeros(n_city)
    best_summary: Dict[str, float] = {}

    trust_center = None
    trust_radius = cfg.trust_region_start
    stall_counter = 0
    total_cuts = 0

    # Warm-start incumbent and initial cuts.
    if cfg.warm_start:
        x0, cap0 = _initial_heuristic(expected_demand, fixed, capex, cap_min, cap_max, proc_cost, edge_arrays, cfg)
        init_eval = _evaluate_solution(
            cap=cap0,
            demand_matrix=demand_matrix,
            scenario_probs=scenario_probs,
            edge_arrays=edge_arrays,
            proc_cost=proc_cost,
            proc_emission=proc_emission,
            city_landfill_cost=city_landfill_cost,
            cfg=cfg,
            mats=mats,
        )
        if bool(init_eval["success"]):
            first_stage = float(np.dot(fixed, x0) + np.dot(capex, cap0))
            ub0 = first_stage + float(init_eval["expected_recourse"])
            best_ub = ub0
            best_x = x0.copy()
            best_cap = cap0.copy()
            best_theta_s = np.asarray(init_eval["theta_rhs"], dtype=float)
            best_flow = np.asarray(init_eval["expected_flow"], dtype=float)
            best_shortage = np.asarray(init_eval["expected_shortage"], dtype=float)

            best_summary = {
                "first_stage_cost_yuan": first_stage,
                "expected_recourse_cost_yuan": float(init_eval["expected_recourse"]),
                "expected_transport_cost_yuan": float(init_eval["expected_transport_cost_yuan"]),
                "expected_processing_cost_yuan": float(init_eval["expected_processing_cost_yuan"]),
                "expected_shortage_cost_yuan": float(init_eval["expected_shortage_cost_yuan"]),
                "expected_subsidy_benefit_yuan": float(init_eval["expected_subsidy_benefit_yuan"]),
                "expected_carbon_cost_yuan": float(init_eval["expected_carbon_cost_yuan"]),
                "expected_total_emission_tco2": float(init_eval["expected_total_emission_tco2"]),
                "expected_shortage_ton": float(np.sum(best_shortage)),
                "open_facility_count": float(np.sum(best_x > 0.5)),
                "installed_capacity_ton": float(np.sum(best_cap)),
            }
            trust_center = cap0.copy()

            a0 = np.asarray(init_eval["cut_a"], dtype=float)
            b0 = np.asarray(init_eval["cut_b"], dtype=float)
            for s in range(n_scen):
                added = _add_cut(cut_pool, s, float(a0[s]), b0[s], 1.0, 0, cfg.max_cuts_per_scenario)
                if added:
                    total_cuts += 1

    for it in range(1, cfg.max_iterations + 1):
        tic = time.time()

        c_obj, integrality, bounds, constraints = _build_master_problem(
            fixed=fixed,
            capex=capex,
            cap_min=cap_min,
            cap_max=cap_max,
            scenario_probs=scenario_probs,
            target_capacity=target_capacity,
            cfg=cfg,
            cut_pool=cut_pool,
            trust_center=trust_center,
            trust_radius=trust_radius,
        )

        master = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)
        if not master.success and cfg.stabilization and trust_center is not None:
            # Fallback: if trust-region box accidentally excludes feasible transitions,
            # re-solve this iteration without stabilization constraints.
            c_obj2, integrality2, bounds2, constraints2 = _build_master_problem(
                fixed=fixed,
                capex=capex,
                cap_min=cap_min,
                cap_max=cap_max,
                scenario_probs=scenario_probs,
                target_capacity=target_capacity,
                cfg=cfg,
                cut_pool=cut_pool,
                trust_center=None,
                trust_radius=trust_radius,
            )
            master = milp(c=c_obj2, integrality=integrality2, bounds=bounds2, constraints=constraints2)
            if master.success:
                trust_center = None
                trust_radius = min(cfg.trust_region_max, trust_radius * cfg.trust_region_expand)

        if not master.success:
            if log_rows and float(log_rows[-1]["gap"]) <= cfg.tolerance:
                return AcceleratedSolverResult(
                    success=True,
                    message=f"Benders accelerated early-stop after feasible gap threshold; last master failure at iter {it}: {master.message}",
                    iterations=len(log_rows),
                    lower_bound=float(log_rows[-1]["lower_bound"]),
                    upper_bound=float(best_ub),
                    best_x=best_x,
                    best_cap=best_cap,
                    best_theta_s=best_theta_s,
                    solver_log=pd.DataFrame(log_rows),
                    expected_flow=best_flow,
                    expected_shortage=best_shortage,
                    summary_metrics={
                        **best_summary,
                        "total_runtime_sec": float(time.time() - t0),
                        "total_cuts": float(total_cuts),
                    },
                )
            return AcceleratedSolverResult(
                success=False,
                message=f"Master infeasible at iter {it}: {master.message}",
                iterations=it,
                lower_bound=np.nan,
                upper_bound=np.nan,
                best_x=best_x,
                best_cap=best_cap,
                best_theta_s=best_theta_s,
                solver_log=pd.DataFrame(log_rows),
                expected_flow=best_flow,
                expected_shortage=best_shortage,
                summary_metrics=best_summary,
            )

        x_sol = np.clip(master.x[:n_fac], 0.0, 1.0)
        cap_sol = np.maximum(master.x[n_fac : 2 * n_fac], 0.0)
        theta_sol = np.maximum(master.x[2 * n_fac :], 0.0)

        candidate_master_objective = float(master.fun)
        first_stage = float(np.dot(fixed, x_sol) + np.dot(capex, cap_sol))

        eva = _evaluate_solution(
            cap=cap_sol,
            demand_matrix=demand_matrix,
            scenario_probs=scenario_probs,
            edge_arrays=edge_arrays,
            proc_cost=proc_cost,
            proc_emission=proc_emission,
            city_landfill_cost=city_landfill_cost,
            cfg=cfg,
            mats=mats,
        )
        if not bool(eva["success"]):
            return AcceleratedSolverResult(
                success=False,
                message=f"Subproblem failure at iter {it}: {eva['message']}",
                iterations=it,
                lower_bound=candidate_master_objective,
                upper_bound=np.nan,
                best_x=best_x,
                best_cap=best_cap,
                best_theta_s=best_theta_s,
                solver_log=pd.DataFrame(log_rows),
                expected_flow=best_flow,
                expected_shortage=best_shortage,
                summary_metrics=best_summary,
            )

        ub_now = first_stage + float(eva["expected_recourse"])

        improved = False
        if ub_now < best_ub:
            best_ub = ub_now
            best_x = x_sol.copy()
            best_cap = cap_sol.copy()
            best_theta_s = theta_sol.copy()
            best_flow = np.asarray(eva["expected_flow"], dtype=float)
            best_shortage = np.asarray(eva["expected_shortage"], dtype=float)
            best_summary = {
                "first_stage_cost_yuan": first_stage,
                "expected_recourse_cost_yuan": float(eva["expected_recourse"]),
                "expected_transport_cost_yuan": float(eva["expected_transport_cost_yuan"]),
                "expected_processing_cost_yuan": float(eva["expected_processing_cost_yuan"]),
                "expected_shortage_cost_yuan": float(eva["expected_shortage_cost_yuan"]),
                "expected_subsidy_benefit_yuan": float(eva["expected_subsidy_benefit_yuan"]),
                "expected_carbon_cost_yuan": float(eva["expected_carbon_cost_yuan"]),
                "expected_total_emission_tco2": float(eva["expected_total_emission_tco2"]),
                "expected_shortage_ton": float(np.sum(best_shortage)),
                "open_facility_count": float(np.sum(best_x > 0.5)),
                "installed_capacity_ton": float(np.sum(best_cap)),
            }
            improved = True

        # Manage cuts.
        a_arr = np.asarray(eva["cut_a"], dtype=float)
        b_arr = np.asarray(eva["cut_b"], dtype=float)
        rhs = np.asarray(eva["theta_rhs"], dtype=float)

        new_cuts = 0
        for s in range(n_scen):
            violation = float(rhs[s] - theta_sol[s])
            if violation > cfg.cut_violation_tol:
                added = _add_cut(
                    cut_pool=cut_pool,
                    scenario=s,
                    a=float(a_arr[s]),
                    b=b_arr[s],
                    violation=violation,
                    it=it,
                    max_cuts=cfg.max_cuts_per_scenario,
                )
                if added:
                    new_cuts += 1
                    total_cuts += 1

        # Stabilization update.
        if cfg.stabilization:
            if improved:
                trust_center = best_cap.copy()
                trust_radius = max(cfg.trust_region_start, trust_radius * 0.90)
                stall_counter = 0
            else:
                stall_counter += 1
                if stall_counter >= cfg.stall_iter_expand:
                    trust_radius = min(cfg.trust_region_max, trust_radius * cfg.trust_region_expand)
                    stall_counter = 0

        # A stabilized master is a restriction of the original master and its
        # objective is not a global lower bound. Re-solve the current master
        # without the trust-region box after adding cuts to obtain a valid bound.
        global_c, global_integrality, global_bounds, global_constraints = _build_master_problem(
            fixed=fixed,
            capex=capex,
            cap_min=cap_min,
            cap_max=cap_max,
            scenario_probs=scenario_probs,
            target_capacity=target_capacity,
            cfg=cfg,
            cut_pool=cut_pool,
            trust_center=None,
            trust_radius=trust_radius,
        )
        global_master = milp(
            c=global_c,
            integrality=global_integrality,
            bounds=global_bounds,
            constraints=global_constraints,
        )
        if not global_master.success:
            return AcceleratedSolverResult(
                success=False,
                message=f"Global bound master failed at iter {it}: {global_master.message}",
                iterations=it,
                lower_bound=np.nan,
                upper_bound=float(best_ub),
                best_x=best_x,
                best_cap=best_cap,
                best_theta_s=best_theta_s,
                solver_log=pd.DataFrame(log_rows),
                expected_flow=best_flow,
                expected_shortage=best_shortage,
                summary_metrics=best_summary,
            )
        global_lb = float(global_master.fun)
        raw_gap = (best_ub - global_lb) / (abs(best_ub) + 1e-9)
        bound_consistent = global_lb <= best_ub + 1e-8 * max(abs(best_ub), 1.0)
        gap = max(0.0, raw_gap) if bound_consistent else np.inf

        log_rows.append(
            {
                "iteration": it,
                "lower_bound": global_lb,
                "candidate_master_objective": candidate_master_objective,
                "upper_bound": best_ub,
                "gap": gap,
                "raw_gap": raw_gap,
                "bound_consistent": bool(bound_consistent),
                "new_cuts": float(new_cuts),
                "total_cuts": float(total_cuts),
                "trust_radius": float(trust_radius),
                "solve_time_sec": float(time.time() - tic),
                "open_facility_count": float(np.sum(x_sol > 0.5)),
                "installed_capacity_ton": float(np.sum(cap_sol)),
            }
        )

        if bound_consistent and gap <= cfg.tolerance:
            break

    status = "Benders accelerated converged" if log_rows and log_rows[-1]["gap"] <= cfg.tolerance else "Benders accelerated max-iter"

    return AcceleratedSolverResult(
        success=True,
        message=status,
        iterations=len(log_rows),
        lower_bound=float(log_rows[-1]["lower_bound"]) if log_rows else np.nan,
        upper_bound=float(best_ub),
        best_x=best_x,
        best_cap=best_cap,
        best_theta_s=best_theta_s,
        solver_log=pd.DataFrame(log_rows),
        expected_flow=best_flow,
        expected_shortage=best_shortage,
        summary_metrics={
            **best_summary,
            "total_runtime_sec": float(time.time() - t0),
            "total_cuts": float(total_cuts),
        },
    )
