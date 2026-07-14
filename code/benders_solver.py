#!/usr/bin/env python3
"""Two-stage stochastic facility-location solver using Benders decomposition.

The implementation supports the CDW recycling network model:
- First stage: open facilities + set capacities.
- Second stage: scenario-based transport allocation + shortage handling.
- Algorithm: single-cut Benders decomposition with LP dual cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


@dataclass
class BendersConfig:
    max_iterations: int = 20
    tolerance: float = 1e-3
    coverage_ratio: float = 0.90
    min_open_facilities: int = 12
    max_open_facilities: int = 32
    shortage_penalty: float = 450.0
    landfill_penalty: float = 0.0
    subsidy_per_ton: float = 0.0
    carbon_price_per_tco2: float = 0.0
    shortage_emission_tco2_per_t: float = 0.35


@dataclass
class SolverResult:
    success: bool
    message: str
    iterations: int
    lower_bound: float
    upper_bound: float
    best_x: np.ndarray
    best_cap: np.ndarray
    best_theta: float
    solver_log: pd.DataFrame
    expected_flow: np.ndarray
    expected_shortage: np.ndarray
    summary_metrics: Dict[str, float]


def build_index(city_df: pd.DataFrame, facility_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    city_ids = city_df["city_code_final"].astype(int).tolist()
    fac_ids = facility_df["site_id"].astype(int).tolist()
    city_to_idx = {cid: idx for idx, cid in enumerate(city_ids)}
    fac_to_idx = {fid: idx for idx, fid in enumerate(fac_ids)}
    return city_to_idx, fac_to_idx


def prepare_edge_arrays(
    edge_df: pd.DataFrame,
    city_to_idx: Dict[int, int],
    fac_to_idx: Dict[int, int],
) -> Dict[str, np.ndarray]:
    arr = edge_df[["city_code", "site_id", "unit_transport_cost", "transport_emission_tco2_per_t"]].copy()
    arr["city_idx"] = arr["city_code"].astype(int).map(city_to_idx)
    arr["fac_idx"] = arr["site_id"].astype(int).map(fac_to_idx)
    arr = arr.dropna(subset=["city_idx", "fac_idx"])
    arr["city_idx"] = arr["city_idx"].astype(int)
    arr["fac_idx"] = arr["fac_idx"].astype(int)

    return {
        "city_idx": arr["city_idx"].to_numpy(dtype=int),
        "fac_idx": arr["fac_idx"].to_numpy(dtype=int),
        "unit_transport_cost": arr["unit_transport_cost"].to_numpy(dtype=float),
        "transport_emission": arr["transport_emission_tco2_per_t"].to_numpy(dtype=float),
    }


def _build_subproblem_matrices(
    n_city: int,
    n_fac: int,
    edge_city_idx: np.ndarray,
    edge_fac_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_edge = len(edge_city_idx)
    n_var = n_edge + n_city

    a_eq = np.zeros((n_city, n_var), dtype=float)
    for e in range(n_edge):
        a_eq[edge_city_idx[e], e] = 1.0
    for i in range(n_city):
        a_eq[i, n_edge + i] = 1.0

    a_ub = np.zeros((n_fac, n_var), dtype=float)
    for e in range(n_edge):
        a_ub[edge_fac_idx[e], e] = 1.0

    return a_eq, a_ub


def solve_subproblem(
    demand: np.ndarray,
    cap: np.ndarray,
    edge_arrays: Dict[str, np.ndarray],
    facility_proc_cost: np.ndarray,
    facility_proc_emission: np.ndarray,
    city_landfill_cost: np.ndarray,
    cfg: BendersConfig,
    prebuilt_mats: Tuple[np.ndarray, np.ndarray] | None = None,
) -> Dict[str, np.ndarray | float | bool | str]:
    edge_city = edge_arrays["city_idx"]
    edge_fac = edge_arrays["fac_idx"]
    edge_tcost = edge_arrays["unit_transport_cost"]
    edge_temission = edge_arrays["transport_emission"]

    n_city = demand.size
    n_fac = cap.size
    n_edge = edge_city.size

    if prebuilt_mats is None:
        a_eq, a_ub = _build_subproblem_matrices(n_city, n_fac, edge_city, edge_fac)
    else:
        a_eq, a_ub = prebuilt_mats

    b_eq = demand.astype(float)
    b_ub = cap.astype(float)

    edge_unit_cost = (
        edge_tcost
        + facility_proc_cost[edge_fac]
        - cfg.subsidy_per_ton
        + cfg.carbon_price_per_tco2 * (edge_temission + facility_proc_emission[edge_fac])
    )
    edge_unit_cost = np.maximum(edge_unit_cost, 0.0)

    shortage_cost = (
        cfg.shortage_penalty
        + cfg.landfill_penalty
        + city_landfill_cost
        + cfg.carbon_price_per_tco2 * cfg.shortage_emission_tco2_per_t
    )

    c = np.concatenate([edge_unit_cost, shortage_cost])
    bounds = [(0.0, None)] * (n_edge + n_city)

    lp = linprog(
        c=c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not lp.success:
        return {
            "success": False,
            "message": lp.message,
            "objective": np.inf,
        }

    x = lp.x
    y = x[:n_edge]
    z = x[n_edge:]

    pi = np.asarray(lp.eqlin.marginals, dtype=float)
    mu = np.asarray(lp.ineqlin.marginals, dtype=float)

    transport_cost = float(np.dot(y, edge_tcost))
    processing_cost = float(np.dot(y, facility_proc_cost[edge_fac]))
    subsidy_benefit = float(np.sum(y) * cfg.subsidy_per_ton)
    shortage_cost_total = float(np.dot(z, cfg.shortage_penalty + cfg.landfill_penalty + city_landfill_cost))

    total_emission = float(
        np.dot(y, edge_temission + facility_proc_emission[edge_fac])
        + np.sum(z) * cfg.shortage_emission_tco2_per_t
    )
    carbon_cost = float(total_emission * cfg.carbon_price_per_tco2)

    return {
        "success": True,
        "message": "optimal",
        "objective": float(lp.fun),
        "y": y,
        "z": z,
        "pi": pi,
        "mu": mu,
        "transport_cost": transport_cost,
        "processing_cost": processing_cost,
        "subsidy_benefit": subsidy_benefit,
        "shortage_cost": shortage_cost_total,
        "carbon_cost": carbon_cost,
        "total_emission_tco2": total_emission,
    }


def solve_benders(
    city_df: pd.DataFrame,
    facility_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    scenario_probs: np.ndarray,
    cfg: BendersConfig,
) -> SolverResult:
    city_to_idx, fac_to_idx = build_index(city_df, facility_df)
    edge_arrays = prepare_edge_arrays(edge_df, city_to_idx, fac_to_idx)

    n_city = len(city_df)
    n_fac = len(facility_df)
    n_scen = scenario_probs.size

    edge_city = edge_arrays["city_idx"]
    edge_fac = edge_arrays["fac_idx"]
    n_edge = len(edge_city)

    scenario_names = [f"S{i+1:03d}" for i in range(n_scen)]
    demand_matrix = np.zeros((n_scen, n_city), dtype=float)

    for s_idx, s_name in enumerate(scenario_names):
        sub = scenario_df[scenario_df["scenario_id"] == s_name][["city_code_final", "demand_ton"]]
        for _, row in sub.iterrows():
            cidx = city_to_idx.get(int(row["city_code_final"]))
            if cidx is not None:
                demand_matrix[s_idx, cidx] = float(row["demand_ton"])

    expected_demand = scenario_probs @ demand_matrix

    cap_max = facility_df["capacity_max_ton"].to_numpy(dtype=float)
    cap_min = facility_df["capacity_min_ton"].to_numpy(dtype=float)
    fixed_cost = facility_df["fixed_cost_yuan"].to_numpy(dtype=float)
    capex_cost = facility_df["capacity_cost_yuan_per_ton"].to_numpy(dtype=float)
    proc_cost = facility_df["processing_cost_yuan_per_t"].to_numpy(dtype=float)
    proc_emission = facility_df["processing_emission_tco2_per_t"].to_numpy(dtype=float)
    city_landfill_cost = city_df["landfill_cost_yuan_per_t"].to_numpy(dtype=float)

    mats = _build_subproblem_matrices(n_city, n_fac, edge_city, edge_fac)

    benders_cuts: List[Tuple[float, np.ndarray]] = []
    log_rows = []

    best_ub = np.inf
    best_x = np.zeros(n_fac)
    best_cap = np.zeros(n_fac)
    best_theta = 0.0
    best_flow = np.zeros(n_edge)
    best_shortage = np.zeros(n_city)
    best_summary: Dict[str, float] = {}

    target_capacity = float(cfg.coverage_ratio * expected_demand.sum())

    for it in range(1, cfg.max_iterations + 1):
        n_var = 2 * n_fac + 1
        theta_idx = 2 * n_fac

        c_obj = np.concatenate([fixed_cost, capex_cost, np.array([1.0])])
        integrality = np.concatenate([np.ones(n_fac, dtype=int), np.zeros(n_fac + 1, dtype=int)])

        lb = np.concatenate([np.zeros(n_fac), np.zeros(n_fac), np.array([0.0])])
        ub = np.concatenate([np.ones(n_fac), cap_max, np.array([np.inf])])
        bounds = Bounds(lb=lb, ub=ub)

        constraints: List[LinearConstraint] = []

        mat1 = np.zeros((n_fac, n_var), dtype=float)
        for j in range(n_fac):
            mat1[j, j] = -cap_max[j]
            mat1[j, n_fac + j] = 1.0
        constraints.append(LinearConstraint(mat1, -np.inf, 0.0))

        mat2 = np.zeros((n_fac, n_var), dtype=float)
        for j in range(n_fac):
            mat2[j, j] = cap_min[j]
            mat2[j, n_fac + j] = -1.0
        constraints.append(LinearConstraint(mat2, -np.inf, 0.0))

        mat3 = np.zeros((1, n_var), dtype=float)
        mat3[0, n_fac : 2 * n_fac] = 1.0
        constraints.append(LinearConstraint(mat3, target_capacity, np.inf))

        mat4 = np.zeros((1, n_var), dtype=float)
        mat4[0, :n_fac] = 1.0
        constraints.append(LinearConstraint(mat4, cfg.min_open_facilities, cfg.max_open_facilities))

        for a_cut, b_cut in benders_cuts:
            cut_mat = np.zeros((1, n_var), dtype=float)
            cut_mat[0, n_fac : 2 * n_fac] = -b_cut
            cut_mat[0, theta_idx] = 1.0
            constraints.append(LinearConstraint(cut_mat, a_cut, np.inf))

        master = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

        if not master.success:
            return SolverResult(
                success=False,
                message=f"Master problem failed at iteration {it}: {master.message}",
                iterations=it,
                lower_bound=np.nan,
                upper_bound=np.nan,
                best_x=best_x,
                best_cap=best_cap,
                best_theta=best_theta,
                solver_log=pd.DataFrame(log_rows),
                expected_flow=best_flow,
                expected_shortage=best_shortage,
                summary_metrics=best_summary,
            )

        x_sol = np.clip(master.x[:n_fac], 0.0, 1.0)
        # Guard against tiny negative values from numerical tolerance.
        cap_sol = np.maximum(master.x[n_fac : 2 * n_fac], 0.0)
        theta_sol = float(master.x[theta_idx])
        lb_now = float(master.fun)

        first_stage_cost = float(np.dot(fixed_cost, x_sol) + np.dot(capex_cost, cap_sol))

        recourse_values = []
        weighted_const = 0.0
        weighted_cap_coef = np.zeros(n_fac, dtype=float)

        scenario_transport = 0.0
        scenario_processing = 0.0
        scenario_shortage = 0.0
        scenario_subsidy = 0.0
        scenario_carbon = 0.0
        scenario_emissions = 0.0

        flow_acc = np.zeros(n_edge, dtype=float)
        shortage_acc = np.zeros(n_city, dtype=float)

        for s_idx in range(n_scen):
            sub = solve_subproblem(
                demand=demand_matrix[s_idx],
                cap=cap_sol,
                edge_arrays=edge_arrays,
                facility_proc_cost=proc_cost,
                facility_proc_emission=proc_emission,
                city_landfill_cost=city_landfill_cost,
                cfg=cfg,
                prebuilt_mats=mats,
            )
            if not sub["success"]:
                return SolverResult(
                    success=False,
                    message=f"Subproblem failed at iteration {it}, scenario {s_idx}: {sub['message']}",
                    iterations=it,
                    lower_bound=lb_now,
                    upper_bound=np.nan,
                    best_x=best_x,
                    best_cap=best_cap,
                    best_theta=best_theta,
                    solver_log=pd.DataFrame(log_rows),
                    expected_flow=best_flow,
                    expected_shortage=best_shortage,
                    summary_metrics=best_summary,
                )

            p_s = float(scenario_probs[s_idx])
            recourse_values.append(float(sub["objective"]))

            pi = np.asarray(sub["pi"], dtype=float)
            mu = np.asarray(sub["mu"], dtype=float)

            weighted_const += p_s * float(np.dot(pi, demand_matrix[s_idx]))
            weighted_cap_coef += p_s * mu

            flow_acc += p_s * np.asarray(sub["y"], dtype=float)
            shortage_acc += p_s * np.asarray(sub["z"], dtype=float)

            scenario_transport += p_s * float(sub["transport_cost"])
            scenario_processing += p_s * float(sub["processing_cost"])
            scenario_shortage += p_s * float(sub["shortage_cost"])
            scenario_subsidy += p_s * float(sub["subsidy_benefit"])
            scenario_carbon += p_s * float(sub["carbon_cost"])
            scenario_emissions += p_s * float(sub["total_emission_tco2"])

        expected_recourse = float(np.dot(scenario_probs, np.array(recourse_values)))
        ub_now = first_stage_cost + expected_recourse

        if ub_now < best_ub:
            best_ub = ub_now
            best_x = x_sol.copy()
            best_cap = cap_sol.copy()
            best_theta = theta_sol
            best_flow = flow_acc.copy()
            best_shortage = shortage_acc.copy()
            best_summary = {
                "first_stage_cost_yuan": first_stage_cost,
                "expected_recourse_cost_yuan": expected_recourse,
                "expected_transport_cost_yuan": scenario_transport,
                "expected_processing_cost_yuan": scenario_processing,
                "expected_shortage_cost_yuan": scenario_shortage,
                "expected_subsidy_benefit_yuan": scenario_subsidy,
                "expected_carbon_cost_yuan": scenario_carbon,
                "expected_total_emission_tco2": scenario_emissions,
                "expected_shortage_ton": float(np.sum(shortage_acc)),
                "open_facility_count": float(np.sum(best_x > 0.5)),
                "installed_capacity_ton": float(np.sum(best_cap)),
            }

        raw_gap = (best_ub - lb_now) / (abs(best_ub) + 1e-9)
        bound_consistent = lb_now <= best_ub + 1e-8 * max(abs(best_ub), 1.0)
        gap = max(0.0, raw_gap) if bound_consistent else np.inf

        log_rows.append(
            {
                "iteration": it,
                "lower_bound": lb_now,
                "upper_bound": best_ub,
                "gap": gap,
                "raw_gap": raw_gap,
                "bound_consistent": bool(bound_consistent),
                "theta_master": theta_sol,
                "expected_recourse": expected_recourse,
                "open_facility_count": float(np.sum(x_sol > 0.5)),
                "installed_capacity_ton": float(np.sum(cap_sol)),
            }
        )

        benders_cuts.append((weighted_const, weighted_cap_coef.copy()))

        if bound_consistent and gap <= cfg.tolerance:
            break

    return SolverResult(
        success=True,
        message="Benders converged" if len(log_rows) < cfg.max_iterations else "Benders reached max iterations",
        iterations=len(log_rows),
        lower_bound=float(log_rows[-1]["lower_bound"]) if log_rows else np.nan,
        upper_bound=float(best_ub),
        best_x=best_x,
        best_cap=best_cap,
        best_theta=best_theta,
        solver_log=pd.DataFrame(log_rows),
        expected_flow=best_flow,
        expected_shortage=best_shortage,
        summary_metrics=best_summary,
    )


def flow_vector_to_matrix(
    flow_vector: np.ndarray,
    edge_df: pd.DataFrame,
    city_df: pd.DataFrame,
    facility_df: pd.DataFrame,
) -> np.ndarray:
    city_ids = city_df["city_code_final"].astype(int).tolist()
    fac_ids = facility_df["site_id"].astype(int).tolist()
    city_to_idx = {cid: i for i, cid in enumerate(city_ids)}
    fac_to_idx = {fid: j for j, fid in enumerate(fac_ids)}

    mat = np.zeros((len(city_ids), len(fac_ids)), dtype=float)
    for val, (_, row) in zip(flow_vector, edge_df.iterrows()):
        i = city_to_idx[int(row["city_code"])]
        j = fac_to_idx[int(row["site_id"])]
        mat[i, j] += float(val)
    return mat


def summarize_open_facilities(result: SolverResult, facility_df: pd.DataFrame) -> pd.DataFrame:
    out = facility_df.copy()
    out["x_opt"] = result.best_x
    out["capacity_opt_ton"] = result.best_cap
    out["is_open"] = (out["x_opt"] > 0.5).astype(int)
    return out.sort_values(["is_open", "capacity_opt_ton"], ascending=[False, False])
