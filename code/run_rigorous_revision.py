#!/usr/bin/env python3
"""Build and solve the frozen evidence set for the independent CDW study."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from benders_solver import BendersConfig, _build_subproblem_matrices, build_index, prepare_edge_arrays, solve_subproblem
from benders_solver_advanced import AcceleratedBendersConfig, _evaluate_solution
from run_cplex_monolithic_benchmark import (
    CPLEX_BIN,
    MonolithicConfig,
    _parse_cplex_log,
    _prepare_instance,
    _write_expr_terms,
    _write_lp,
)
from release_layout import (
    DATA_DIR,
    OUT_DIR,
    SCRIPT_DIR,
    SOURCE_CODE_NAMES,
    SOURCE_PATHS,
    VERSION,
    public_path_for,
)
from scenario_generator import FAMILY_WEIGHTS, GENERATOR_VERSION, demand_matrix, generate_scenarios


LOG_DIR = OUT_DIR / "logs"
SOLUTION_DIR = OUT_DIR / "solutions"
TMP_DIR = OUT_DIR / "tmp_lp"

PLANNING_YEARS = tuple(range(2025, 2036))
REPORTING_YEARS = (2025, 2027, 2030, 2035)
BASE_YEAR = PLANNING_YEARS[0]
ANNUAL_DISCOUNT_RATE = 0.05
PRIMARY_SCENARIO_SEED = 20_260_713
ROAD_EMISSION_KG_PER_TKM = 0.12
SHORTAGE_EMISSION_TCO2_PER_T = 0.35


def discount_factor(year: int) -> float:
    """Present-value factor for a constant-price cash flow in a calendar year."""
    return (1.0 + ANNUAL_DISCOUNT_RATE) ** -(int(year) - BASE_YEAR)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    group: str
    carbon_price: float = 0.0
    landfill_penalty: float = 0.0
    coverage_ratio: float = 0.90
    max_open: int = 114
    fixed_cost_scale: float = 1.0
    capex_scale: float = 1.0
    operating_cost_scale: float = 1.0
    shortage_emission: float = SHORTAGE_EMISSION_TCO2_PER_T
    base_shortage_penalty: float = 460.0
    demand_scale: float = 1.0
    parameter_value: float = 0.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _required_city_codes(pred: pd.DataFrame) -> set[int]:
    required = pred[pred["year"].isin(PLANNING_YEARS) & pred["scenario"].isin(FAMILY_WEIGHTS)].copy()
    counts = required.groupby("city_code_final").size()
    return set(counts[counts == len(PLANNING_YEARS) * len(FAMILY_WEIGHTS)].index.astype(int))


def build_revision_inputs() -> Dict[str, object]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    city_all = pd.read_csv(SOURCE_PATHS["city_master"])
    fac = pd.read_csv(SOURCE_PATHS["candidate_sites"])
    pred = pd.read_csv(SOURCE_PATHS["demand_projections"])
    regional = pd.read_csv(SOURCE_PATHS["regional_operating_parameters"])
    transport = pd.read_csv(SOURCE_PATHS["transport_matrix"])

    required_codes = _required_city_codes(pred)
    city = city_all[city_all["city_code_final"].astype(int).isin(required_codes)].copy()
    excluded = city_all[~city_all["city_code_final"].astype(int).isin(required_codes)].copy()
    excluded["exclusion_reason"] = "No complete city-level trajectory for all three demand families in every year from 2025 through 2035; excluded without imputation"
    excluded["data_version"] = VERSION

    regional_by_province = regional.drop_duplicates("province").set_index("province")
    fac["processing_emission_tco2_per_t"] = (
        fac["province"].map(regional_by_province["emission_factor_kgco2_per_t"]).astype(float) / 1000.0
    )
    fac["capacity_parameter_source"] = "deterministic planning proxy generated from macro-regional scale factors and fixed site seeds"
    fac["fixed_capex_parameter_source"] = "deterministic planning proxy generated from macro-regional scale factors and fixed site seeds"
    fac["processing_parameter_source"] = "three-macro-region cost and emissions inventory mapped by province"
    fac["data_version"] = VERSION

    city_codes = set(city["city_code_final"].astype(int))
    site_ids = set(fac["site_id"].astype(int))
    edge = transport[
        transport["city_code"].astype(int).isin(city_codes)
        & transport["site_id"].astype(int).isin(site_ids)
    ].copy()
    province_by_city = city.assign(_code=city["city_code_final"].astype(int)).set_index("_code")["province"]
    transport_rate = regional_by_province["transport_cost_yuan_per_tkm"]
    edge["origin_province"] = edge["city_code"].astype(int).map(province_by_city)
    edge["unit_transport_cost"] = (
        edge["highway_distance_km"].astype(float)
        * edge["origin_province"].map(transport_rate).astype(float)
    )
    edge["transport_emission_tco2_per_t"] = (
        edge["highway_distance_km"].astype(float) * ROAD_EMISSION_KG_PER_TKM / 1000.0
    )
    edge["best_mode"] = "road"
    edge["network_rule"] = "12 lowest road-cost candidate facilities per matched city"
    edge["cost_source"] = "macro-regional transport rate mapped by origin province and multiplied by road distance"
    edge["distance_source"] = "great-circle distance multiplied by road detour factor 1.3"
    edge["data_version"] = VERSION
    edge = edge.sort_values(["city_code", "unit_transport_cost"]).groupby("city_code", as_index=False).head(12)

    scenario, probs = generate_scenarios(
        pred,
        city,
        scenario_count=16,
        seed=PRIMARY_SCENARIO_SEED,
    )
    scenario["data_version"] = VERSION
    scenario["missing_city_treatment"] = "complete-case matched spatial universe; no demand imputation"
    city["data_version"] = VERSION
    city["spatial_scope"] = "complete-case matched city-level units"

    if len(city) != 333:
        raise RuntimeError(f"Expected 333 complete-case cities, found {len(city)}")
    edge_counts = edge.groupby("city_code").size()
    if len(edge_counts) != len(city) or not (edge_counts == 12).all():
        raise RuntimeError("Road network does not contain exactly 12 arcs per matched city")
    if scenario.groupby("scenario_id")["city_code_final"].nunique().nunique() != 1:
        raise RuntimeError("Scenario city coverage is inconsistent")

    paths = {
        "city": DATA_DIR / "city_list_complete_case.csv",
        "facility": DATA_DIR / "facility_list_transparent_proxy.csv",
        "edge": DATA_DIR / "road_transport_edges.csv",
        "scenario": DATA_DIR / "stochastic_scenarios_2030.csv",
        "probability": DATA_DIR / "scenario_probabilities_2030.npy",
        "excluded": DATA_DIR / "excluded_city_audit.csv",
    }
    city.to_csv(paths["city"], index=False)
    fac.to_csv(paths["facility"], index=False)
    edge.to_csv(paths["edge"], index=False)
    scenario.to_csv(paths["scenario"], index=False)
    np.save(paths["probability"], probs)
    excluded.to_csv(paths["excluded"], index=False)

    return {
        "city": city,
        "facility": fac,
        "edge": edge,
        "scenario": scenario,
        "probs": probs,
        "pred": pred,
        "paths": paths,
    }


def _solution_values(path: Path) -> Tuple[float, Dict[str, float]]:
    root = ET.parse(path).getroot()
    header = root.find(".//header")
    if header is None:
        raise RuntimeError(f"Missing CPLEX solution header: {path}")
    values = {v.attrib["name"]: float(v.attrib["value"]) for v in root.findall(".//variable")}
    return float(header.attrib["objectiveValue"]), values


def _write_facility_mip_start(fac: pd.DataFrame, solution_csv: Path, path: Path) -> None:
    start = pd.read_csv(solution_csv).set_index("site_id")
    root = ET.Element("CPLEXSolution", version="1.2")
    ET.SubElement(root, "header", solutionName="nested_feasible_start")
    variables = ET.SubElement(root, "variables")
    for j, site_id in enumerate(fac["site_id"]):
        row = start.loc[site_id]
        ET.SubElement(variables, "variable", name=f"x_{j}", value=str(int(row["open"])))
        ET.SubElement(
            variables,
            "variable",
            name=f"u_{j}",
            value=f"{float(row['installed_capacity_ton']):.17g}",
        )
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)


def _path_is_ascii(path: Path) -> bool:
    try:
        str(path.resolve()).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _run_cplex(
    lp_path: Path,
    solution_path: Path,
    log_path: Path,
    time_limit: int = 300,
    mip_start_path: Path | None = None,
) -> Dict[str, object]:
    # CPLEX 22.1.1 on Windows cannot read or write files whose paths contain
    # non-ASCII characters even when the paths are valid. When the working
    # directory contains such characters, stage the LP (and any MIP start) in
    # an ASCII temporary directory, run the solver there, and copy the
    # solution back to its canonical path.
    staged = not _path_is_ascii(lp_path.parent)
    if staged:
        stage_dir = Path(tempfile.gettempdir()) / "cdw_cplex_stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        solver_lp = stage_dir / lp_path.name
        shutil.copy(lp_path, solver_lp)
        solver_start = None
        if mip_start_path is not None:
            solver_start = stage_dir / mip_start_path.name
            shutil.copy(mip_start_path, solver_start)
        solver_solution = stage_dir / solution_path.name
    else:
        stage_dir = lp_path.parent
        solver_lp = lp_path
        solver_start = mip_start_path
        solver_solution = solution_path

    cmd = [
        str(CPLEX_BIN),
        "-c",
        f'read "{solver_lp}"',
    ]
    if solver_start is not None:
        cmd.append(f'read "{solver_start}"')
    cmd += [
        f"set timelimit {time_limit}",
        "set threads 16",
        "set mip tolerances mipgap 1e-9",
        "set emphasis numerical y",
        "optimize",
        f'write "{solver_solution}"',
        "quit",
    ]
    t0 = time.time()
    # errors="replace" keeps the run alive when the console codepage of the
    # solver process cannot be decoded (e.g., non-ASCII working-directory
    # characters on Windows); parsing below relies on ASCII status lines only.
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        cwd=stage_dir,
        timeout=time_limit + 30,
    )
    if staged and solver_solution.exists():
        shutil.copy(solver_solution, solution_path)
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(text, encoding="utf-8")
    parsed = _parse_cplex_log(text)
    parsed["runtime_sec"] = time.time() - t0
    parsed["return_code"] = proc.returncode
    return parsed


def solve_single_period(
    bundle: Dict[str, object],
    spec: PolicySpec,
    *,
    time_limit: int = 300,
    mip_start_solution_csv: Path | None = None,
) -> Dict[str, object]:
    city = bundle["city"].copy()
    fac = bundle["facility"].copy()
    edge = bundle["edge"].copy()
    scenario = bundle["scenario"].copy()
    scenario["demand_ton"] = scenario["demand_ton"].astype(float) * spec.demand_scale
    probs = bundle["probs"]
    fac["fixed_cost_yuan"] *= spec.fixed_cost_scale
    fac["capacity_cost_yuan_per_ton"] *= spec.capex_scale
    fac["processing_cost_yuan_per_t"] *= spec.operating_cost_scale
    edge["unit_transport_cost"] *= spec.operating_cost_scale
    city["landfill_cost_yuan_per_t"] *= spec.operating_cost_scale

    cfg = MonolithicConfig(
        coverage_ratio=spec.coverage_ratio,
        min_open_facilities=24,
        max_open_facilities=spec.max_open,
        shortage_penalty=spec.base_shortage_penalty,
        landfill_penalty=spec.landfill_penalty,
        subsidy_per_ton=0.0,
        carbon_price_per_tco2=spec.carbon_price,
        shortage_emission_tco2_per_t=spec.shortage_emission,
        time_limit_sec=time_limit,
        threads=16,
        mip_gap=1e-9,
    )
    instance = _prepare_instance(city, fac, edge, scenario, probs, cfg)
    instance["min_open"] = 24
    instance["max_open"] = spec.max_open
    lp = TMP_DIR / f"single_{spec.name}.lp"
    sol = TMP_DIR / f"single_{spec.name}.sol"
    log = LOG_DIR / f"single_{spec.name}.log"
    _write_lp(instance, lp)
    mip_start = None
    baseline_solution = SOLUTION_DIR / "single_baseline_facility_solution.csv"
    if spec.group == "coverage_sweep" and baseline_solution.exists():
        mip_start = TMP_DIR / f"single_{spec.name}.mst"
        _write_facility_mip_start(fac, baseline_solution, mip_start)
    if mip_start_solution_csv is not None and mip_start_solution_csv.exists():
        mip_start = TMP_DIR / f"single_{spec.name}.mst"
        _write_facility_mip_start(fac, mip_start_solution_csv, mip_start)
    parsed = _run_cplex(lp, sol, log, time_limit=time_limit, mip_start_path=mip_start)
    if not sol.exists():
        return {**asdict(spec), **parsed, "success": False}

    cplex_obj, values = _solution_values(sol)
    x = np.array([values.get(f"x_{j}", 0.0) for j in range(len(fac))])
    cap = np.array([values.get(f"u_{j}", 0.0) for j in range(len(fac))])
    if cap.min() < -1e-5:
        raise RuntimeError(f"CPLEX returned materially negative capacity: {cap.min():.6g}")
    cap = np.maximum(cap, 0.0)
    city_to_idx, fac_to_idx = build_index(city, fac)
    edge_arrays = prepare_edge_arrays(edge, city_to_idx, fac_to_idx)
    dm, _ = demand_matrix(city, scenario, probs)
    mats = _build_subproblem_matrices(len(city), len(fac), edge_arrays["city_idx"], edge_arrays["fac_idx"])
    eval_cfg = AcceleratedBendersConfig(
        coverage_ratio=spec.coverage_ratio,
        min_open_facilities=24,
        max_open_facilities=spec.max_open,
        shortage_penalty=spec.base_shortage_penalty,
        landfill_penalty=spec.landfill_penalty,
        carbon_price_per_tco2=spec.carbon_price,
        shortage_emission_tco2_per_t=spec.shortage_emission,
    )
    eva = _evaluate_solution(
        cap,
        dm,
        probs,
        edge_arrays,
        fac["processing_cost_yuan_per_t"].to_numpy(float),
        fac["processing_emission_tco2_per_t"].to_numpy(float),
        city["landfill_cost_yuan_per_t"].to_numpy(float),
        eval_cfg,
        mats,
    )
    first_stage = float(fac["fixed_cost_yuan"].to_numpy(float) @ x + fac["capacity_cost_yuan_per_ton"].to_numpy(float) @ cap)
    reevaluated = first_stage + float(eva["expected_recourse"])
    solution_table = fac[["site_id", "facility_city", "province", "region"]].copy()
    solution_table["open"] = (x > 0.5).astype(int)
    solution_table["installed_capacity_ton"] = cap
    solution_table["policy"] = spec.name
    solution_table["data_version"] = VERSION
    solution_table.to_csv(SOLUTION_DIR / f"single_{spec.name}_facility_solution.csv", index=False)
    return {
        **asdict(spec),
        "success": parsed["status"] in {"optimal", "time_limit_feasible"},
        "solver_status": parsed["status"],
        "runtime_sec": parsed["runtime_sec"],
        "cplex_objective_yuan": cplex_obj,
        "reevaluated_objective_yuan": reevaluated,
        "objective_crosscheck_relative_error": (reevaluated - cplex_obj) / max(abs(cplex_obj), 1.0),
        "best_bound_yuan": parsed["best_bound"],
        "solver_gap": parsed["gap"],
        "open_facilities": int((x > 0.5).sum()),
        "installed_capacity_ton": float(cap.sum()),
        "first_stage_cost_yuan": first_stage,
        "expected_recourse_cost_yuan": float(eva["expected_recourse"]),
        "expected_transport_cost_yuan": float(eva["expected_transport_cost_yuan"]),
        "expected_processing_cost_yuan": float(eva["expected_processing_cost_yuan"]),
        "expected_shortage_cost_yuan": float(eva["expected_shortage_cost_yuan"]),
        "expected_carbon_cost_yuan": float(eva["expected_carbon_cost_yuan"]),
        "expected_emission_tco2": float(eva["expected_total_emission_tco2"]),
        "expected_shortage_ton": float(np.asarray(eva["expected_shortage"]).sum()),
        "data_version": VERSION,
    }


def single_period_specs() -> List[PolicySpec]:
    specs = [
        PolicySpec("baseline", "primary"),
        PolicySpec("carbon_tax_120", "primary", carbon_price=120.0, parameter_value=120.0),
        PolicySpec("landfill_penalty_260", "primary", landfill_penalty=260.0, parameter_value=260.0),
        PolicySpec("combined_120_260", "primary", carbon_price=120.0, landfill_penalty=260.0, parameter_value=1.0),
    ]
    specs += [PolicySpec(f"carbon_{v}", "carbon_sweep", carbon_price=float(v), parameter_value=float(v)) for v in (0, 60, 120, 180, 240)]
    specs += [PolicySpec(f"landfill_{v}", "landfill_sweep", landfill_penalty=float(v), parameter_value=float(v)) for v in (0, 130, 260, 390)]
    specs += [PolicySpec(f"maxopen_{v}", "max_open_sweep", max_open=v, parameter_value=float(v)) for v in (70, 80, 92, 104, 114)]
    specs += [PolicySpec(f"coverage_{v:.2f}", "coverage_sweep", coverage_ratio=v, parameter_value=v) for v in (0.85, 0.90, 0.95)]
    specs += [PolicySpec(f"fixed_{v:.2f}", "fixed_cost_sweep", fixed_cost_scale=v, parameter_value=v) for v in (0.75, 1.00, 1.25)]
    specs += [PolicySpec(f"capex_{v:.2f}", "capex_sweep", capex_scale=v, parameter_value=v) for v in (0.75, 1.00, 1.25)]
    specs += [PolicySpec(f"operating_{v:.2f}", "operating_cost_sweep", operating_cost_scale=v, parameter_value=v) for v in (0.80, 1.00, 1.20)]
    specs += [PolicySpec(f"demand_{v:.2f}", "demand_scale_sweep", demand_scale=v, parameter_value=v) for v in (0.90, 1.00, 1.10)]
    specs += [
        PolicySpec(
            f"shortage_penalty_{int(v)}",
            "base_shortage_penalty_sweep",
            base_shortage_penalty=float(v),
            parameter_value=float(v),
        )
        for v in (300, 460, 620)
    ]
    specs += [
        PolicySpec(
            f"shortage_ef_{v:.2f}",
            "shortage_emission_sweep",
            carbon_price=120.0,
            shortage_emission=v,
            parameter_value=v,
        )
        for v in (0.20, 0.35, 0.50)
    ]
    return specs


def run_single_period_suite(bundle: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for spec in single_period_specs():
        print(f"[single] {spec.name}", flush=True)
        rows.append(solve_single_period(bundle, spec))
    out = pd.DataFrame(rows)
    baseline = out[(out["group"] == "primary") & (out["name"] == "baseline")].iloc[0]
    out["cost_change_pct_vs_primary_baseline"] = (out["reevaluated_objective_yuan"] / baseline["reevaluated_objective_yuan"] - 1.0) * 100.0
    out["emission_reduction_pct_vs_primary_baseline"] = (1.0 - out["expected_emission_tco2"] / baseline["expected_emission_tco2"]) * 100.0
    out["shortage_reduction_pct_vs_primary_baseline"] = (1.0 - out["expected_shortage_ton"] / baseline["expected_shortage_ton"]) * 100.0
    out.to_csv(OUT_DIR / "single_period_exact_results.csv", index=False)
    return out


def _trajectory_nodes(city: pd.DataFrame, pred: pd.DataFrame) -> List[Dict[str, object]]:
    codes = city["city_code_final"].astype(int).tolist()
    nodes = []
    for family, probability in FAMILY_WEIGHTS.items():
        for t, year in enumerate(PLANNING_YEARS):
            sub = pred[(pred["scenario"] == family) & (pred["year"] == year)].set_index("city_code_final")
            if not set(codes).issubset(set(sub.index.astype(int))):
                raise RuntimeError(f"Incomplete trajectory for {family}, {year}")
            demand = sub.reindex(codes)["cdw_pred_final"].to_numpy(float)
            nodes.append(
                {
                    "id": f"{family}_{year}",
                    "family": family,
                    "probability": probability,
                    "period": t,
                    "year": year,
                    "demand": demand,
                }
            )
    return nodes


def _write_multiperiod_lp(
    city: pd.DataFrame,
    fac: pd.DataFrame,
    edge: pd.DataFrame,
    nodes: List[Dict[str, object]],
    spec: PolicySpec,
    path: Path,
) -> None:
    city_to_idx, fac_to_idx = build_index(city, fac)
    edge_arrays = prepare_edge_arrays(edge, city_to_idx, fac_to_idx)
    n_fac = len(fac)
    n_city = len(city)
    edge_city = edge_arrays["city_idx"]
    edge_fac = edge_arrays["fac_idx"]
    cap_min = fac["capacity_min_ton"].to_numpy(float)
    cap_max = fac["capacity_max_ton"].to_numpy(float)
    fixed = fac["fixed_cost_yuan"].to_numpy(float) * spec.fixed_cost_scale
    capex = fac["capacity_cost_yuan_per_ton"].to_numpy(float) * spec.capex_scale
    proc_cost = fac["processing_cost_yuan_per_t"].to_numpy(float)
    proc_emission = fac["processing_emission_tco2_per_t"].to_numpy(float)
    city_landfill = city["landfill_cost_yuan_per_t"].to_numpy(float)
    edge_cost = edge_arrays["unit_transport_cost"] + proc_cost[edge_fac] + spec.carbon_price * (
        edge_arrays["transport_emission"] + proc_emission[edge_fac]
    )
    shortage_cost = spec.base_shortage_penalty + spec.landfill_penalty + city_landfill + spec.carbon_price * spec.shortage_emission
    city_edge = {i: np.where(edge_city == i)[0].tolist() for i in range(n_city)}
    fac_edge = {j: np.where(edge_fac == j)[0].tolist() for j in range(n_fac)}

    obj: Dict[str, float] = defaultdict(float)
    for j in range(n_fac):
        for t, year in enumerate(PLANNING_YEARS):
            disc = discount_factor(year)
            obj[f"x_{j}_{t}"] += disc * fixed[j]
            obj[f"u_{j}_{t}"] += disc * capex[j]
            if t > 0:
                obj[f"x_{j}_{t-1}"] -= disc * fixed[j]
                obj[f"u_{j}_{t-1}"] -= disc * capex[j]
    for n, node in enumerate(nodes):
        weight = float(node["probability"]) * discount_factor(int(node["year"]))
        for e in range(len(edge)):
            obj[f"y_{n}_{e}"] += weight * float(edge_cost[e])
        for i in range(n_city):
            obj[f"z_{n}_{i}"] += weight * float(shortage_cost[i])

    with path.open("w", encoding="utf-8") as f:
        f.write("Minimize\n obj:\n")
        _write_expr_terms(f, [(coef, name) for name, coef in obj.items()])
        f.write("Subject To\n")
        for n, node in enumerate(nodes):
            t = int(node["period"])
            demand = np.asarray(node["demand"], dtype=float)
            for i in range(n_city):
                f.write(f" demand_{n}_{i}:\n")
                _write_expr_terms(f, [(1.0, f"y_{n}_{e}") for e in city_edge[i]] + [(1.0, f"z_{n}_{i}")])
                f.write(f"   = {demand[i]:.10f}\n")
            for j in range(n_fac):
                f.write(f" cap_{n}_{j}:\n")
                _write_expr_terms(f, [(1.0, f"y_{n}_{e}") for e in fac_edge[j]] + [(-1.0, f"u_{j}_{t}")])
                f.write("   <= 0\n")
        for t in range(len(PLANNING_YEARS)):
            for j in range(n_fac):
                f.write(f" capmax_{j}_{t}:\n")
                _write_expr_terms(f, [(1.0, f"u_{j}_{t}"), (-cap_max[j], f"x_{j}_{t}")])
                f.write("   <= 0\n")
                f.write(f" capmin_{j}_{t}:\n")
                _write_expr_terms(f, [(cap_min[j], f"x_{j}_{t}"), (-1.0, f"u_{j}_{t}")])
                f.write("   <= 0\n")
                if t > 0:
                    f.write(f" irrev_x_{j}_{t}: + 1 x_{j}_{t} - 1 x_{j}_{t-1} >= 0\n")
                    f.write(f" irrev_u_{j}_{t}: + 1 u_{j}_{t} - 1 u_{j}_{t-1} >= 0\n")
            expected_demand = sum(
                float(node["probability"]) * float(np.asarray(node["demand"]).sum())
                for node in nodes
                if int(node["period"]) == t
            )
            f.write(f" coverage_{t}:\n")
            _write_expr_terms(f, [(1.0, f"u_{j}_{t}") for j in range(n_fac)])
            f.write(f"   >= {spec.coverage_ratio * expected_demand:.10f}\n")
            f.write(f" open_lb_{t}:\n")
            _write_expr_terms(f, [(1.0, f"x_{j}_{t}") for j in range(n_fac)])
            f.write("   >= 24\n")
            f.write(f" open_ub_{t}:\n")
            _write_expr_terms(f, [(1.0, f"x_{j}_{t}") for j in range(n_fac)])
            f.write(f"   <= {spec.max_open}\n")
        f.write("Bounds\n")
        for t in range(len(PLANNING_YEARS)):
            for j in range(n_fac):
                f.write(f" 0 <= u_{j}_{t} <= {cap_max[j]:.10f}\n")
        f.write("Binaries\n")
        for t in range(len(PLANNING_YEARS)):
            for j in range(n_fac):
                f.write(f" x_{j}_{t}\n")
        f.write("End\n")


def solve_multiperiod(bundle: Dict[str, object], spec: PolicySpec) -> Tuple[Dict[str, object], pd.DataFrame]:
    city = bundle["city"]
    fac = bundle["facility"].copy()
    edge = bundle["edge"]
    nodes = _trajectory_nodes(city, bundle["pred"])
    lp = TMP_DIR / f"multi_{spec.name}.lp"
    sol = TMP_DIR / f"multi_{spec.name}.sol"
    log = LOG_DIR / f"multi_{spec.name}.log"
    _write_multiperiod_lp(city, fac, edge, nodes, spec, lp)
    parsed = _run_cplex(lp, sol, log, time_limit=1200)
    if not sol.exists():
        return ({**asdict(spec), **parsed, "success": False}, pd.DataFrame())
    objective, values = _solution_values(sol)
    n_fac = len(fac)
    x = np.array([[values.get(f"x_{j}_{t}", 0.0) for t in range(len(PLANNING_YEARS))] for j in range(n_fac)])
    cap = np.array([[values.get(f"u_{j}_{t}", 0.0) for t in range(len(PLANNING_YEARS))] for j in range(n_fac)])
    city_to_idx, fac_to_idx = build_index(city, fac)
    edge_arrays = prepare_edge_arrays(edge, city_to_idx, fac_to_idx)
    mats = _build_subproblem_matrices(len(city), len(fac), edge_arrays["city_idx"], edge_arrays["fac_idx"])
    proc_cost = fac["processing_cost_yuan_per_t"].to_numpy(float)
    proc_emission = fac["processing_emission_tco2_per_t"].to_numpy(float)
    landfill = city["landfill_cost_yuan_per_t"].to_numpy(float)
    cfg = BendersConfig(
        shortage_penalty=spec.base_shortage_penalty,
        landfill_penalty=spec.landfill_penalty,
        carbon_price_per_tco2=spec.carbon_price,
        shortage_emission_tco2_per_t=spec.shortage_emission,
    )
    period_rows = []
    discounted_recourse = 0.0
    for t, year in enumerate(PLANNING_YEARS):
        expected_cost = 0.0
        expected_emission = 0.0
        expected_shortage = 0.0
        for node in [n for n in nodes if int(n["period"]) == t]:
            sub = solve_subproblem(
                np.asarray(node["demand"]),
                cap[:, t],
                edge_arrays,
                proc_cost,
                proc_emission,
                landfill,
                cfg,
                mats,
            )
            p = float(node["probability"])
            expected_cost += p * float(sub["objective"])
            expected_emission += p * float(sub["total_emission_tco2"])
            expected_shortage += p * float(np.asarray(sub["z"]).sum())
        discounted_recourse += discount_factor(year) * expected_cost
        period_rows.append(
            {
                "policy": spec.name,
                "year": year,
                "open_facilities": int((x[:, t] > 0.5).sum()),
                "installed_capacity_ton": float(cap[:, t].sum()),
                "expected_operating_cost_yuan": expected_cost,
                "expected_emission_tco2": expected_emission,
                "expected_shortage_ton": expected_shortage,
                "discount_factor": discount_factor(year),
                "data_version": VERSION,
            }
        )
    fixed = fac["fixed_cost_yuan"].to_numpy(float) * spec.fixed_cost_scale
    capex = fac["capacity_cost_yuan_per_ton"].to_numpy(float) * spec.capex_scale
    investment = 0.0
    for t, year in enumerate(PLANNING_YEARS):
        dx = x[:, t] - (x[:, t - 1] if t else 0.0)
        du = cap[:, t] - (cap[:, t - 1] if t else 0.0)
        investment += discount_factor(year) * (fixed @ dx + capex @ du)
    reevaluated = investment + discounted_recourse
    sol_rows = []
    for j, row in fac.iterrows():
        for t, year in enumerate(PLANNING_YEARS):
            sol_rows.append(
                {
                    "policy": spec.name,
                    "site_id": row["site_id"],
                    "facility_city": row["facility_city"],
                    "year": year,
                    "open": int(x[j, t] > 0.5),
                    "installed_capacity_ton": cap[j, t],
                    "data_version": VERSION,
                }
            )
    pd.DataFrame(sol_rows).to_csv(SOLUTION_DIR / f"multi_{spec.name}_facility_period_solution.csv", index=False)
    summary = {
        **asdict(spec),
        "success": parsed["status"] in {"optimal", "time_limit_feasible"},
        "solver_status": parsed["status"],
        "runtime_sec": parsed["runtime_sec"],
        "cplex_objective_yuan": objective,
        "reevaluated_objective_yuan": reevaluated,
        "objective_crosscheck_relative_error": (reevaluated - objective) / max(abs(objective), 1.0),
        "best_bound_yuan": parsed["best_bound"],
        "solver_gap": parsed["gap"],
        "discounted_investment_cost_yuan": investment,
        "discounted_expected_recourse_cost_yuan": discounted_recourse,
        "data_version": VERSION,
    }
    return summary, pd.DataFrame(period_rows)


def run_multiperiod_suite(bundle: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        PolicySpec("baseline", "primary"),
        PolicySpec("carbon_tax_120", "primary", carbon_price=120.0, parameter_value=120.0),
        PolicySpec("landfill_penalty_260", "primary", landfill_penalty=260.0, parameter_value=260.0),
        PolicySpec("combined_120_260", "primary", carbon_price=120.0, landfill_penalty=260.0, parameter_value=1.0),
    ]
    summaries = []
    periods = []
    for spec in specs:
        print(f"[multi] {spec.name}", flush=True)
        summary, period = solve_multiperiod(bundle, spec)
        summaries.append(summary)
        periods.append(period)
    summary_df = pd.DataFrame(summaries)
    period_df = pd.concat(periods, ignore_index=True)
    summary_df.to_csv(OUT_DIR / "multiperiod_exact_summary.csv", index=False)
    period_df.to_csv(OUT_DIR / "multiperiod_exact_by_period.csv", index=False)
    return summary_df, period_df


def write_manifest(bundle: Dict[str, object]) -> None:
    metadata = {
        "version": VERSION,
        "created": "2026-07-13",
        "city_count": len(bundle["city"]),
        "facility_count": len(bundle["facility"]),
        "road_arc_count": len(bundle["edge"]),
        "single_period_scenarios": len(bundle["probs"]),
        "multiperiod_years": PLANNING_YEARS,
        "reported_milestone_years": REPORTING_YEARS,
        "trajectory_weights": FAMILY_WEIGHTS,
        "annual_discount_rate": ANNUAL_DISCOUNT_RATE,
        "discount_convention": "(1 + annual_discount_rate)^-(year - 2025)",
        "primary_scenario_seed": PRIMARY_SCENARIO_SEED,
        "scenario_generator": GENERATOR_VERSION,
        "road_emission_kg_per_tkm": ROAD_EMISSION_KG_PER_TKM,
        "shortage_emission_tco2_per_t": SHORTAGE_EMISSION_TCO2_PER_T,
        "cplex_numerical_emphasis": True,
        "nested_coverage_mip_start": "Primary baseline facility solution",
        "solver": "IBM ILOG CPLEX 22.1.1 command-line optimizer",
    }
    metadata_path = OUT_DIR / "revision_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    files = []
    source_paths = list(SOURCE_PATHS.values()) + [SCRIPT_DIR / name for name in SOURCE_CODE_NAMES]
    generated = sorted(
        [p for p in DATA_DIR.rglob("*") if p.is_file()]
        + [
            p
            for p in OUT_DIR.rglob("*")
            if p.is_file()
            and "tmp_lp" not in p.parts
            and p.name != "sha256_manifest.csv"
        ]
    )
    for role, paths in (("source", source_paths), ("generated", generated)):
        for path in paths:
            files.append(
                {
                    "role": role,
                    "path": public_path_for(path).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(files).to_csv(OUT_DIR / "sha256_manifest.csv", index=False)


def main() -> None:
    for directory in (DATA_DIR, OUT_DIR, LOG_DIR, SOLUTION_DIR, TMP_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    bundle = build_revision_inputs()
    single = run_single_period_suite(bundle)
    multi_summary, multi_period = run_multiperiod_suite(bundle)
    write_manifest(bundle)
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print(single[single["group"] == "primary"].to_string(index=False))
    print(multi_summary.to_string(index=False))
    print(multi_period.to_string(index=False))


if __name__ == "__main__":
    main()
