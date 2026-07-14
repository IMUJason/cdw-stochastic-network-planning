#!/usr/bin/env python3
"""Run monolithic deterministic-equivalent CPLEX benchmarks for the CDW model.

This script builds the same scalability instances used in the journal-upgrade
pipeline and solves their deterministic equivalent with the commercial CPLEX
optimizer via the command-line interface. It is intended to provide an
external exact-solver benchmark against the custom Benders implementations.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scalability_instances import build_scalability_instances
from release_layout import DATA_DIR as RELEASE_DATA_DIR
from release_layout import OUT_DIR as RELEASE_OUT_DIR
from release_layout import ROOT


# CPLEX binary lookup order:
# 1) Environment variable CDW_CPLEX_BIN (preferred for portability)
# 2) Default Windows path of CPLEX Studio 22.1.1
# 3) macOS/Linux fallback paths (legacy development environments)
_DEFAULT_CPLEX_CANDIDATES = [
    Path("C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.exe"),
    Path("C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cplex/bin/x64_win64/cplex.bat"),
    Path("/Applications/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex"),
    Path("/opt/ibm/ILOG/CPLEX_Studio2211/cplex/bin/x86-64_linux/cplex"),
]


def _resolve_cplex_bin() -> Path:
    env_bin = os.environ.get("CDW_CPLEX_BIN")
    if env_bin:
        p = Path(env_bin)
        if p.exists():
            return p
    for cand in _DEFAULT_CPLEX_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "CPLEX binary not found. Set environment variable CDW_CPLEX_BIN "
        "to the absolute path of the cplex executable, or install CPLEX Studio "
        "to one of the default locations."
    )


CPLEX_BIN = _resolve_cplex_bin()
OUT_DIR = RELEASE_OUT_DIR / "benchmarks"
QA_DIR = RELEASE_OUT_DIR / "benchmark_reports"
TMP_DIR = RELEASE_OUT_DIR / "benchmark_lp"


@dataclass
class MonolithicConfig:
    coverage_ratio: float
    min_open_facilities: int
    max_open_facilities: int
    shortage_penalty: float
    landfill_penalty: float
    subsidy_per_ton: float
    carbon_price_per_tco2: float
    shortage_emission_tco2_per_t: float = 0.35
    time_limit_sec: int = 300
    threads: int = 8
    mip_gap: float = 1e-9


def _city_and_facility_index(city_df: pd.DataFrame, fac_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    city_ids = city_df["city_code_final"].astype(int).tolist()
    fac_ids = fac_df["site_id"].astype(int).tolist()
    return {cid: i for i, cid in enumerate(city_ids)}, {fid: j for j, fid in enumerate(fac_ids)}


def _prepare_instance(
    city_df: pd.DataFrame,
    fac_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    scen_df: pd.DataFrame,
    probs: np.ndarray,
    cfg: MonolithicConfig,
) -> Dict[str, object]:
    city_to_idx, fac_to_idx = _city_and_facility_index(city_df, fac_df)

    edge = edge_df[["city_code", "site_id", "unit_transport_cost", "transport_emission_tco2_per_t"]].copy()
    edge["city_idx"] = edge["city_code"].astype(int).map(city_to_idx)
    edge["fac_idx"] = edge["site_id"].astype(int).map(fac_to_idx)
    edge = edge.dropna(subset=["city_idx", "fac_idx"]).copy()
    edge["city_idx"] = edge["city_idx"].astype(int)
    edge["fac_idx"] = edge["fac_idx"].astype(int)

    scen_ids = sorted(scen_df["scenario_id"].unique())
    n_scen = len(scen_ids)
    n_city = len(city_df)
    demand_matrix = np.zeros((n_scen, n_city), dtype=float)
    for s_idx, sid in enumerate(scen_ids):
        sub = scen_df[scen_df["scenario_id"] == sid][["city_code_final", "demand_ton"]]
        for _, row in sub.iterrows():
            c_idx = city_to_idx.get(int(row["city_code_final"]))
            if c_idx is not None:
                demand_matrix[s_idx, c_idx] = float(row["demand_ton"])

    cap_max = fac_df["capacity_max_ton"].to_numpy(dtype=float)
    cap_min = fac_df["capacity_min_ton"].to_numpy(dtype=float)
    fixed_cost = fac_df["fixed_cost_yuan"].to_numpy(dtype=float)
    capex_cost = fac_df["capacity_cost_yuan_per_ton"].to_numpy(dtype=float)
    proc_cost = fac_df["processing_cost_yuan_per_t"].to_numpy(dtype=float)
    proc_emission = fac_df["processing_emission_tco2_per_t"].to_numpy(dtype=float)
    city_landfill_cost = city_df["landfill_cost_yuan_per_t"].to_numpy(dtype=float)

    target_capacity = float(cfg.coverage_ratio * float(np.dot(probs, demand_matrix.sum(axis=1))))

    edge_unit_cost = (
        edge["unit_transport_cost"].to_numpy(dtype=float)
        + proc_cost[edge["fac_idx"].to_numpy(dtype=int)]
        - cfg.subsidy_per_ton
        + cfg.carbon_price_per_tco2
        * (
            edge["transport_emission_tco2_per_t"].to_numpy(dtype=float)
            + proc_emission[edge["fac_idx"].to_numpy(dtype=int)]
        )
    )
    edge_unit_cost = np.maximum(edge_unit_cost, 0.0)

    shortage_cost = (
        cfg.shortage_penalty
        + cfg.landfill_penalty
        + city_landfill_cost
        + cfg.carbon_price_per_tco2 * cfg.shortage_emission_tco2_per_t
    )

    city_edge_map: Dict[int, List[int]] = {i: [] for i in range(n_city)}
    fac_edge_map: Dict[int, List[int]] = {j: [] for j in range(len(fac_df))}
    for e_idx, row in edge.reset_index(drop=True).iterrows():
        city_edge_map[int(row["city_idx"])].append(e_idx)
        fac_edge_map[int(row["fac_idx"])].append(e_idx)

    return {
        "edge": edge.reset_index(drop=True),
        "probs": probs.astype(float),
        "demand_matrix": demand_matrix,
        "cap_max": cap_max,
        "cap_min": cap_min,
        "fixed_cost": fixed_cost,
        "capex_cost": capex_cost,
        "edge_unit_cost": edge_unit_cost,
        "shortage_cost": shortage_cost,
        "target_capacity": target_capacity,
        "city_edge_map": city_edge_map,
        "fac_edge_map": fac_edge_map,
        "n_city": n_city,
        "n_fac": len(fac_df),
        "n_scen": n_scen,
        "n_edge": len(edge),
        "n_var": int(2 * len(fac_df) + n_scen * (len(edge) + n_city)),
        "n_constraint": int(n_scen * (n_city + len(fac_df)) + 2 * len(fac_df) + 2),
    }


def _write_expr_terms(handle, terms: List[Tuple[float, str]], prefix: str = "  ", max_terms_per_line: int = 8) -> None:
    current: List[str] = []
    for coeff, name in terms:
        if abs(coeff) < 1e-14:
            continue
        sign = "+" if coeff >= 0 else "-"
        token = f"{sign} {abs(coeff):.10f} {name}"
        current.append(token)
        if len(current) >= max_terms_per_line:
            handle.write(prefix + " ".join(current) + "\n")
            current = []
    if current:
        handle.write(prefix + " ".join(current) + "\n")


def _write_lp(instance: Dict[str, object], lp_path: Path) -> None:
    edge = instance["edge"]
    probs = instance["probs"]
    demand_matrix = instance["demand_matrix"]
    cap_max = instance["cap_max"]
    cap_min = instance["cap_min"]
    fixed_cost = instance["fixed_cost"]
    capex_cost = instance["capex_cost"]
    edge_unit_cost = instance["edge_unit_cost"]
    shortage_cost = instance["shortage_cost"]
    target_capacity = float(instance["target_capacity"])
    city_edge_map = instance["city_edge_map"]
    fac_edge_map = instance["fac_edge_map"]
    n_city = int(instance["n_city"])
    n_fac = int(instance["n_fac"])
    n_scen = int(instance["n_scen"])

    with lp_path.open("w", encoding="utf-8") as f:
        f.write("Minimize\n")
        f.write(" obj:\n")
        obj_terms: List[Tuple[float, str]] = []
        for j in range(n_fac):
            obj_terms.append((float(fixed_cost[j]), f"x_{j}"))
            obj_terms.append((float(capex_cost[j]), f"u_{j}"))
        for s in range(n_scen):
            p = float(probs[s])
            for e_idx in range(len(edge)):
                obj_terms.append((p * float(edge_unit_cost[e_idx]), f"y_{s}_{e_idx}"))
            for i in range(n_city):
                obj_terms.append((p * float(shortage_cost[i]), f"z_{s}_{i}"))
        _write_expr_terms(f, obj_terms)

        f.write("Subject To\n")
        # Demand balance by scenario and city.
        for s in range(n_scen):
            for i in range(n_city):
                terms = [(1.0, f"y_{s}_{e_idx}") for e_idx in city_edge_map[i]]
                terms.append((1.0, f"z_{s}_{i}"))
                f.write(f" demand_{s}_{i}:\n")
                _write_expr_terms(f, terms)
                f.write(f"   = {float(demand_matrix[s, i]):.10f}\n")

        # Facility capacity by scenario.
        for s in range(n_scen):
            for j in range(n_fac):
                terms = [(1.0, f"y_{s}_{e_idx}") for e_idx in fac_edge_map[j]]
                terms.append((-1.0, f"u_{j}"))
                f.write(f" cap_{s}_{j}:\n")
                _write_expr_terms(f, terms)
                f.write("   <= 0\n")

        for j in range(n_fac):
            f.write(f" cap_link_max_{j}:\n")
            _write_expr_terms(f, [(1.0, f"u_{j}"), (-float(cap_max[j]), f"x_{j}")])
            f.write("   <= 0\n")

            f.write(f" cap_link_min_{j}:\n")
            _write_expr_terms(f, [(float(cap_min[j]), f"x_{j}"), (-1.0, f"u_{j}")])
            f.write("   <= 0\n")

        f.write(" coverage:\n")
        _write_expr_terms(f, [(1.0, f"u_{j}") for j in range(n_fac)])
        f.write(f"   >= {target_capacity:.10f}\n")

        f.write(" open_lb:\n")
        _write_expr_terms(f, [(1.0, f"x_{j}") for j in range(n_fac)])
        f.write(f"   >= {int(instance['min_open'])}\n")

        f.write(" open_ub:\n")
        _write_expr_terms(f, [(1.0, f"x_{j}") for j in range(n_fac)])
        f.write(f"   <= {int(instance['max_open'])}\n")

        f.write("Bounds\n")
        for j in range(n_fac):
            f.write(f" 0 <= u_{j} <= {float(cap_max[j]):.10f}\n")

        f.write("Binaries\n")
        for j in range(n_fac):
            f.write(f" x_{j}\n")
        f.write("End\n")


def _parse_cplex_log(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "status": "unknown",
        "objective": np.nan,
        "best_bound": np.nan,
        "gap": np.nan,
        "nodes": np.nan,
        "iterations": np.nan,
    }

    patterns = [
        (
            "optimal",
            re.compile(
                r"MIP - Integer optimal(?: solution|, tolerance \([^)]+\)):\s+Objective =\s+([0-9eE+\-.]+)"
            ),
        ),
        (
            "time_limit_feasible",
            re.compile(
                r"MIP - Time limit exceeded, integer feasible:\s+Objective =\s+([0-9eE+\-.]+)\s+best bound =\s+([0-9eE+\-.]+), gap =\s+([0-9eE+\-.]+)%"
            ),
        ),
        (
            "time_limit_no_feasible",
            re.compile(
                r"MIP - Time limit exceeded, no integer solution, best bound =\s+([0-9eE+\-.]+)"
            ),
        ),
    ]

    for status, pat in patterns:
        matches = pat.findall(text)
        if not matches:
            continue
        last = matches[-1]
        out["status"] = status
        if status == "optimal":
            out["objective"] = float(last)
            out["best_bound"] = float(last)
            out["gap"] = 0.0
        elif status == "time_limit_feasible":
            out["objective"] = float(last[0])
            out["best_bound"] = float(last[1])
            out["gap"] = float(last[2]) / 100.0
        else:
            out["best_bound"] = float(last)
        break

    node_match = re.findall(r"Solution time =\s+[0-9.]+\s+sec\.\s+Iterations =\s+([0-9]+)\s+Nodes =\s+([0-9]+)", text)
    if node_match:
        iters, nodes = node_match[-1]
        out["iterations"] = float(iters)
        out["nodes"] = float(nodes)

    if out["status"] == "unknown":
        if "infeasible" in text.lower():
            out["status"] = "infeasible"
        elif "CPLEX Error" in text:
            out["status"] = "error"

    return out


def _run_cplex(lp_path: Path, log_path: Path, cfg: MonolithicConfig) -> Dict[str, object]:
    cmd = [
        str(CPLEX_BIN),
        "-c",
        f'read "{lp_path}"',
        f"set timelimit {cfg.time_limit_sec}",
        f"set threads {cfg.threads}",
        "set mip tolerances mipgap 1e-9",
        "optimize",
        "display solution objective",
        "quit",
    ]
    tic = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(lp_path.parent))
    runtime = time.time() - tic
    log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(log_text, encoding="utf-8")
    parsed = _parse_cplex_log(log_text)
    parsed["runtime_sec"] = runtime
    parsed["return_code"] = proc.returncode
    return parsed


def run_benchmark() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    city = pd.read_csv(RELEASE_DATA_DIR / "city_list_complete_case.csv")
    fac = pd.read_csv(RELEASE_DATA_DIR / "facility_list_transparent_proxy.csv")
    edge = pd.read_csv(RELEASE_DATA_DIR / "road_transport_edges.csv")
    scen = pd.read_csv(RELEASE_DATA_DIR / "stochastic_scenarios_2030.csv")
    probs = np.load(RELEASE_DATA_DIR / "scenario_probabilities_2030.npy")

    instances = build_scalability_instances(city, fac, edge, scen, probs)
    rows: List[Dict[str, object]] = []

    for inst in instances:
        size_metric = int(inst["size_metric"])
        tag = f"cplex_monolithic_{inst['n_city']}c_{inst['n_facility']}f_{inst['n_scenario']}s"
        lp_path = TMP_DIR / f"{tag}.lp"
        log_path = OUT_DIR / f"{tag}.log"

        cfg = MonolithicConfig(
            coverage_ratio=float(inst["coverage_ratio"]),
            min_open_facilities=int(inst["min_open_facilities"]),
            max_open_facilities=int(inst["max_open_facilities"]),
            shortage_penalty=float(inst["shortage_penalty"]),
            landfill_penalty=float(inst["landfill_penalty"]),
            subsidy_per_ton=float(inst["subsidy_per_ton"]),
            carbon_price_per_tco2=float(inst["carbon_price_per_tco2"]),
        )

        prepared = _prepare_instance(inst["city"], inst["fac"], inst["edge"], inst["scen"], inst["probs"], cfg)
        prepared["min_open"] = cfg.min_open_facilities
        prepared["max_open"] = cfg.max_open_facilities

        _write_lp(prepared, lp_path)
        # CPLEX 22.1.1 on Windows fails to read LP files whose path contains
        # non-ASCII characters even when the path is valid UTF-8. Copy the LP
        # to an ASCII staging directory before invoking CPLEX, then write the
        # log back to the canonical path.
        ascii_lp_dir = Path(os.environ.get("CDW_CPLEX_ASCII_DIR", r"C:/tmp/cdw_lp"))
        ascii_lp_dir.mkdir(parents=True, exist_ok=True)
        ascii_lp = ascii_lp_dir / lp_path.name
        shutil.copy(lp_path, ascii_lp)
        parsed = _run_cplex(ascii_lp, log_path, cfg)

        rows.append(
            {
                "solver": "monolithic_cplex",
                "n_city": inst["n_city"],
                "n_facility": inst["n_facility"],
                "n_scenario": inst["n_scenario"],
                "size_metric": size_metric,
                "runtime_sec": float(parsed["runtime_sec"]),
                "final_gap": float(parsed["gap"]) if np.isfinite(parsed["gap"]) else np.nan,
                "success": parsed["status"] in {"optimal", "time_limit_feasible"},
                "status": parsed["status"],
                "objective": float(parsed["objective"]) if np.isfinite(parsed["objective"]) else np.nan,
                "best_bound": float(parsed["best_bound"]) if np.isfinite(parsed["best_bound"]) else np.nan,
                "nodes": float(parsed["nodes"]) if np.isfinite(parsed["nodes"]) else np.nan,
                "iterations": float(parsed["iterations"]) if np.isfinite(parsed["iterations"]) else np.nan,
                "time_limit_sec": int(cfg.time_limit_sec),
                "threads": int(cfg.threads),
                "log_path": str(log_path.relative_to(ROOT)),
                "n_var": int(prepared["n_var"]),
                "n_constraint": int(prepared["n_constraint"]),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "monolithic_cplex_benchmark.csv", index=False)

    comparison = None
    sc_path = OUT_DIR / "scalability_results.csv"
    if sc_path.exists():
        sc = pd.read_csv(sc_path)
        acc = sc[sc["solver"] == "accelerated_multicut"][
            ["n_city", "n_facility", "n_scenario", "size_metric", "runtime_sec", "final_gap"]
        ].rename(columns={"runtime_sec": "accelerated_runtime_sec", "final_gap": "accelerated_final_gap"})
        comparison = out.merge(acc, on=["n_city", "n_facility", "n_scenario", "size_metric"], how="left")
        if "accelerated_runtime_sec" in comparison.columns:
            comparison["cplex_vs_accelerated_runtime_ratio"] = comparison["runtime_sec"] / comparison["accelerated_runtime_sec"]

    lines = [
        "# Commercial Exact-Solver Benchmark Report",
        "",
        "Monolithic deterministic-equivalent benchmark solved with IBM ILOG CPLEX 22.1.1 command-line optimizer.",
        "",
        out[
            [
                "n_city",
                "n_facility",
                "n_scenario",
                "size_metric",
                "runtime_sec",
                "final_gap",
                "status",
                "objective",
                "best_bound",
                "nodes",
                "n_var",
                "n_constraint",
            ]
        ].to_markdown(index=False),
        "",
        "The benchmark is intended as an external exact-solver reference for the same scaled instances used in the Benders scalability experiment.",
    ]
    if comparison is not None:
        lines.extend(
            [
                "",
                "## Comparison against accelerated Benders on aligned scales",
                "",
                comparison[
                    [
                        "size_metric",
                        "runtime_sec",
                        "accelerated_runtime_sec",
                        "cplex_vs_accelerated_runtime_ratio",
                        "final_gap",
                        "accelerated_final_gap",
                    ]
                ].to_markdown(index=False),
            ]
        )
    (QA_DIR / "commercial_solver_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")

    # Remove temporary LP files to keep repository size under control.
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return out


if __name__ == "__main__":
    if not CPLEX_BIN.exists():
        raise SystemExit(f"CPLEX executable not found: {CPLEX_BIN}")
    df = run_benchmark()
    print(df.to_string(index=False))
