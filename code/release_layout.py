#!/usr/bin/env python3
"""Resolve local-development and public-release paths for the CDW study."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


VERSION = "cdw_rln_v2.2.0-20260713"
SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_MODE = SCRIPT_DIR.name == "code"

if ARCHIVE_MODE:
    ROOT = SCRIPT_DIR.parent
    STUDY_DIR = ROOT
    DATA_DIR = ROOT / "data/derived" / VERSION
    OUT_DIR = ROOT / "results" / VERSION
    REV_DIR = ROOT / "artifacts"
    SOURCE_PATHS = {
        "city_master": ROOT / "data/source/city_master.csv",
        "demand_projections": ROOT / "data/source/cdw_demand_projections_2023_2035.csv",
        "regional_operating_parameters": ROOT / "data/source/regional_operating_parameters.csv",
        "candidate_sites": ROOT / "data/source/candidate_site_source.csv",
        "transport_matrix": ROOT / "data/source/transport_cost_matrix.csv",
    }
else:
    ROOT = SCRIPT_DIR.parents[1]
    STUDY_DIR = SCRIPT_DIR.parent
    DATA_DIR = ROOT / "data/processed" / VERSION
    OUT_DIR = ROOT / "output/optimization" / VERSION
    REV_DIR = STUDY_DIR / "submission/revision_20260713_v2.2"
    regional_parameter_candidates = sorted((ROOT / "data/processed").glob("*_parameters_for_optimization.csv"))
    if len(regional_parameter_candidates) != 1:
        raise RuntimeError("Expected exactly one regional operating-parameter source file")
    SOURCE_PATHS = {
        "city_master": ROOT / "data/processed/city_list.csv",
        "demand_projections": ROOT / "data/processed/cdw_predictions_2023_2035.csv",
        "regional_operating_parameters": regional_parameter_candidates[0],
        "candidate_sites": ROOT / "data/processed/facility_list.csv",
        "transport_matrix": ROOT / "data/data_package/E/transport_cost_matrix.csv",
    }

PUBLIC_SOURCE_PATHS = {
    "city_master": PurePosixPath("data/source/city_master.csv"),
    "demand_projections": PurePosixPath("data/source/cdw_demand_projections_2023_2035.csv"),
    "regional_operating_parameters": PurePosixPath("data/source/regional_operating_parameters.csv"),
    "candidate_sites": PurePosixPath("data/source/candidate_site_source.csv"),
    "transport_matrix": PurePosixPath("data/source/transport_cost_matrix.csv"),
}

SOURCE_CODE_NAMES = (
    "release_layout.py",
    "run_rigorous_revision.py",
    "scenario_generator.py",
    "stochastic_analysis.py",
    "run_stochastic_value_analysis.py",
    "run_scenario_stability.py",
    "run_network_density_sensitivity.py",
    "benders_solver.py",
    "benders_solver_advanced.py",
    "run_cplex_monolithic_benchmark.py",
    "scalability_instances.py",
)


def public_path_for(path: Path) -> PurePosixPath:
    """Return the neutral path used in manifests and public archives."""
    resolved = path.resolve()
    for key, source in SOURCE_PATHS.items():
        if resolved == source.resolve():
            return PUBLIC_SOURCE_PATHS[key]
    if resolved.parent == SCRIPT_DIR.resolve() and resolved.name in SOURCE_CODE_NAMES:
        return PurePosixPath("code") / resolved.name
    if resolved.is_relative_to(DATA_DIR.resolve()):
        return PurePosixPath("data/derived") / VERSION / PurePosixPath(*resolved.relative_to(DATA_DIR.resolve()).parts)
    if resolved.is_relative_to(OUT_DIR.resolve()):
        return PurePosixPath("results") / VERSION / PurePosixPath(*resolved.relative_to(OUT_DIR.resolve()).parts)
    raise ValueError(f"No public release path defined for {path}")


def resolve_public_path(public_path: str) -> Path:
    """Map a public manifest path to the corresponding file in either layout."""
    pure = PurePosixPath(public_path)
    for key, published in PUBLIC_SOURCE_PATHS.items():
        if pure == published:
            return SOURCE_PATHS[key]
    if pure.parts[:1] == ("code",):
        return SCRIPT_DIR / Path(*pure.parts[1:])
    derived_prefix = ("data", "derived", VERSION)
    if pure.parts[: len(derived_prefix)] == derived_prefix:
        return DATA_DIR / Path(*pure.parts[len(derived_prefix) :])
    result_prefix = ("results", VERSION)
    if pure.parts[: len(result_prefix)] == result_prefix:
        return OUT_DIR / Path(*pure.parts[len(result_prefix) :])
    raise ValueError(f"Unknown public release path: {public_path}")
