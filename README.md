# CDW stochastic network planning

Minimal data and optimization code for the evidence release
`cdw_rln_v2.2.0-20260713`. The model addresses stochastic facility location
and phased capacity planning for a construction and demolition waste (CDW)
reverse-logistics network.

This repository contains only computational inputs and core optimization
code. It does not contain the manuscript, figures, figure-generation code,
submission files, or proprietary solver software.

## Scope

The released instance contains:

- 371 source city-level units, of which 333 have complete 2025--2035 demand
  trajectories and enter the optimization;
- 114 candidate planning locations;
- 42,294 source city--candidate road-distance records, reduced by the model to
  the 12 lowest-cost arcs per retained city (3,996 arcs);
- three demand families and a 16-scenario primary 2030 ensemble;
- annual phased capacity decisions from 2025 through 2035.

Candidate locations are provisioned planning points, not an observed plant
inventory. Demand is model-derived. Site capacities, fixed investment, and
capacity costs are planning proxies. Results must therefore be interpreted as
conditional planning evidence rather than site-specific investment advice.

## Repository layout

```text
code/          Core scenario, optimization, and evaluation scripts
data/source/   Five minimal input tables used by the scripts
data/derived/  Generated model-ready inputs (created at runtime)
results/       Generated solver outputs (created at runtime)
```

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for fields and units.

## Requirements

- Python 3.10 or newer
- NumPy, pandas, and SciPy (see `requirements.txt`)
- IBM ILOG CPLEX Optimization Studio 22.1.1 command-line executable

CPLEX is proprietary and is not distributed here. Point the workflow to an
installed executable when it is not in a default location:

```bash
export CDW_CPLEX_BIN=/absolute/path/to/cplex
```

On Windows, CPLEX 22.1.1 cannot read or write files under paths that contain
non-ASCII characters. The workflows detect this case and stage solver inputs in
an ASCII temporary directory, copying solutions back to their canonical
locations; set `CDW_CPLEX_ASCII_DIR` to choose the staging directory (defaults
to the system temporary directory).

Create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the core results

Run commands from the repository root:

```bash
PYTHONPATH=code python code/run_rigorous_revision.py
```

This command constructs the complete-case city set, road network, and primary
scenarios, then solves the single-period policy/sensitivity suite and the four
2025--2035 phased models. Outputs are written under
`results/cdw_rln_v2.2.0-20260713/`; generated inputs are written under
`data/derived/cdw_rln_v2.2.0-20260713/`.

Run the additional reported experiments after the core workflow:

```bash
PYTHONPATH=code python code/run_stochastic_value_analysis.py
PYTHONPATH=code python code/run_network_density_sensitivity.py
PYTHONPATH=code python code/run_scenario_stability.py
```

The stability experiment defaults to 10 independent training samples at 8,
16, 32, and 64 scenarios, evaluated on one 500-scenario holdout. It is the
longest workflow. A small execution check can be run with:

```bash
PYTHONPATH=code python code/run_scenario_stability.py \
  --counts 8 --replicates 2 --holdout-size 16
```

## Run the policy-dose and robustness extensions

These experiments extend the policy comparison and the sensitivity evidence:
the shortage-margin leverage comparison and the extended carbon-price sweep
toward the iso-leverage price, the base-shortage-cost fine grid, the
out-of-sample policy ranking on the common holdout, the scenario-weight and
correlation sensitivity, the two-dimensional demand-coefficient grid, and the
MILP size/certification summary. Run them after the core workflow:

```bash
PYTHONPATH=code python code/run_policy_extensions.py --only dose
PYTHONPATH=code python code/run_policy_extensions.py --only shortage
PYTHONPATH=code python code/run_policy_extensions.py --only ranking
PYTHONPATH=code python code/run_policy_extensions.py --only specification
PYTHONPATH=code python code/run_policy_extensions.py --only coefficients
PYTHONPATH=code python code/run_policy_extensions.py --only statistics
# or, all families in sequence:
PYTHONPATH=code python code/run_policy_extensions.py --only all
```

Outputs are written under `results/cdw_rln_v2.2.0-20260713/` with the prefix
`policy_extension_`. The `dose` and `ranking` families require the primary
results and facility solutions produced by the core workflow.

## Integrity and reproducibility

`checksums.sha256` records the SHA-256 digest of every released input and code
file. The scenario generator uses fixed disclosed seeds and a mean-preserving
correlated lognormal construction. CPLEX logs, solution tables, and a generated
manifest are written at runtime and are intentionally not versioned.

The minimal package was validated on 2026-07-14 with Python 3.13.9 and CPLEX
22.1.1. The primary baseline reached global optimality and reproduced the
frozen objective (`838207709146.2753` CNY), 80 open facilities, installed
capacity (`4665183553.222687` t/year), expected emissions, and expected
shortage. The independent objective reconstruction error was
`1.75e-13`. A two-replication scenario-stability execution check also
completed successfully.

CSV serialization can differ in the last machine-precision digit across
NumPy/pandas versions. Numerical comparisons should therefore use the model's
documented tolerances rather than requiring byte-identical generated CSV text.

The source matrix has been reduced to fields and city--candidate pairs that can
enter the current model. This removes unused rail/water attributes and
non-candidate locations without changing the road-network feasible set or any
numeric model coefficient.

## License status

No reuse license is assigned in this initial repository release. The authors
must confirm source-data redistribution rights and select explicit code and
data licenses before declaring the repository openly reusable.

## Citation

Citation metadata are provided in `CITATION.cff`. A permanent repository DOI
should be added after archiving a tagged release.
