# Data dictionary

All input files are UTF-8 CSV files. Mass is measured in tonnes, distance in
kilometres, monetary values in Chinese yuan (CNY), and emissions in tonnes of
CO2 equivalent unless a field states otherwise.

## `data/source/city_master.csv`

| Field | Meaning |
| --- | --- |
| `city_code_final` | Stable city-level administrative code used for joins |
| `city_name` | City name |
| `province` | Province-level administrative unit |
| `region` | Macro-region: `east`, `central`, or `west` |
| `landfill_cost_yuan_per_t` | Regional landfill-cost planning input, CNY/t |

The table contains 371 source units. The code retains the 333 units with a
complete three-family demand trajectory for every year from 2025 to 2035 and
reports the remaining 38 without imputation.

## `data/source/cdw_demand_projections_2023_2035.csv`

| Field | Meaning |
| --- | --- |
| `city_code_final` | City join key |
| `year` | Projection year, 2023--2035 |
| `scenario` | Demand family: `baseline`, `strong_management`, or `high_growth` |
| `cdw_pred_final` | Projected CDW demand, t/year |

These are modeled planning trajectories, not treatment-plant measurements.
They supply the annual phased model and the mean levels used by the stochastic
scenario generator.

## `data/source/candidate_site_source.csv`

| Field | Meaning |
| --- | --- |
| `site_id` | Candidate-site identifier |
| `facility_city` | City label associated with the candidate point |
| `province` | Province associated with the candidate point |
| `region` | Macro-region |
| `capacity_min_ton` | Minimum installed annual capacity if opened, t/year |
| `capacity_max_ton` | Maximum installed annual capacity, t/year |
| `fixed_cost_yuan` | Fixed opening-cost proxy, CNY |
| `capacity_cost_yuan_per_ton` | Capacity investment-cost proxy, CNY per t/year |
| `processing_cost_yuan_per_t` | Processing-cost planning input, CNY/t |

The 114 rows are planning locations. Capacity and capital fields are
deterministic proxies rather than observed plant accounts.

## `data/source/regional_operating_parameters.csv`

| Field | Meaning |
| --- | --- |
| `province` | Province used to map regional parameters |
| `region` | Macro-region |
| `transport_cost_yuan_per_tkm` | Road transport cost, CNY/(t km) |
| `emission_factor_kgco2_per_t` | Processing emissions, kg CO2e/t |

Values repeat within the three macro-regions and should not be interpreted as
province-specific observations.

## `data/source/transport_cost_matrix.csv`

| Field | Meaning |
| --- | --- |
| `city_code` | Origin city code |
| `site_id` | Candidate-site identifier |
| `highway_distance_km` | Road distance used by the model, km |

The table contains all 371-by-114 source city--candidate combinations. The
model maps the regional CNY/(t km) rate to each origin, calculates road cost,
and retains the 12 lowest-cost candidates for each of the 333 modeled cities.
Rail and water attributes are intentionally excluded because feasible terminal
access and mode capacities are not established.

## Generated data

`run_rigorous_revision.py` creates the following under the versioned
`data/derived/` directory:

- complete-case and excluded-city tables;
- candidate planning parameters with provenance labels;
- retained road edges and calculated transport emissions;
- the primary 16-scenario demand ensemble and probabilities.

The stochastic-value and stability scripts add their expected-demand and
holdout scenario files to the same generated directory.
