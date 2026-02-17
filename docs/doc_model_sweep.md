# Model Sweep Configuration

## Overview

This configuration file defines the parameters for a simulation models that perform parameter sweeps. The model explores how pioneer agents influence the behavior and spatial distribution of follower agents across a grid environment over a series of discrete time steps.

---

## Configuration Sections

### `[model]`

Core simulation parameters controlling the model type, scale, and behavioral properties.

| Parameter | Value | Description |
|---|---|---|
| `class` | `pioneer_follower` | Model class identifier (optionally positional\_info) |
| `model` | `pioneer_follower` | Model name (optionally positional\_info |
| `num_models` | `6` | Number of model instances to run |
| `num_agents` | `70` | Total number of agents in the simulation |
| `num_steps` | `240` | Total number of simulation time steps |
| `vol_analysis_start` | `20` | Time step at which volumetric analysis begins |
| `vol_analysis_end` | `220` | Time step at which volumetric analysis ends |
| `num_pioneers` | `10` | Number of pioneer agents |
| `average_response_rate` | `0` | Effectively the average number of pioneers that a follower will respond to. -1: no affinity, 0: exactly 1 pioneer, 1 or more: the average number |
| `locality_mean` | `0` | Specifies that followers will have affinity for, on average, the dth closest pioneer. 0: take the closest pioneer |
| `locality_std` | `1` | Standard deviation around the locality\_mean |
| `response_seed` | `123` | Random seed for response rate assignment |
| `locality_seed` | `321` | Random seed for locality assignment |
| `pioneer_flips` | `0` | Number of pioneers 'affinity molecules' to be flipped on average |
| `pioneer_locality_mean` | `0` | Specifies that the flipped molecule should correspond to, on average, the dth nearest pioneer. 0: take the closest pioneer |
| `pioneer_locality_std` | `0` | Standard deviation around the pioneer\_locality\_mean |
| `pioneer_flips_seed` | `456` | Random seed for pioneer flip events |
| `pioneer_locality_seed` | `654` | Random seed for pioneer locality assignment |

---

### `[grid]`

Parameters defining the spatial grid on which agents are placed and move.

| Parameter | Value | Description |
|---|---|---|
| `dim_rows` | `10` | Number of rows in the grid |
| `dim_cols` | `20` | Number of columns in the grid |
| `pioneer_zone` | `3,6,0,19` | Bounding box for pioneer placement (rows 3–6, cols 0–19) |
| `pioneer_seed` | `1261` | Random seed for pioneer initial placement |
| `agent_seed` | `1165` | Random seed for follower agent initial placement |

---

### `[sweep]`

Parameter sweep configuration for running experiments across a range of values.

| Parameter | Value | Description |
|---|---|---|
| `model_class` | `${model:class}` | Model class (inherited from `[model]` section) |
| `range_pioneers` | `0, 34` | Range of pioneer counts to sweep over |
| `range_response` | `-1, 0` | Range of response rates to sweep over. -1: not pioneer-follower affinity, 0: followers have affinity for exactly 1 pioneer, 1 or more: average number of pioneers |
| `range_locality_mean` | `0, 0` | Range of locality means (fixed) |
| `range_locality_std` | `0, 0` | Range of locality standard deviations (fixed) |
| `range_pioneers_flips` | `0, 0` | Range of pioneer flips (fixed) |
| `range_pioneers_locality_mean` | `0, 0` | Range of pioneer locality means (fixed) |
| `range_pioneers_locality_std` | `0, 0` | Range of pioneer locality standard deviations (fixed) |
| `max_seed_val` | `100000000` | Maximum value for random seed generation |
| `depth` | `100` | Number of replications per parameter combination |

---

### `[analyze]`

Post-simulation analysis settings.

| Parameter | Value | Description |
|---|---|---|
| `pipe_file` | `configs/pipe.csv` | Path to the pipeline configuration file which specifies which analyses to perform|
| `max_dendrogram_distance` | `10` | Maximum distance threshold for dendrogram clustering analysis |

---

## Model Description

**pioneer_follower:** Simulates pioneer-follower affinity. Followers stochastic growth will be biased towards their corresponding pioneer. 

**positional_info:** Simulation globally encoded positions of axons. Still includes pioneers, but followers have no affinity for pioneers. Instead, followers are biased to stay at their 2D starting postions due to an external signal (e.g. a morphgenetic gradient). 

### Key Concepts

- **Pioneers** — A subset of agents (defined by `num_pioneers`) that exhibit independent or leading behavior within the grid.
- **Followers** — The remaining agents whose behavior is influenced by pioneers according to `average_response_rate` and spatial `locality` parameters.
- **Response** — The specificity of follower axons, how many pioneers a follower will respond to. 
- **Locality** — Controls how spatially contrained followers are to their corresponding pioneer. Higher `locality_std` indicates greater variation in spatial constraint.
- **Volume Window** — Analysis is restricted to steps `vol_analysis_start` through `vol_analysis_end` to exclude initialization and tail effects.

---

## Outputs
- Sweep log file that specifies the parameters for each simulation.
- Pickle files of complete simulations in the sims directory. 

---

## Notes

- The sweep covers **pioneer counts from 0 to 34** and **response rates from -1 to 0**, with **100 replications** per combination.
- All other sweep ranges are fixed, isolating the effect of pioneer count and response rate.
- Random seeds are fully specified for reproducibility across all components.

---

