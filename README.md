# netcoevolve

Fast Rust implementation of the stochastic co-evolving network simulation introduced in the article ["Co-evolving vertex and edge dynamics in dense graphs"](https://arxiv.org/abs/2504.06493) by S. Athreya, F. Hollander, A. Röllin. Includes a Python visualisation script for plotting time series of various subgraph densities.

## Build Requirements
- Rust (stable) with Cargo (edition 2021).
- Python 3.9+ (for visualisation) with packages: `pandas`, `matplotlib` (and `numpy`, installed as a dependency).

## Building
In order to compile the project with all optimizations turned on, run:
```bash
cargo build --release
```
The optimized binary will be at `target/release/netcoevolve`.

The development version can be compiled with
```bash
cargo build
```
but it will include debug information and no optimizations.

## Running the Simulation
Basic run (defaults):
```bash
./target/release/netcoevolve
```
Example with custom parameters:
```bash
./target/release/netcoevolve \
  --n 500 --eta 1.0 --rho 2.0\
  --sd0 0.0 --sd1 1.0 --sc0 0.0 --sc1 1.0 \
  --sample_delta 0.01 --t_max 5.0 --seed 42
```

### CLI Parameters

Parameters to control simulation dynamics:

| Flag | Meaning | Default |
|------|---------|---------|
| `--n` | Number of vertices | 1000 |
| `--eta` | Colour-flip driver | 1.0 |
| `--rho` | Global edge event rate multiplier (mutually exclusive with `--beta`) | 1.0 |
| `--beta` | Convenience: sets `rho = n` and `eta = beta` | (none) |
| `--sd0` | Rate multiplier: discordant absent -> present | 0.7 |
| `--sd1` | Rate multiplier: discordant present -> absent | 2.0 |
| `--sc0` | Rate multiplier: concordant absent -> present | 1.5 |
| `--sc1` | Rate multiplier: concordant present -> absent | 0.3 |


Parameters to initialise the graph at time 0:

| Flag | Meaning | Default |
|------|---------|---------|
| `--p1` | Probability that a vertex has colour 1 | 0.5 |
| `--p00` | Edge probability between two colour-0 vertices | 0.2 |
| `--p01` | Edge probability between different-colour vertices | 0.8 |
| `--p11` | Edge probability between two colour-1 vertices | 0.2 |

Parameters for simulation control:

| Flag | Meaning | Default |
|------|---------|---------|
| `--sample_delta` | Time between statistic samples | 0.01 |
| `--t_max` | Maximum simulation time | 1.0 |
| `--seed` | RNG seed (integer) or `random` (time-based 0..65535) | 42 |
| `--output` | CSV filename | output/simulation-\<timestamp\>.csv |
| `--dump_adj` | Save adjacency matrix snapshots to a subdirectory | (disabled) |

## Python Scripts Reference

The `scripts/` directory contains tools for visualisation, analysis, and batch execution.

### `scripts/analyse.py`
Unified CLI for processing adjacency snapshots (requires `--dump_adj`).

**Usage:** `python scripts/analyse.py <subcommand> [directory] [options]`

**Subcommands:**
*   `animate`: Render an animation of the network's adjacency matrix.
*   `diagnostics`: Compute spectral and rank-1 approximation metrics.
*   `info`: Summarize dataset metadata (frame count, n, time range).
*   `correlations`: Compute vertex & edge correlations vs a reference time.

**Common Options:**
*   `--order <mode>`: Reordering mode for vertices (`none`, `global-degree`, `degree`, `in-degree`, `out-degree`).
*   `--quiet`: Suppress output.
*   `--no-progress`: Hide progress bar.

**Animation Options:**
*   `--out <file>`: Output file (`.mp4` or `.gif`).
*   `--interval <ms>`: Delay between frames.
*   `--avg <k>`: Apply k×k averaging kernel for smoothing.
*   `--group-lines`: Draw lines at community boundaries.

**Diagnostics Options:**
*   `--rank1`: Compute global rank-1 closeness metrics.
*   `--extended`: Add block-level spectral metrics and correlations.
*   `--csv <file>`: Write metrics to CSV.

### `scripts/visualise.py`
Plots time series of edge densities, colour fractions, and subgraph densities from a simulation CSV.

**Usage:** `python scripts/visualise.py [csv_file] [options]`

**Options:**
*   `--out <file>`: Save plot to file (default: `<csv_stem>-plot.png`).
*   `--show`: Display the plot interactively.
*   `--split-panels`: Save each panel as a separate image.
*   `--triangles`: Show triangle density panel.
*   `--2paths`: Show 2-path density panel.
*   `--3paths`: Show 3-path density panel.
*   `--3stars`: Show 3-star density panel.
*   `--all`: Enable all subgraph panels.
*   `--projections`: Overlay theoretical projections (dotted lines).
*   `--ratio <W:H>`: Set aspect ratio for panels.

### `scripts/polarisation.py`
Scans a directory of simulation outputs to detect polarisation events (stabilisation of colour fractions).

**Usage:** `python scripts/polarisation.py [options]`

**Options:**
*   `--dir <path>`: Directory containing simulation CSVs (default: `output`).
*   `--k <int>`: Number of final rows to check for stability (default: 10).
*   `--out <file>`: Output summary CSV (default: `polarisation-summary.csv`).

### `scripts/dispatch.py`
A Tkinter-based GUI for managing batch simulations. Useful for exploring parameter spaces (e.g., varying `n` and `eta`).

**Usage:** `python scripts/dispatch.py`

**Features:**
*   Concurrent execution of multiple simulation jobs.
*   Live progress tracking for each job.
*   Automatic file naming and organization.
*   Parameter sweep configuration (n, eta).

## Figures from the article

A simulation similar to Figure 1 of the article "Co-evolving vertex and edge dynamics in dense graphs" can be obtained by running
```bash
./target/release/netcoevolve --n 1000 --eta 1.0 --rho 1.1 --sc0 1.5 --sd0 0.7 --sc1 0.5 --sd1 2.0 --sample_delta 0.005 --t_max 3.0 --seed 61
python scripts/visualise.py --out plot.png --split-panels
```

A simulation similar to Figure 2, where polarisation occurs, can be obtained by running
```bash
./target/release/netcoevolve --n 1000 --eta 1.0 --rho 2.0 --sc0 0.0 --sd0 0.0 --sc1 1.0 --sd1 1.0 --sample_delta 0.005 --t_max 3.0 --seed 17
python scripts/visualise.py --out plot.png --split-panels
```



## Absorbing States
If the system reaches a state with zero total event rate (no colour flips or edge changes possible under the current parameters), the simulation flags an absorbing state and keeps emitting samples at the frozen configuration until `t_max`. The final progress message is annotated with `(absorbing)`.

## Performance Notes
- Use `--release` for meaningful speed (LTO and optimizations configured in `Cargo.toml`).
- Stats sampling cost grows mainly with building block matrices; increasing `SAMPLE_DELTA` reduced overhead, but also makes the statistics less granular.

## Reproducibility
- All simulation parameters (including the resolved seed, and effective `rho` when using `--beta`) are printed to stdout and embedded in the CSV comment line.
- `--seed=random` still yields a deterministic run once the printed seed value is reused.
