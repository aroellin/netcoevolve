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

## Visualisation & Analysis
Install Python dependencies (example using `pip`):
```bash
pip install pandas matplotlib
```

The `scripts/` directory contains several Python tools for analysing simulation results.

### Time Series Plotting (`visualise.py`)
Plots time series of edge densities, colour fractions, and triangle densities from a simulation CSV.

```bash
python scripts/visualise.py [simulation.csv] [options]
```
Common options:
- `--out <file>`: Save plot to file (e.g., `plot.png`).
- `--split-panels`: Save each panel (colours, edges, triangles) as a separate image.
- `--triangles`: Explicitly show the triangles panel (if not shown by default).
- `--show`: Display the plot interactively.

### Unified Analysis & Animation (`analyse.py`)
A unified CLI for processing adjacency snapshots (requires `--dump_adj`). Supersedes the legacy `heatmap.py`.

#### Animation
Render an animation of the network's adjacency matrix over time.

```bash
# Animate with global degree ordering (all vertices sorted by degree)
python scripts/analyse.py animate output/simulation-<timestamp>-adj --order global-degree

# Animate with community-aware ordering (sorted within colour groups)
python scripts/analyse.py animate output/simulation-<timestamp>-adj --order degree
```
Common animation options:
- `--out <file.mp4>`: Save to video (requires ffmpeg).
- `--interval <ms>`: Frame delay.
- `--avg <k>`: Apply kxk averaging kernel (smooths the display).

#### Diagnostics
Compute spectral and rank-1 approximation metrics for the adjacency snapshots.

```bash
# Compute basic rank-1 metrics
python scripts/analyse.py diagnostics output/simulation-<timestamp>-adj --rank1

# Compute extended metrics (block-level spectral stats, correlations)
python scripts/analyse.py diagnostics output/simulation-<timestamp>-adj --extended --csv diagnostics.csv
```

### Polarisation Analysis (`polarisation.py`)
Scans a directory of simulation outputs to detect if and when the network polarised (stabilised into fixed colour fractions).

```bash
python scripts/polarisation.py --dir output --out summary.csv
```

### GUI Dispatcher (`dispatch.py`)
A Tkinter-based GUI for managing batches of simulations, useful for exploring parameter spaces (e.g., varying `n` and `eta`).

```bash
python scripts/dispatch.py
```

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
