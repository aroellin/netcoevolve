# netcoevolve

Fast stochastic coevolving network simulation with two vertex colours and adaptive edge dynamics, implemented in Rust. Includes a Python visualisation script for plotting time series of colour fractions, edge densities, and colour-partitioned triangle densities.

## Features
- Gillespie (continuous-time) simulation with five event types (colour flip via discordant-present edge; four edge add/remove processes by concordance & presence state).
- Compact adjacency (u8) + four dense edge buckets for O(1) random edge sampling (swap–pop) and fast rate recalculation.
- Periodic sampling at fixed time delta (SAMPLE_DELTA) writing a CSV.
- Colour-partitioned statistics:
  - Colour fractions: `frac0`, `frac1`.
  - Edge partition densities: `e00` (between two colour-0 vertices), `e01` (cross), `e11` (between two colour-1 vertices). Each is a simple proportion of possible such edges (undirected, no loops).
  - Triangle colour-type densities: `3cyc000`, `3cyc001`, `3cyc011`, `3cyc111`, each equal to (count of triangles of that colour multiset) / C(N,3). Their sum is the overall triangle density (equals 1 only for the complete graph).
- Matrix-based statistics (via `faer`) for clear formulation of block adjacency and triangle counts.
- Python plotting script auto-selects latest output if path omitted.

## Build Requirements
- Rust (stable) with Cargo (edition 2021).
- Python 3.9+ (for visualisation) with packages: `pandas`, `matplotlib` (and `numpy`, installed as a dependency).

## Building
```bash
cargo build --release
```
The optimized binary will be at `target/release/netcoevolve`.

## Running the Simulation
Basic run (defaults):
```bash
./target/release/netcoevolve
```
Example with custom parameters:
```bash
./target/release/netcoevolve \
  --N 1000 \
  --RHO 1.0 \
  --ETA 1.0 \
  --SD0 0.7 --SD1 2.0 \
  --SC0 1.5 --SC1 0.3 \
  --SAMPLE_DELTA 0.01 \
  --T_MAX 5.0 \
  --SEED 42
```
If `--OUTPUT_FILE` is omitted a timestamped file like `output/simulation-YYYYMMDD-HHMMSS.csv` is created (directory auto-created).

### CLI Parameters
| Flag | Meaning | Default |
|------|---------|---------|
| `--N` | Number of vertices | 1000 |
| `--RHO` | Global edge event rate multiplier | 1.0 |
| `--ETA` | Colour-flip driver (discordant-present) | 1.0 |
| `--SD0` | Rate multiplier: discordant absent -> present | 0.7 |
| `--SD1` | Rate multiplier: discordant present -> absent | 2.0 |
| `--SC0` | Rate multiplier: concordant absent -> present | 1.5 |
| `--SC1` | Rate multiplier: concordant present -> absent | 0.3 |
| `--SAMPLE_DELTA` | Time between statistic samples | 0.01 |
| `--T_MAX` | Maximum simulation time | 1.0 |
| `--SEED` | RNG seed | 42 |
| `--OUTPUT_FILE` | CSV path (optional) | auto timestamp |

### CSV Output Schema
First line (comment) lists parameters, e.g.:
```
# N=1000 RHO=1 ETA=1 SD0=0.7 SD1=2 SC0=1.5 SC1=0.3 SAMPLE_DELTA=0.01 T_MAX=5 SEED=42 OUTPUT_FILE=output/simulation-20250816-120000.csv
```
Header line:
```
time,frac0,frac1,e00,e01,e11,3cyc000,3cyc001,3cyc011,3cyc111
```
Where:
- `frac0`, `frac1`: fractions of vertices of each colour (sum = 1).
- `e00` = (# edges between 0-colour vertices) / C(c0,2).
- `e11` = (# edges between 1-colour vertices) / C(c1,2).
- `e01` = (# cross-colour edges) / (c0 * c1).
- Triangle density columns are each divided by C(N,3). Total triangle density = sum of the four columns.

## Visualisation
Install Python dependencies (example using `pip`):
```bash
pip install pandas matplotlib
```
Run the plotter (auto-picks latest CSV in current working directory; `cd output` to pick newest there):
```bash
python scripts/visualise.py --show
```
Or specify a file in `output/` and save a PNG:
```bash
python scripts/visualise.py output/simulation-20250816-120000.csv --out run.png
```
Three panels are produced:
1. Colour fractions over time.
2. Edge densities: concordant, discordant, total.
3. Triangle colour-type densities (per C(N,3)) and their sum (overall triangle density).

## Performance Notes
- Use `--release` for meaningful speed (LTO and optimizations configured in `Cargo.toml`).
- Stats sampling cost grows mainly with building block matrices; reducing `SAMPLE_DELTA` increases overhead.

## Reproducibility
- All simulation parameters are printed to stdout and embedded in the CSV comment line.
- RNG is deterministic given `--SEED`.

---
Feel free to open an issue or PR once this is hosted on GitHub for improvements or feature requests.
