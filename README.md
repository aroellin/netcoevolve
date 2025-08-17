# netcoevolve

Fast Rust implementation of the stochastic co-evolving network simulation introduced in the article ["Co-evolving vertex and edge dynamics in dense graphs"](https://arxiv.org/abs/2504.06493) by S. Athreya, F. Hollander, A. Röllin. Includes a Python visualisation script for plotting time series of various subgraph densities. 

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
  --n 500 --eta 1.0 --rho 2.0\
  --sd0 0.0 --sd1 1.0 --sc0 0.0 --sc1 1.0 \
  --sample_delta 0.01 --t_max 5.0 --seed 42
```

### CLI Parameters
| Flag | Meaning | Default |
|------|---------|---------|
| `--n` | Number of vertices | 1000 |
| `--eta` | Colour-flip driver | 1.0 |
| `--rho` | Global edge event rate multiplier | 1.0 |
| `--sd0` | Rate multiplier: discordant absent -> present | 0.7 |
| `--sd1` | Rate multiplier: discordant present -> absent | 2.0 |
| `--sc0` | Rate multiplier: concordant absent -> present | 1.5 |
| `--sc1` | Rate multiplier: concordant present -> absent | 0.3 |
| `--sample_delta` | Time between statistic samples | 0.01 |
| `--t_max` | Maximum simulation time | 1.0 |
| `--seed` | RNG seed | 42 |
| `--output_file` | CSV filename | output/simulation-\<timestamp\>.csv |


## Visualisation
Install Python dependencies (example using `pip`):
```bash
pip install pandas matplotlib
```
Run the plotter with no arguments to visualise the most recent CSV in `output/` and save a plot as `<csv-filename>-plot.png` in the same directory as the CSV:
```bash
python scripts/visualise.py
```

Currently, three panels are produced: Colour fractions, edge densities, triangle densities.
 
## Performance Notes
- Use `--release` for meaningful speed (LTO and optimizations configured in `Cargo.toml`).
- Stats sampling cost grows mainly with building block matrices; increasing `SAMPLE_DELTA` reduced overhead, but also makes the statistics less granular.

## Reproducibility
- All simulation parameters are printed to stdout and embedded in the CSV comment line.
- RNG is deterministic given `--SEED`.

---
Feel free to open an issue or PR once this is hosted on GitHub for improvements or feature requests.
