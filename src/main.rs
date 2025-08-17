//! Fast coloured-edge simulation (CPU, u8 adjacency)
//!
//! - Adjacency: u8, row-major, symmetric (0/1).
//! - Four dense buckets (C0, C1, D0, D1) with swap–pop.
//! - Position table: upper-triangle u32 mapping pair -> index in its bucket.
//! - Event selection: one Exp(1), then categorical by rates (Gillespie direct).
//! - RNG: Xoshiro256++ (non-crypto), reproducible seed.
//! - CLI: ALL-CAPS parameters; --help prints usage.
//! - Progress bar shows: time, edge density, fraction of 1s.
//! - Stats: dummy placeholder for later.

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Exp1};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;
mod colnetwork;
use colnetwork::{ColNetwork, BucketKind, bidx};


/// CLI: ALL-CAPS parameters configurable.
#[derive(Parser, Debug)]
#[command(
    name = "simulation",
    about = "Fast coloured-edge simulation with u8 adjacency and four dense buckets.",
    long_about = r#"
Runs a stochastic coloured-edge simulation using four dense buckets:
  C0: concordant ABSENT,  C1: concordant PRESENT,
  D0: discordant ABSENT,  D1: discordant PRESENT.

Event selection uses one standard-exponential draw (Gillespie direct method),
then a categorical choice proportional to each bucket's rate. A progress bar updates
at the sampling rate and shows time, edge density, and the fraction of vertices with colour 1.
Subgraph statistics are not implemented yet (dummy placeholder).
"#)]
struct Cli {
    /// Number of vertices (N)
    #[arg(long = "N", default_value_t = 1000u32)]
    n: u32,

    /// RHO (rate multiplier, default 1.0)
    #[arg(long = "RHO", default_value_t = 1.0)]
    rho: f64,

    /// ETA (base for discordant-present driven colour flips)
    #[arg(long = "ETA", default_value_t = 1.0)]
    eta: f64,

    /// SAMPLE_DELTA (time between samples)
    #[arg(long = "SAMPLE_DELTA", default_value_t = 0.01)]
    sample_delta: f64,

    /// T_MAX (maximum simulation time)
    #[arg(long = "T_MAX", default_value_t = 1.0)]
    t_max: f64,

    /// SD0 (discordant ABSENT -> PRESENT multiplier)
    #[arg(long = "SD0", default_value_t = 0.7)]
    sd0: f64,

    /// SD1 (discordant PRESENT -> ABSENT multiplier)
    #[arg(long = "SD1", default_value_t = 2.0)]
    sd1: f64,

    /// SC0 (concordant ABSENT -> PRESENT multiplier)
    #[arg(long = "SC0", default_value_t = 1.5)]
    sc0: f64,

    /// SC1 (concordant PRESENT -> ABSENT multiplier)
    #[arg(long = "SC1", default_value_t = 0.3)]
    sc1: f64,

    /// RNG seed
    #[arg(long = "SEED", default_value_t = 42u64)]
    seed: u64,
    
    /// OUTPUT_FILE (CSV path; default auto timestamp)
    #[arg(long = "OUTPUT_FILE")]
    output_file: Option<String>,
}
// (stats module handles its own file I/O; no direct file imports needed here)
mod stats;
use stats::{init_stats_writer, compute_stats, flush_stats};

// (colnetwork module encapsulates bucket + adjacency logic)
#[inline] fn set_edge(adj: &mut [u8], n: usize, u: usize, v: usize, present: bool) {
    let val = if present { 1u8 } else { 0u8 };
    adj[u * n + v] = val;
    adj[v * n + u] = val;
}

/// Remove (u,v) from its bucket using `pos[]`, fix `pos[]` if swap occurred,
/// and return which bucket it was in. Requires canonical (u<v) inside.
// (legacy per-edge removal/add helpers replaced by ColNetwork methods)

/// Progress-bar message helper: sets "t=..., dens=..., frac1=..."
#[inline]
fn update_bar(pb: &ProgressBar, t_now: f64, present_edges: usize, ones_count: usize, denom_pairs: f64, n: usize) {
    let density = 2.0 * (present_edges as f64) / denom_pairs;
    let frac1 = (ones_count as f64) / (n as f64);
    pb.set_message(format!("t={:.6}  edge_density={:.6}  colour_1_fraction={:.6}", t_now, density, frac1));
}

/// Initialise colours (first half 0, second half 1) and adjacency matrix using kappa functions.
fn initialise_colours_and_adjacency<R: Rng>(colour: &mut [u8], adj: &mut [u8], n: usize, rng: &mut R) {
    debug_assert!(colour.len() == n && adj.len() == n * n);
    for i in 0..n { colour[i] = if (i as f64) / (n as f64) < 0.5 { 0 } else { 1 }; }
    #[inline] fn kappa_c(_x: f64, _y: f64) -> f64 { 0.2 }
    #[inline] fn kappa_d(_x: f64, _y: f64) -> f64 { 0.8 }
    for u in 0..(n as u32) {
        for v in (u + 1)..(n as u32) {
            let same = colour[u as usize] == colour[v as usize];
            let p = if same { kappa_c(0.0,0.0) } else { kappa_d(0.0,0.0) }; // simplified; original x,y unused
            let present = rng.random::<f64>() < p;
            set_edge(adj, n, u as usize, v as usize, present);
        }
    }
}

/// Build buckets + pos[] from colour and adjacency; returns present edge count.
// build_buckets_and_positions now handled inside ColNetwork::new

fn main() {
    let args = Cli::parse();

    let n = args.n as usize;
    assert!(n >= 2, "N must be at least 2");

    // Derived and display parameters
    let rho = args.rho;
    let t_max = args.t_max;
    let sample_delta = args.sample_delta;

    // Decide output file path (generate timestamped if not provided)
    // Ensure output directory exists and generate default path inside it
    let output_path: String = args.output_file.clone().unwrap_or_else(|| {
        let _ = std::fs::create_dir_all("output");
        format!("output/simulation-{}.csv", chrono::Local::now().format("%Y%m%d-%H%M%S"))
    });

    // ---- Print all parameters up-front (for reproducibility) ----
    println!("Parameters:");
    println!("  SEED           = {}", args.seed);
    println!("  SAMPLE_DELTA   = {}", args.sample_delta);
    println!("  T_MAX          = {}", args.t_max);
    println!("  N              = {}", args.n);
    println!("  RHO            = {}", args.rho);
    println!("  ETA            = {}", args.eta);
    println!("  SD0            = {}", args.sd0);
    println!("  SD1            = {}", args.sd1);
    println!("  SC0            = {}", args.sc0);
    println!("  SC1            = {}", args.sc1);
    println!("  OUTPUT_FILE    = {}", output_path);
    println!();

    // Progress bar (ticks at sampling times)
    let total_ticks = ((t_max / sample_delta).ceil()) as u64;
    let pb = ProgressBar::new(total_ticks.max(1));
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len}  {msg}"
        )
        .unwrap()
        .progress_chars("##-"),
    );

    // RNG (non-crypto)
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(args.seed);

    // Initialise colours + adjacency via helper, then wrap in ColNetwork
    let mut adj: Vec<u8> = vec![0; n * n];
    let mut colour: Vec<u8> = vec![0; n];
    initialise_colours_and_adjacency(&mut colour, &mut adj, n, &mut rng);
    let mut net = ColNetwork::new(adj, colour);

    // Precompute normaliser for density: N*(N-1)
    let denom_pairs = (args.n as f64) * ((args.n as f64) - 1.0);

    // Simulation loop (Gillespie direct method)
    let mut t = 0.0f64;
    let mut samples_done: u64 = 0;
    let mut sim_steps: u64 = 0;
    let mut sim_steps_v: u64 = 0; // colour flips
    let started = Instant::now();

    // Statistics writer init + header (compute_stats will append rows)
    init_stats_writer(Some(output_path.clone()), &args);
    compute_stats(0.0, net.adj(), net.colour(), n);

    while t < t_max {
        // Sampling tick? update progress + (later) stats
        let next_tick_t = (samples_done as f64) * sample_delta;
        if t >= next_tick_t {
            pb.set_position(samples_done.min(total_ticks));
            update_bar(&pb, t, net.present_edges(), net.ones_count(), denom_pairs, n);
            compute_stats(t, net.adj(), net.colour(), n);
            samples_done += 1;
        }

        // Event rates
        let r0 = if !net.bucket_is_empty(bidx(BucketKind::D1)) {
            args.eta * 2.0 * (net.bucket_len(bidx(BucketKind::D1)) as f64)
        } else { 0.0 };
        let r1 = if args.sc0 > 0.0 {
            rho * args.sc0 * (net.bucket_len(bidx(BucketKind::C0)) as f64)
        } else { 0.0 };
        let r2 = if args.sc1 > 0.0 {
            rho * args.sc1 * (net.bucket_len(bidx(BucketKind::C1)) as f64)
        } else { 0.0 };
        let r3 = if args.sd0 > 0.0 {
            rho * args.sd0 * (net.bucket_len(bidx(BucketKind::D0)) as f64)
        } else { 0.0 };
        let r4 = if args.sd1 > 0.0 {
            rho * args.sd1 * (net.bucket_len(bidx(BucketKind::D1)) as f64)
        } else { 0.0 };
        let r_tot = r0 + r1 + r2 + r3 + r4;
        if r_tot <= 0.0 { break; }

        // Δt from one Exp(1): Δt = E / R
        let e1: f64 = Exp1.sample(&mut rng);
        t += e1 / r_tot;

        // Choose event by one uniform
    let x = rng.random::<f64>() * r_tot;
        let ev = {
            let mut s = r0;
            if x < s { 0 } else { s += r1; if x < s { 1 } else { s += r2; if x < s { 2 } else { s += r3; if x < s { 3 } else { 4 }}}}
        };

    match ev {
            // (0) discordant-present edge triggers a colour flip at a random endpoint
            0 => {
                if let Some((u,v)) = net.pick_random(bidx(BucketKind::D1), &mut rng) {
                    let u0 = if rng.random::<bool>() { u } else { v }; // endpoint to flip
                    net.flip_colour(u0);
                    sim_steps_v += 1;
                }
            }
            // (1..=4) edge add/remove within concordant/discordant buckets
            1 | 2 | 3 | 4 => {
                use BucketKind::*;
                let (from_idx, to_idx) = match ev {
                    1 => (bidx(C0), bidx(C1)), // absent concordant -> present
                    2 => (bidx(C1), bidx(C0)), // present concordant -> absent
                    3 => (bidx(D0), bidx(D1)), // absent discordant -> present
                    4 => (bidx(D1), bidx(D0)), // present discordant -> absent
                    _ => unreachable!(),
                };
                let _ = net.move_edge(from_idx, to_idx, &mut rng);
            }
            _ => unreachable!(),
        }

        // Optional invariant (enable in debug builds)
        debug_assert_eq!(
            net.bucket_len(0) + net.bucket_len(1) + net.bucket_len(2) + net.bucket_len(3),
            (args.n as usize) * (args.n as usize - 1) / 2
        );

        sim_steps += 1;
    }

    // Final progress update
    pb.set_position(total_ticks);
    update_bar(&pb, t, net.present_edges(), net.ones_count(), denom_pairs, n);
    flush_stats();
    pb.finish_with_message(format!(
        "Done. t={:.6}, steps={}, flips={}, elapsed={:?}",
        t, sim_steps, sim_steps_v, started.elapsed()
    ));
}
