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
    /// Number of vertices (n)
    #[arg(long = "n", default_value_t = 1000u32)]
    n: u32,

    /// rho (rate multiplier, default 1.0 unless --beta used)
    #[arg(long = "rho")]
    rho: Option<f64>,

    /// beta (sets rho = n and eta = beta)
    #[arg(long = "beta")]
    beta: Option<f64>,

    /// eta (base for discordant-present driven colour flips)
    #[arg(long = "eta", default_value_t = 1.0)]
    eta: f64,

    /// sample_delta (time between samples)
    #[arg(long = "sample_delta", default_value_t = 0.01)]
    sample_delta: f64,

    /// t_max (maximum simulation time)
    #[arg(long = "t_max", default_value_t = 1.0)]
    t_max: f64,

    /// sd0 (discordant ABSENT -> PRESENT multiplier)
    #[arg(long = "sd0", default_value_t = 0.7)]
    sd0: f64,

    /// sd1 (discordant PRESENT -> ABSENT multiplier)
    #[arg(long = "sd1", default_value_t = 2.0)]
    sd1: f64,

    /// sc0 (concordant ABSENT -> PRESENT multiplier)
    #[arg(long = "sc0", default_value_t = 1.5)]
    sc0: f64,

    /// sc1 (concordant PRESENT -> ABSENT multiplier)
    #[arg(long = "sc1", default_value_t = 0.3)]
    sc1: f64,

    /// RNG seed; use --seed=random for time-based seed
    #[arg(long = "seed", default_value = "42")]
    seed: String,
    
    /// p1 (initial probability a vertex has colour 1)
    #[arg(long = "p1", default_value_t = 0.5)]
    p1: f64,

    /// p00 (initial edge probability between two colour-0 vertices)
    #[arg(long = "p00", default_value_t = 0.5)]
    p00: f64,

    /// p01 (initial edge probability between different-colour vertices)
    #[arg(long = "p01", default_value_t = 0.5)]
    p01: f64,

    /// p11 (initial edge probability between two colour-1 vertices)
    #[arg(long = "p11", default_value_t = 0.5)]
    p11: f64,

    /// output_file (CSV path; default auto timestamp)
    #[arg(long = "output")]
    output_file: Option<String>,

    /// dump_adj (if set, dump adjacency matrix each sample; reordered: all 0s then 1s, degree-ascending within group)
    #[arg(long = "dump_adj", default_value_t = false)]
    dump_adj: bool,
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

/// Progress-bar message helper: sets "t=..., dens=..., col1=..."
#[inline]
fn update_bar(pb: &ProgressBar, t_now: f64, present_edges: usize, ones_count: usize, denom_pairs: f64, n: usize) {
    let density = 2.0 * (present_edges as f64) / denom_pairs;
    let col1 = (ones_count as f64) / (n as f64);
    pb.set_message(format!("t={:.6}  edge_density={:.6}  colour_1_fraction={:.6}", t_now, density, col1));
}

/// Initialise colours iid Bernoulli(p1) and adjacency with probabilities p00/p01/p11.
fn initialise_colours_and_adjacency<R: Rng>(colour: &mut [u8], adj: &mut [u8], n: usize, rng: &mut R,
    p1: f64, p00: f64, p01: f64, p11: f64) {
    debug_assert!(colour.len() == n && adj.len() == n * n);
    for i in 0..n { colour[i] = if rng.random::<f64>() < p1 { 1 } else { 0 }; }
    for u in 0..n {
        for v in (u + 1)..n {
            let cu = colour[u]; let cv = colour[v];
            let p = match (cu, cv) { (0,0) => p00, (1,1) => p11, _ => p01 };
            let present = rng.random::<f64>() < p;
            set_edge(adj, n, u, v, present);
        }
    }
}

/// Build buckets + pos[] from colour and adjacency; returns present edge count.
// build_buckets_and_positions now handled inside ColNetwork::new

fn main() {
    let mut args = Cli::parse();

    let n = args.n as usize;
    assert!(n >= 2, "N must be at least 2");
    
    // Keep originals for display if beta provided
    let mut rho = args.rho.unwrap_or(1.0);
    if let Some(beta) = args.beta {
        assert!(beta > 0.0, "beta must be > 0");
        assert!(args.rho.is_none(), "Cannot specify both --rho and --beta");
        // New semantics: when beta is set, set eta = beta and rho = n
        args.eta = beta;             // eta equals beta
        rho = n as f64;              // rho equals n
        args.rho = Some(rho);        // reflect effective rho in args for stats
    }

    // Derived and display parameters (rho possibly overridden)
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
    // Determine effective seed (support --seed=random)
    let seed_arg = args.seed.clone();
    let (effective_seed, seed_random) = if seed_arg.eq_ignore_ascii_case("random") {
        // Generate a small reproducible seed (0..=65535) from current time.
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let raw = now.as_nanos();
        // Simple bit fold then mask to 16 bits for manageability
        let folded = raw ^ (raw >> 17) ^ (raw >> 33);
        (((folded as u64) & 0xFFFF), true)
    } else {
        (seed_arg.parse::<u64>().expect("seed must be an integer or 'random'"), false)
    };
    println!("  seed           = {}{}", effective_seed, if seed_random { " (random)" } else { "" });
    println!("  sample_delta   = {}", args.sample_delta);
    println!("  t_max          = {}", args.t_max);
    println!("  n              = {}", args.n);
    if args.beta.is_some() {
        println!("  rho            = {} = n", rho);
        println!("  eta            = {} = beta", args.eta);
    } else {
        println!("  rho            = {}", rho);
        println!("  eta            = {}", args.eta);
    }
    println!("  sd0            = {}", args.sd0);
    println!("  sd1            = {}", args.sd1);
    println!("  sc0            = {}", args.sc0);
    println!("  sc1            = {}", args.sc1);
    println!("  p1             = {}", args.p1);
    println!("  p00            = {}", args.p00);
    println!("  p01            = {}", args.p01);
    println!("  p11            = {}", args.p11);
    println!("  output         = {}", output_path);
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
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(effective_seed);

    // Initialise colours + adjacency via helper, then wrap in ColNetwork
    let mut adj: Vec<u8> = vec![0; n * n];
    let mut colour: Vec<u8> = vec![0; n];
    initialise_colours_and_adjacency(&mut colour, &mut adj, n, &mut rng,
        args.p1, args.p00, args.p01, args.p11);
    let mut net = ColNetwork::new(adj, colour);

    // Precompute normaliser for density: N*(N-1)
    let denom_pairs = (args.n as f64) * ((args.n as f64) - 1.0);

    // Simulation loop (Gillespie direct method)
    let mut t = 0.0f64;
    let mut samples_done: u64 = 0;
    let mut sim_steps: u64 = 0;
    let mut sim_steps_v: u64 = 0; // colour flips
    let mut absorbing_state = false;
    let started = Instant::now();

    // Statistics writer init + header (compute_stats will append rows)
    init_stats_writer(Some(output_path.clone()), &args, effective_seed, seed_random, args.dump_adj);
    compute_stats(0.0, net.adj(), net.colour(), net.last_flip_times(), n);

    while t < t_max {
        // Sampling tick? update progress + (later) stats
        let next_tick_t = (samples_done as f64) * sample_delta;
        if t >= next_tick_t {
            pb.set_position(samples_done.min(total_ticks));
            update_bar(&pb, t, net.present_edges(), net.ones_count(), denom_pairs, n);
            compute_stats(t, net.adj(), net.colour(), net.last_flip_times(), n);
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
        if r_tot <= 0.0 {
            absorbing_state = true;
            // Fill remaining sampling points with constant state
            while samples_done <= total_ticks {
                let tick_t = (samples_done as f64) * sample_delta;
                if tick_t > t_max + 1e-12 { break; }
                pb.set_position(samples_done.min(total_ticks));
                update_bar(&pb, t, net.present_edges(), net.ones_count(), denom_pairs, n);
                compute_stats(tick_t.min(t_max), net.adj(), net.colour(), net.last_flip_times(), n);
                samples_done += 1;
            }
            t = t_max; // jump to end
            break;
        }

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
                    net.flip_colour(u0, t);
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

    // absorbing_state continuation handled inside loop when detected

    // Final progress update
    pb.set_position(total_ticks);
    update_bar(&pb, t, net.present_edges(), net.ones_count(), denom_pairs, n);
    flush_stats();
    if absorbing_state {
        pb.finish_with_message(format!(
            "Done (absorbing). t={:.6}, steps={}, flips={}, elapsed={:?}",
            t, sim_steps, sim_steps_v, started.elapsed()
        ));
    } else {
        pb.finish_with_message(format!(
            "Done. t={:.6}, steps={}, flips={}, elapsed={:?}",
            t, sim_steps, sim_steps_v, started.elapsed()
        ));
    }
}
