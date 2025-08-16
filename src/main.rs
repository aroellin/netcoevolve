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


/// Upper-triangle index for (u,v) with u < v.
#[inline]
fn tri_index(u: u32, v: u32) -> usize {
    debug_assert!(u < v);
    (u as u64 + (v as u64) * ((v as u64) - 1) / 2) as usize
}

/// Bucket kind from (present?, same_colour?).
#[repr(usize)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BucketKind { C0 = 0, C1 = 1, D0 = 2, D1 = 3 }
#[inline] fn bidx(b: BucketKind) -> usize { b as usize }

#[inline]
fn which_bucket(present: bool, same_colour: bool) -> BucketKind {
    match (present, same_colour) {
        (false, true)  => BucketKind::C0,
        (true,  true)  => BucketKind::C1,
        (false, false) => BucketKind::D0,
        (true,  false) => BucketKind::D1,
    }
}

/// Dense bucket of canonical edges with swap–pop deletion.
#[derive(Default)]
struct Bucket {
    a: Vec<(u32, u32)>, // edges with u < v
}
impl Bucket {
    #[inline] fn len(&self) -> usize { self.a.len() }
    #[inline] fn is_empty(&self) -> bool { self.a.is_empty() }

    /// Append canonical (u,v); return position where it was placed.
    #[inline] fn push(&mut self, u: u32, v: u32) -> usize {
        debug_assert!(u < v);
        let idx = self.a.len();   // next free slot
        self.a.push((u, v));      // placed at idx
        idx
    }

    /// Remove element at index i via swap–pop; return removed edge.
    #[inline] fn pop_at(&mut self, i: usize) -> (u32, u32) {
        let last = self.a.len() - 1;
        if i != last { self.a.swap(i, last); }
        self.a.pop().unwrap()
    }

    /// Remove a random element; returns (u,v,i) where i was the index removed.
    #[inline] fn remove_random<R: Rng>(&mut self, rng: &mut R) -> Option<(u32, u32, usize)> {
        if self.a.is_empty() { return None; }
    let i = rng.random_range(0..self.a.len());
        let (u, v) = self.pop_at(i);
        Some((u, v, i))
    }

    /// Pick a random edge without removing it.
    #[inline] fn pick_random<R: Rng>(&self, rng: &mut R) -> Option<(u32,u32)> {
        if self.a.is_empty() { return None; }
        let i = rng.random_range(0..self.a.len());
        Some(self.a[i])
    }
}

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

#[inline] fn is_present(adj: &[u8], n: usize, u: usize, v: usize) -> bool {
    debug_assert!(u != v);
    adj[u * n + v] != 0
}
#[inline] fn set_edge(adj: &mut [u8], n: usize, u: usize, v: usize, present: bool) {
    let val = if present { 1u8 } else { 0u8 };
    adj[u * n + v] = val;
    adj[v * n + u] = val;
}

/// Remove (u,v) from its bucket using `pos[]`, fix `pos[]` if swap occurred,
/// and return which bucket it was in. Requires canonical (u<v) inside.
fn remove_specific(
    u: u32,
    v: u32,
    n: usize,
    adj: &[u8],
    colour: &[u8],
    pos: &mut [u32],
    buckets: &mut [Bucket; 4],
) -> BucketKind {
    debug_assert!(u < v, "remove_specific requires u < v");
    let k = tri_index(u, v);
    let i = pos[k] as usize;

    let ui = u as usize;
    let vi = v as usize;
    let present = is_present(adj, n, ui, vi);
    let same = colour[ui] == colour[vi];
    let b = which_bucket(present, same);

    let bkt = &mut buckets[bidx(b)];
    let (ru, rv) = bkt.pop_at(i);
    debug_assert_eq!((ru, rv), (u, v));

    // If we swapped in a different edge at index i, update its pos[] entry.
    if i < bkt.len() {
        let (su, sv) = bkt.a[i];
        let kk = tri_index(su, sv);
        pos[kk] = i as u32;
    }
    b
}

/// Append (u,v) to bucket b and record its position in `pos[]`. Requires canonical (u<v) inside.
fn add_to_bucket(
    u: u32,
    v: u32,
    b: BucketKind,
    pos: &mut [u32],
    buckets: &mut [Bucket; 4],
) {
    debug_assert!(u < v, "add_to_bucket requires u < v");
    let idx = buckets[bidx(b)].push(u, v);
    let k = tri_index(u, v);
    pos[k] = idx as u32;
}

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
fn build_buckets_and_positions(colour: &[u8], adj: &[u8], n_u32: u32, pos: &mut [u32], buckets: &mut [Bucket; 4]) -> usize {
    let n = n_u32 as usize;
    let mut present_edges = 0usize;
    for u in 0..n_u32 {
        for v in (u + 1)..n_u32 {
            let ui = u as usize; let vi = v as usize;
            let present = is_present(adj, n, ui, vi);
            let same = colour[ui] == colour[vi];
            if present { present_edges += 1; }
            let b = which_bucket(present, same);
            let idx = buckets[bidx(b)].push(u, v);
            let k = tri_index(u, v); pos[k] = idx as u32;
        }
    }
    present_edges
}

fn main() {
    let args = Cli::parse();

    let n = args.n as usize;
    assert!(n >= 2, "N must be at least 2");

    // Derived and display parameters
    let rho = args.rho;
    let t_max = args.t_max;
    let sample_delta = args.sample_delta;

    // Decide output file path (generate timestamped if not provided)
    let output_path: String = args.output_file.clone().unwrap_or_else(|| {
        format!("simulation-{}.csv", chrono::Local::now().format("%Y%m%d-%H%M%S"))
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

    // Storage
    let mut adj: Vec<u8> = vec![0; n * n]; // symmetric 0/1
    let mut colour: Vec<u8> = vec![0; n];  // 0/1
    // Position table: one entry per unordered pair (u<v): N*(N-1)/2
    let mut pos: Vec<u32> = vec![0; (args.n as usize * (args.n as usize - 1)) / 2];
    let mut buckets: [Bucket; 4] = [
        Bucket::default(), Bucket::default(), Bucket::default(), Bucket::default()
    ];

    // Initialise colours + adjacency via helper
    initialise_colours_and_adjacency(&mut colour, &mut adj, n, &mut rng);
    // Track fraction of 1s efficiently
    let mut ones_count: usize = colour.iter().map(|&c| c as usize).sum();
    // Build buckets + pos[] and obtain present edge count
    let mut present_edges: usize = build_buckets_and_positions(&colour, &adj, args.n, &mut pos, &mut buckets);

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
    compute_stats(0.0, &adj, &colour, n);

    while t < t_max {
        // Sampling tick? update progress + (later) stats
        let next_tick_t = (samples_done as f64) * sample_delta;
        if t >= next_tick_t {
            pb.set_position(samples_done.min(total_ticks));
            update_bar(&pb, t, present_edges, ones_count, denom_pairs, n);
            compute_stats(t, &adj, &colour, n);
            samples_done += 1;
        }

        // Event rates
        let r0 = if !buckets[bidx(BucketKind::D1)].is_empty() { args.eta * 2.0 * (buckets[bidx(BucketKind::D1)].len() as f64) } else { 0.0 };
        let r1 = if args.sc0 > 0.0 { rho * args.sc0 * (buckets[bidx(BucketKind::C0)].len() as f64) } else { 0.0 };
        let r2 = if args.sc1 > 0.0 { rho * args.sc1 * (buckets[bidx(BucketKind::C1)].len() as f64) } else { 0.0 };
        let r3 = if args.sd0 > 0.0 { rho * args.sd0 * (buckets[bidx(BucketKind::D0)].len() as f64) } else { 0.0 };
        let r4 = if args.sd1 > 0.0 { rho * args.sd1 * (buckets[bidx(BucketKind::D1)].len() as f64) } else { 0.0 };
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
                // Sample a discordant-present edge without structural mutation
                if let Some((u, v)) = buckets[bidx(BucketKind::D1)].pick_random(&mut rng) {
                    // Flip one endpoint and move all incident edges (u0, w)
                    let u0 = if rng.random::<bool>() { u } else { v };
                    let u0i = u0 as usize;
                    let old_col = colour[u0i];

                    // Two loops to avoid branch on w==u0 and to hand over canonical (u<v) ordering directly.
                    // Loop 1: w in [0, u0)  => canonical pair is (w, u0)
                    for w in 0..u0 {
                        let wi = w as usize;
                        let _old = remove_specific(w, u0, n, &adj, &colour, &mut pos, &mut buckets);
                        let present = is_present(&adj, n, wi, u0i); // adjacency symmetric
                        let same_after = (1 - old_col) == colour[wi];
                        let b_new = which_bucket(present, same_after);
                        add_to_bucket(w, u0, b_new, &mut pos, &mut buckets);
                    }
                    // Loop 2: w in (u0, N)  => canonical pair is (u0, w)
                    for w in (u0+1)..args.n {
                        let wi = w as usize;
                        let _old = remove_specific(u0, w, n, &adj, &colour, &mut pos, &mut buckets);
                        let present = is_present(&adj, n, u0i, wi);
                        let same_after = (1 - old_col) == colour[wi];
                        let b_new = which_bucket(present, same_after);
                        add_to_bucket(u0, w, b_new, &mut pos, &mut buckets);
                    }
                    colour[u0i] ^= 1;
                    // Update ones_count quickly
                    if old_col == 0 { ones_count += 1; } else { ones_count -= 1; }
                    sim_steps_v += 1;
                }
            }
            // (1..=4) edge add/remove within concordant/discordant buckets
            1 | 2 | 3 | 4 => {
                use BucketKind::*;
                let (from_kind, to_kind, add_edge) = match ev {
                    1 => (C0, C1, true),  // absent concordant -> present
                    2 => (C1, C0, false), // present concordant -> absent
                    3 => (D0, D1, true),  // absent discordant -> present
                    4 => (D1, D0, false), // present discordant -> absent
                    _ => unreachable!(),
                };
                let sampled = {
                    let bkt = &mut buckets[bidx(from_kind)];
                    bkt.remove_random(&mut rng).map(|(u, v, i_removed)| {
                        if let Some(&(su, sv)) = bkt.a.get(i_removed) {
                            pos[tri_index(su, sv)] = i_removed as u32;
                        }
                        (u, v)
                    })
                };
                if let Some((u, v)) = sampled {
                    set_edge(&mut adj, n, u as usize, v as usize, add_edge);
                    add_to_bucket(u, v, to_kind, &mut pos, &mut buckets);
                    if add_edge { present_edges += 1; } else { present_edges -= 1; }
                }
            }
            _ => unreachable!(),
        }

        // Optional invariant (enable in debug builds)
        debug_assert_eq!(
            buckets[bidx(BucketKind::C0)].len()
            + buckets[bidx(BucketKind::C1)].len()
            + buckets[bidx(BucketKind::D0)].len()
            + buckets[bidx(BucketKind::D1)].len(),
            (args.n as usize) * (args.n as usize - 1) / 2
        );

        sim_steps += 1;
    }

    // Final progress update
    pb.set_position(total_ticks);
    update_bar(&pb, t, present_edges, ones_count, denom_pairs, n);
    flush_stats();
    pb.finish_with_message(format!(
        "Done. t={:.6}, steps={}, flips={}, elapsed={:?}",
        t, sim_steps, sim_steps_v, started.elapsed()
    ));
}
