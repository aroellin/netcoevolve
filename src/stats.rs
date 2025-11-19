//! Statistics module: computing edge densities and triangle colour pattern frequencies.
//!
//! Exposes:
//!   - init_stats_writer(path_opt, &Cli)
//!   - compute_stats(t, adj, colour, n)
//!   - flush_stats()
//!
//! Internally maintains a thread-local CSV writer so calls stay lightweight.

use std::fs::{File, create_dir_all};
use std::io::{Write, BufWriter};
use chrono::Local;
use crate::Cli; // ensure Cli is in scope after removal of earlier accidental code

thread_local! {
    static STATS_WRITER: std::cell::RefCell<Option<CsvStatsWriter>> = std::cell::RefCell::new(None);
}

pub fn init_stats_writer(path_opt: Option<String>, args: &Cli, effective_seed: u64, seed_random: bool, dump_adj: bool) {
    STATS_WRITER.with(|slot| {
    *slot.borrow_mut() = Some(CsvStatsWriter::new(path_opt, args, effective_seed, seed_random, dump_adj).expect("create output CSV"));
    });
}

/// Compute stats and append CSV row.
#[inline]
pub fn compute_stats(t: f64, adj: &[u8], colour: &[u8], last_flip: &[f64], n: usize) {
    debug_assert!(adj.len() == n * n && colour.len() == n);
    // Build colour index lists
    let mut idx0 = Vec::new();
    let mut idx1 = Vec::new();
    for (i, &c) in colour.iter().enumerate() { if c == 0 { idx0.push(i); } else { idx1.push(i); } }
    let c0 = idx0.len();
    let c1 = idx1.len();
    let col0 = c0 as f64 / n as f64;
    let col1 = c1 as f64 / n as f64;

    // f32 matrices for sub-blocks
    let mut a00 = faer::Mat::<f32>::zeros(c0, c0);
    let mut a11 = faer::Mat::<f32>::zeros(c1, c1);
    let mut a01 = faer::Mat::<f32>::zeros(c0, c1);

    for ii in 0..c0 {
        a00[(ii,ii)] = 0.0;
        for jj in (ii+1)..c0 {
            let u = idx0[ii]; let v = idx0[jj];
            if adj[u*n + v] != 0 { a00[(ii,jj)] = 1.0; a00[(jj,ii)] = 1.0; }
        }
    }
    for ii in 0..c1 {
        a11[(ii,ii)] = 0.0;
        for jj in (ii+1)..c1 {
            let u = idx1[ii]; let v = idx1[jj];
            if adj[u*n + v] != 0 { a11[(ii,jj)] = 1.0; a11[(jj,ii)] = 1.0; }
        }
    }
    for ii in 0..c0 {
        let u = idx0[ii];
        for jj in 0..c1 { let v = idx1[jj]; if adj[u*n + v] != 0 { a01[(ii,jj)] = 1.0; } }
    }

    let sum_sym  = |m: &faer::Mat<f32>| -> f64 { let mut s=0.0; for i in 0..m.nrows() { for j in 0..m.ncols() { s += m[(i,j)] as f64; } } s };
    let sum_rect = |m: &faer::Mat<f32>| -> f64 { let mut s=0.0; for i in 0..m.nrows() { for j in 0..m.ncols() { s += m[(i,j)] as f64; } } s };
    let sum00 = sum_sym(&a00); // counts undirected edges twice
    let sum11 = sum_sym(&a11);
    let sum01 = sum_rect(&a01); // cross once
    // Normalise by total number of vertex pairs (n choose 2)
    let n_pairs = (n as f64) * (n as f64 - 1.0) / 2.0;
    let e00 = if c0 > 1 { (0.5 * sum00) / n_pairs } else { 0.0 };
    let e11 = if c1 > 1 { (0.5 * sum11) / n_pairs } else { 0.0 };
    let e01 = if c0 > 0 && c1 > 0 { (sum01) / n_pairs } else { 0.0 };
    // Non-edge densities within/between colour classes, also normalised by n choose 2
    let ne00 = if c0 > 1 {
        (0.5 * (c0 as f64 * (c0 as f64 - 1.0)) - 0.5 * sum00) / n_pairs
    } else { 0.0 };
    let ne11 = if c1 > 1 {
        (0.5 * (c1 as f64 * (c1 as f64 - 1.0)) - 0.5 * sum11) / n_pairs
    } else { 0.0 };
    let ne01 = if c0 > 0 && c1 > 0 { ((c0 as f64 * c1 as f64) - sum01) / n_pairs } else { 0.0 };

    fn tri_count_f(a: &faer::Mat<f32>) -> f64 {
        let n=a.nrows(); if n<3 { return 0.0; }
        let a2 = a * a; let a3 = &a2 * a; let mut tr=0.0; for i in 0..n { tr += a3[(i,i)] as f64; } tr / 6.0
    }
    let cyc000_count = tri_count_f(&a00);
    let cyc111_count = tri_count_f(&a11);

    let a01_t = a01.as_ref().transpose();
    let c_mat = &a01 * &a01_t; // c0 x c0
    let d_mat = &a01_t * &a01; // c1 x c1
    let mut cyc001_count = 0.0f64;
    for i in 0..c0 { for j in (i+1)..c0 { if a00[(i,j)] != 0.0 { cyc001_count += c_mat[(i,j)] as f64; } } }
    let mut cyc011_count = 0.0f64;
    for p in 0..c1 { for q in (p+1)..c1 { if a11[(p,q)] != 0.0 { cyc011_count += d_mat[(p,q)] as f64; } } }

    // Homomorphism densities migration:
    //  - Triangles: hom(K3_col,G) = 6 * (#colour-pattern triangles) / n^3.
    //  - 2-paths xyz: sum_{middle vertex colour=y} d_x(m) * d_z(m) / n^3.
    //  - 3-paths s0 s1 s2 s3: sum_{(u,v) edge with colours (s1,s2)} d_{s0}(u)*d_{s3}(v) over both orientations / n^4.
    //  - 3-stars centre colour c with leaf multiset (k ones): sum d0^{3-k} d1^{k} / n^4 (leaves treated as labelled in hom definition).

    let n_f = n as f64;
    // Degrees by colour for each vertex
    let mut deg00: Vec<f64> = vec![0.0; c0];
    let mut deg01: Vec<f64> = vec![0.0; c0];
    for i in 0..c0 { for j in 0..c0 { deg00[i] += a00[(i,j)] as f64; } for j in 0..c1 { deg01[i] += a01[(i,j)] as f64; } }
    let mut deg10: Vec<f64> = vec![0.0; c1];
    let mut deg11: Vec<f64> = vec![0.0; c1];
    for j in 0..c1 { for k in 0..c1 { deg11[j] += a11[(j,k)] as f64; } for i in 0..c0 { deg10[j] += a01[(i,j)] as f64; } }

    // Triangle hom densities (replace previous injective densities)
    let cyc000 = if n>=3 { (6.0 * cyc000_count) / (n_f.powi(3)) } else { 0.0 };
    let cyc111 = if n>=3 { (6.0 * cyc111_count) / (n_f.powi(3)) } else { 0.0 };
    let cyc001 = if n>=3 { (6.0 * cyc001_count) / (n_f.powi(3)) } else { 0.0 };
    let cyc011 = if n>=3 { (6.0 * cyc011_count) / (n_f.powi(3)) } else { 0.0 };

    // 2-path hom densities with symmetry factors so coloured sum matches uncoloured Σ deg^2 / n^3.
    // Mixed patterns appear once here; multiply by 2 to account for reversed colour sequence not listed separately.
    let mut h2_000=0.0; let mut h2_001=0.0; let mut h2_101=0.0;
    for i in 0..c0 { let d0=deg00[i]; let d1=deg01[i]; h2_000 += d0*d0; h2_001 += 2.0*d0*d1; h2_101 += d1*d1; }
    let mut h2_010=0.0; let mut h2_011=0.0; let mut h2_111=0.0;
    for j in 0..c1 { let d0=deg10[j]; let d1=deg11[j]; h2_010 += d0*d0; h2_011 += 2.0*d0*d1; h2_111 += d1*d1; }
    let denom_h2 = n_f.powi(3);
    let p000 = if denom_h2>0.0 { h2_000/denom_h2 } else { 0.0 };
    let p001 = if denom_h2>0.0 { h2_001/denom_h2 } else { 0.0 };
    let p010 = if denom_h2>0.0 { h2_010/denom_h2 } else { 0.0 };
    let p011 = if denom_h2>0.0 { h2_011/denom_h2 } else { 0.0 };
    let p101 = if denom_h2>0.0 { h2_101/denom_h2 } else { 0.0 };
    let p111 = if denom_h2>0.0 { h2_111/denom_h2 } else { 0.0 };

    // 3-path hom counts aggregated into canonical (sequence or its reverse) with symmetry factors implicit by pairing.
    // Canonical representative = lexicographically min(sequence, reversed_sequence).
    // Canonical set matches existing columns: 0000,0001,0010,0011,0101,0110,0111,1001,1011,1111
    let mut p3 = [0.0f64;10];
    fn canonical_index(mut s: (u8,u8,u8,u8)) -> Option<usize> {
        let r = (s.3,s.2,s.1,s.0);
        if r < s { s = r; }
        match s { (0,0,0,0)=>Some(0),(0,0,0,1)=>Some(1),(0,0,1,0)=>Some(2),(0,0,1,1)=>Some(3),
                  (0,1,0,1)=>Some(4),(0,1,1,0)=>Some(5),(0,1,1,1)=>Some(6),(1,0,0,1)=>Some(7),
                  (1,0,1,1)=>Some(8),(1,1,1,1)=>Some(9), _=>None }
    }
    // helper closures to get d0/d1 for colour+index
    let d0_zero = |i:usize| deg00[i];
    let d1_zero = |i:usize| deg01[i];
    let d0_one  = |j:usize| deg10[j];
    let d1_one  = |j:usize| deg11[j];
    // zero-zero edges
    for i in 0..c0 { for j in (i+1)..c0 { if a00[(i,j)]==0.0 { continue; }
        for &(s0, s3) in &[(0u8,0u8),(0,1),(1,0),(1,1)] {
            // direction i->j and j->i both contribute; canonicalization merges reverses.
            let left = if s0==0 { d0_zero(i) } else { d1_zero(i) };
            let right= if s3==0 { d0_zero(j) } else { d1_zero(j) };
            if let Some(k) = canonical_index((s0,0,0,s3)) { p3[k] += left*right; }
            let left2 = if s0==0 { d0_zero(j) } else { d1_zero(j) };
            let right2= if s3==0 { d0_zero(i) } else { d1_zero(i) };
            if let Some(k) = canonical_index((s0,0,0,s3)) { p3[k] += left2*right2; }
        }
    }}
    // one-one edges
    for i in 0..c1 { for j in (i+1)..c1 { if a11[(i,j)]==0.0 { continue; }
        for &(s0, s3) in &[(0u8,0u8),(0,1),(1,0),(1,1)] {
            let left = if s0==0 { d0_one(i) } else { d1_one(i) };
            let right= if s3==0 { d0_one(j) } else { d1_one(j) };
            if let Some(k)=canonical_index((s0,1,1,s3)) { p3[k] += left*right; }
            let left2 = if s0==0 { d0_one(j) } else { d1_one(j) };
            let right2= if s3==0 { d0_one(i) } else { d1_one(i) };
            if let Some(k)=canonical_index((s0,1,1,s3)) { p3[k] += left2*right2; }
        }
    }}
    // zero-one edges (i zero, j one)
    for i in 0..c0 { for j in 0..c1 { if a01[(i,j)]==0.0 { continue; }
        for &(s0,s3) in &[(0u8,0u8),(0,1),(1,0),(1,1)] {
            let left = if s0==0 { d0_zero(i) } else { d1_zero(i) };
            let right= if s3==0 { d0_one(j) } else { d1_one(j) };
            if let Some(k)=canonical_index((s0,0,1,s3)) { p3[k]+= left*right; }
            let left2 = if s0==0 { d0_one(j) } else { d1_one(j) };
            let right2= if s3==0 { d0_zero(i) } else { d1_zero(i) };
            if let Some(k)=canonical_index((s0,1,0,s3)) { p3[k]+= left2*right2; }
        }
    }}
    let denom_3p = n_f.powi(4);
    let p3_0000 = if denom_3p>0.0 { p3[0]/denom_3p } else { 0.0 };
    let p3_0001 = if denom_3p>0.0 { p3[1]/denom_3p } else { 0.0 };
    let p3_0010 = if denom_3p>0.0 { p3[2]/denom_3p } else { 0.0 };
    let p3_0011 = if denom_3p>0.0 { p3[3]/denom_3p } else { 0.0 };
    let p3_0101 = if denom_3p>0.0 { p3[4]/denom_3p } else { 0.0 };
    let p3_0110 = if denom_3p>0.0 { p3[5]/denom_3p } else { 0.0 };
    let p3_0111 = if denom_3p>0.0 { p3[6]/denom_3p } else { 0.0 };
    let p3_1001 = if denom_3p>0.0 { p3[7]/denom_3p } else { 0.0 };
    let p3_1011 = if denom_3p>0.0 { p3[8]/denom_3p } else { 0.0 };
    let p3_1111 = if denom_3p>0.0 { p3[9]/denom_3p } else { 0.0 };

    // 3-star hom densities with symmetry factors (multinomial): coefficients 1,3,3,1 for leaves.
    let mut s0_000=0.0; let mut s0_001=0.0; let mut s0_011=0.0; let mut s0_111=0.0;
    for i in 0..c0 { let d0=deg00[i]; let d1=deg01[i]; s0_000 += d0.powi(3); s0_001 += 3.0*d0.powi(2)*d1; s0_011 += 3.0*d0*d1.powi(2); s0_111 += d1.powi(3); }
    let mut s1_000=0.0; let mut s1_001=0.0; let mut s1_011=0.0; let mut s1_111=0.0;
    for j in 0..c1 { let d0=deg10[j]; let d1=deg11[j]; s1_000 += d0.powi(3); s1_001 += 3.0*d0.powi(2)*d1; s1_011 += 3.0*d0*d1.powi(2); s1_111 += d1.powi(3); }
    let denom_star = n_f.powi(4);
    let s0_000 = if denom_star>0.0 { s0_000/denom_star } else { 0.0 }; 
    let s0_001 = if denom_star>0.0 { s0_001/denom_star } else { 0.0 }; 
    let s0_011 = if denom_star>0.0 { s0_011/denom_star } else { 0.0 }; 
    let s0_111 = if denom_star>0.0 { s0_111/denom_star } else { 0.0 }; 
    let s1_000 = if denom_star>0.0 { s1_000/denom_star } else { 0.0 }; 
    let s1_001 = if denom_star>0.0 { s1_001/denom_star } else { 0.0 }; 
    let s1_011 = if denom_star>0.0 { s1_011/denom_star } else { 0.0 }; 
    let s1_111 = if denom_star>0.0 { s1_111/denom_star } else { 0.0 };

    // Values computed above: p3_* and s*_*
    STATS_WRITER.with(|slot| {
        if let Some(w) = slot.borrow_mut().as_mut() {
            w.write_row_extended(t, col0, col1, e00, e01, e11, ne00, ne01, ne11, cyc000, cyc001, cyc011, cyc111, p000, p001, p010, p011, p101, p111,
                p3_0000, p3_0001, p3_0010, p3_0011, p3_0101, p3_0110, p3_0111, p3_1001, p3_1011, p3_1111,
                s0_000, s0_001, s0_011, s0_111, s1_000, s1_001, s1_011, s1_111);
            if w.dump_adj {
                w.write_adjacency_snapshot(t, adj, colour, last_flip, n);
            }
        }
    });
}

pub fn flush_stats() {
    STATS_WRITER.with(|slot| { if let Some(w) = slot.borrow_mut().as_mut() { w.flush(); } });
}

struct CsvStatsWriter { w: BufWriter<File>, dump_adj: bool, base_path: String, header_line: String }
impl CsvStatsWriter {
    fn new(path_opt: Option<String>, args: &Cli, effective_seed: u64, seed_random: bool, dump_adj: bool) -> std::io::Result<Self> {
        // Ensure output directory exists
        let out_dir = std::path::Path::new("output");
        if !out_dir.exists() { let _ = create_dir_all(out_dir); }
        let path = if let Some(p) = path_opt { p } else {
            let ts = Local::now().format("%Y%m%d-%H%M%S");
            format!("output/simulation-{}.csv", ts)
        };
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
        // Build the main header line first so we can reuse it in adjacency snapshot files verbatim.
        let header_line = format!("# netcoevolve={} n={} rho={} eta={} beta={} sd0={} sd1={} sc0={} sc1={} p1={} p00={} p01={} p11={} sample_delta={} t_max={} seed={}{} output_file={}",
            env!("CARGO_PKG_VERSION"),
            args.n,
            args.rho.unwrap_or(1.0),
            args.eta,
            match args.beta { Some(b) => b.to_string(), None => "-".to_string() },
            args.sd0,
            args.sd1,
            args.sc0,
            args.sc1,
            args.p1,
            args.p00,
            args.p01,
            args.p11,
            args.sample_delta,
            args.t_max,
            effective_seed,
            if seed_random { " (random)" } else { "" },
            path);
    writeln!(w, "{}", header_line)?;
    writeln!(w, "# coloured motif densities include symmetry multiplicities; sums across colour patterns recover uncoloured homomorphism densities for each motif family")?;
        if let Some(beta) = args.beta {
            writeln!(w, "# beta={} (eta=beta, rho=n)", beta)?;
        }
    writeln!(w, "time,col0,col1,e00,e01,e11,ne00,ne01,ne11,3cyc000,3cyc001,3cyc011,3cyc111,2p000,2p001,2p010,2p011,2p101,2p111,3p0000,3p0001,3p0010,3p0011,3p0101,3p0110,3p0111,3p1001,3p1011,3p1111,3s0_000,3s0_001,3s0_011,3s0_111,3s1_000,3s1_001,3s1_011,3s1_111")?;
        Ok(Self { w, dump_adj, base_path: path, header_line })
    }
    #[inline]
    fn write_row_extended(&mut self, t: f64, col0: f64, col1: f64, e00: f64, e01: f64, e11: f64, ne00: f64, ne01: f64, ne11: f64,
        cyc000: f64, cyc001: f64, cyc011: f64, cyc111: f64,
        p000: f64, p001: f64, p010: f64, p011: f64, p101: f64, p111: f64,
        p3_0000: f64, p3_0001: f64, p3_0010: f64, p3_0011: f64, p3_0101: f64, p3_0110: f64, p3_0111: f64, p3_1001: f64, p3_1011: f64, p3_1111: f64,
        s0_000: f64, s0_001: f64, s0_011: f64, s0_111: f64, s1_000: f64, s1_001: f64, s1_011: f64, s1_111: f64) {
        let _ = writeln!(self.w, "{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}",
            t, col0, col1, e00, e01, e11, ne00, ne01, ne11, cyc000, cyc001, cyc011, cyc111, p000, p001, p010, p011, p101, p111,
            p3_0000, p3_0001, p3_0010, p3_0011, p3_0101, p3_0110, p3_0111, p3_1001, p3_1011, p3_1111,
            s0_000, s0_001, s0_011, s0_111, s1_000, s1_001, s1_011, s1_111);
    }
    fn write_adjacency_snapshot(&mut self, t: f64, adj: &[u8], colour: &[u8], _last_flip: &[f64], n: usize) {
        // NEW FORMAT (2025-09-26):
        //  * Keep ORIGINAL vertex order (0..n-1) instead of colour / flip-time reordering.
        //  * Add SECOND comment line: a plain bitstring of vertex colours (0/1) in original order.
        //  * First header line unchanged except still includes group_size = count(colour==0) for backward compatibility.
        // Determine directory of the base CSV and build a per-simulation adjacency subfolder: <stem>-adj
        let base_path = std::path::Path::new(&self.base_path);
        let stem = base_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("simulation");
        let parent_dir = base_path.parent().unwrap_or(std::path::Path::new("."));
        let adj_dir = parent_dir.join(format!("{}-adj", stem));
        let _ = std::fs::create_dir_all(&adj_dir); // ignore errors silently
        let fname = adj_dir.join(format!("{}-adj-{:.6}.txt", stem, t));
        if let Ok(mut f) = File::create(&fname) {
            // Header line: simulation params + file type + time (+ group_size for legacy consumers)
            let group_size = colour.iter().filter(|&&c| c==0).count();
            let _ = writeln!(f, "{} filetype=adjacency_matrix time={:.9} group_size={}", self.header_line, t, group_size);
            // Second line: vertex colours bitstring in ORIGINAL order
            let colour_line: String = colour.iter().map(|&c| if c==0 { '0' } else { '1' }).collect();
            let _ = writeln!(f, "#{}", colour_line);
            // Adjacency matrix rows (original order)
            for u in 0..n {
                let mut line = String::with_capacity(n);
                for v in 0..n { line.push(if adj[u*n + v] != 0 { '1' } else { '0' }); }
                let _ = writeln!(f, "{}", line);
            }
        }
    }
    fn flush(&mut self) { let _ = self.w.flush(); }
}
