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
use crate::Cli;

thread_local! {
    static STATS_WRITER: std::cell::RefCell<Option<CsvStatsWriter>> = std::cell::RefCell::new(None);
}

pub fn init_stats_writer(path_opt: Option<String>, args: &Cli) {
    STATS_WRITER.with(|slot| {
        *slot.borrow_mut() = Some(CsvStatsWriter::new(path_opt, args).expect("create output CSV"));
    });
}

/// Compute stats and append CSV row.
#[inline]
pub fn compute_stats(t: f64, adj: &[u8], colour: &[u8], n: usize) {
    debug_assert!(adj.len() == n * n && colour.len() == n);
    // Build colour index lists
    let mut idx0 = Vec::new();
    let mut idx1 = Vec::new();
    for (i, &c) in colour.iter().enumerate() { if c == 0 { idx0.push(i); } else { idx1.push(i); } }
    let c0 = idx0.len();
    let c1 = idx1.len();
    let frac0 = c0 as f64 / n as f64;
    let frac1 = c1 as f64 / n as f64;

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

    let sum_sym = |m: &faer::Mat<f32>| -> f64 { let mut s=0.0; for i in 0..m.nrows() { for j in 0..m.ncols() { s += m[(i,j)] as f64; } } s };
    let sum_rect = |m: &faer::Mat<f32>| -> f64 { let mut s=0.0; for i in 0..m.nrows() { for j in 0..m.ncols() { s += m[(i,j)] as f64; } } s };
    let sum00 = sum_sym(&a00); // counts undirected edges twice
    let sum11 = sum_sym(&a11);
    let sum01 = sum_rect(&a01); // cross once
    let e00 = if c0 > 1 { sum00 / (c0 as f64 * (c0 as f64 - 1.0)) } else { 0.0 };
    let e11 = if c1 > 1 { sum11 / (c1 as f64 * (c1 as f64 - 1.0)) } else { 0.0 };
    let e01 = if c0 > 0 && c1 > 0 { sum01 / (c0 as f64 * c1 as f64) } else { 0.0 };

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

    let denom_all = if n >= 3 { (n as f64)*(n as f64 -1.0)*(n as f64 -2.0)/6.0 } else { 0.0 };
    let (cyc000,cyc001,cyc011,cyc111) = if denom_all > 0.0 {
        (
            cyc000_count / denom_all,
            cyc001_count / denom_all,
            cyc011_count / denom_all,
            cyc111_count / denom_all,
        )
    } else { (0.0,0.0,0.0,0.0) };

    STATS_WRITER.with(|slot| {
        if let Some(w) = slot.borrow_mut().as_mut() {
            w.write_row_extended(t, frac0, frac1, e00, e01, e11, cyc000, cyc001, cyc011, cyc111);
        }
    });
}

pub fn flush_stats() {
    STATS_WRITER.with(|slot| { if let Some(w) = slot.borrow_mut().as_mut() { w.flush(); } });
}

struct CsvStatsWriter { w: BufWriter<File> }
impl CsvStatsWriter {
    fn new(path_opt: Option<String>, args: &Cli) -> std::io::Result<Self> {
        // Ensure output directory exists
        let out_dir = std::path::Path::new("output");
        if !out_dir.exists() { let _ = create_dir_all(out_dir); }
        let path = if let Some(p) = path_opt { p } else {
            let ts = Local::now().format("%Y%m%d-%H%M%S");
            format!("output/simulation-{}.csv", ts)
        };
        let f = File::create(&path)?;
        let mut w = BufWriter::new(f);
    writeln!(w, "# n={} rho={} eta={} sd0={} sd1={} sc0={} sc1={} sample_delta={} t_max={} seed={} output_file={}",
            args.n, args.rho, args.eta, args.sd0, args.sd1, args.sc0, args.sc1, args.sample_delta, args.t_max, args.seed, path)?;
        writeln!(w, "time,frac0,frac1,e00,e01,e11,3cyc000,3cyc001,3cyc011,3cyc111")?;
        Ok(Self { w })
    }
    #[inline]
    fn write_row_extended(&mut self, t: f64, frac0: f64, frac1: f64, e00: f64, e01: f64, e11: f64, cyc000: f64, cyc001: f64, cyc011: f64, cyc111: f64) {
        let _ = writeln!(self.w, "{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}", t, frac0, frac1, e00, e01, e11, cyc000, cyc001, cyc011, cyc111);
    }
    fn flush(&mut self) { let _ = self.w.flush(); }
}
