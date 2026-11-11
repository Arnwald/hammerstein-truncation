//! Deconvolved response of a ninth-order Hammerstein cascade to an
//! exponential swept sine: the response to the m-th harmonic sits `L ln m`
//! before the linear impulse response.
//!
//! A 10 s Farina sweep sampled at 44.1 kHz (L = 0.88 s) drives the cascade of
//! `sim/systems/cascade9_fs44100.txt`; the powers of the sweep are generated
//! without aliasing ([`crate::spectral::alias_free_power`]), the output is
//! recorded with 0.5 s of silence before and after, and deconvolved with
//! Farina's inverse filter.
//!
//! # Outputs (directory `--out`, `output/ess` by default)
//!
//! - `ir_zoom.npy`: the deconvolved response from 10 ms before the ninth
//!   harmonic to 150 ms after the linear response;
//! - `meta.json`: `fs`, `L`, `t0` (index of the linear peak in the full
//!   response), `ir_zoom_start` (index of the first sample of the zoom),
//!   `harmonics`.

use std::path::{Path, PathBuf};

use crate::filters::{Sos, sosfilt};
use crate::out::{Json, Out};
use crate::spectral::{Deconvolver, alias_free_power, analytic};
use crate::sweep::{Sweep, SweepSpec};

const SYSTEM_FILE: &str = "cascade9_fs44100.txt";
const FS: f64 = 44_100.0;
const DURATION: f64 = 10.0;
const SWEEP_F1: f64 = 1.0;
const SWEEP_F2: f64 = 12_127.5;
const SWEEP_F2_MARGIN: f64 = 1.8;
const AA_PASS: f64 = 0.44;
const HARMONICS: usize = 9;
const SILENCE_MS: f64 = 500.0;

/// One branch of a polynomial cascade: `gain * B[x^power]`.
pub struct Branch {
    pub power: usize,
    pub gain: f64,
    pub sos: Vec<Sos>,
}

/// One branch per line: `order gain b0 b1 b2 a0 a1 a2 [b0 ...]`; `#` starts a comment.
pub fn read_branches(path: &Path) -> Vec<Branch> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out: Vec<Branch> = text
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .map(|t| t.parse().expect("number"))
                .collect();
            assert!(
                v.len() >= 2 && (v.len() - 2).is_multiple_of(6),
                "malformed line: {l}"
            );
            let sos = v[2..]
                .chunks(6)
                .map(|c| {
                    let a0 = c[3];
                    [c[0] / a0, c[1] / a0, c[2] / a0, 1.0, c[4] / a0, c[5] / a0]
                })
                .collect();
            Branch {
                power: v[0] as usize,
                gain: v[1],
                sos,
            }
        })
        .collect();
    out.sort_by_key(|b| b.power);
    for (i, b) in out.iter().enumerate() {
        assert_eq!(b.power, i + 1, "the orders must be 1, 2, 3, ...");
    }
    out
}

/// Index of the largest sample in modulus, and of the first sample of its
/// rising edge above -20 dB of it.
fn peak_and_edge(ir: &[f64]) -> (usize, usize) {
    let mut t0 = 0;
    let mut top = -1.0;
    for (i, v) in ir.iter().enumerate() {
        if v.abs() > top {
            t0 = i;
            top = v.abs();
        }
    }
    let thr = top * 10f64.powf(-20.0 / 20.0);
    let mut rising = t0;
    for i in (1..=t0).rev() {
        if ir[i].abs() < thr {
            rising = i + 1;
            break;
        }
    }
    (t0, rising)
}

pub fn run(argv: Vec<String>) {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../output/ess");
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => dir = PathBuf::from(it.next().expect("--out DIR")),
            _ => panic!("unknown option: {a}"),
        }
    }
    let out = Out::new(dir);
    let sweep = Sweep::new(SweepSpec {
        fs: FS,
        f1: SWEEP_F1,
        duration: DURATION,
        f2: SWEEP_F2,
        margin: SWEEP_F2_MARGIN,
        ceiling: 1.0,
        integer_k: false,
    });
    println!(
        "sweep: {:.2} Hz to {:.1} Hz, L = {:.4} s, N = {}",
        sweep.f1_sweep, sweep.f2_cutoff, sweep.l_param, sweep.n_samples
    );
    let branches = read_branches(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("systems")
            .join(SYSTEM_FILE),
    );
    assert_eq!(branches.len(), HARMONICS);

    // y = sum_n gain_n B_n[x^n], powers without aliasing
    let z = analytic(&sweep.windowed);
    let mut y = vec![0.0; sweep.n_samples];
    for b in &branches {
        let mut p = alias_free_power(&z, b.power, &sweep.f_inst, FS, AA_PASS);
        sosfilt(&b.sos, &mut p);
        y.iter_mut().zip(&p).for_each(|(y, v)| *y += b.gain * v);
    }

    // recording with silence before and after, then deconvolution
    let pad = (SILENCE_MS * FS / 1000.0) as usize;
    let mut rec = vec![0.0; pad + y.len() + pad];
    rec[pad..pad + y.len()].copy_from_slice(&y);
    let ir = Deconvolver::new(&sweep.inverse, rec.len()).deconvolve(&rec);

    let (t0, rising) = peak_and_edge(&ir);
    let pos = |k: usize| rising as f64 - sweep.l_param * (k as f64).ln() * FS;
    let z0 = (pos(HARMONICS) - 0.010 * FS).floor().max(0.0) as usize;
    let z1 = ((rising as f64 + 0.150 * FS).ceil() as usize).min(ir.len());
    out.f64("ir_zoom.npy", &[z1 - z0], &ir[z0..z1]);
    Json::new()
        .str("experiment", "ess")
        .num("fs", FS)
        .num("duration", DURATION)
        .num("f1_sweep", sweep.f1_sweep)
        .num("f2", sweep.f2)
        .num("f2_cutoff", sweep.f2_cutoff)
        .num("L", sweep.l_param)
        .int("harmonics", HARMONICS as i64)
        .int("t0", t0 as i64)
        .int("ir_zoom_start", z0 as i64)
        .write(&out, "meta.json");
    println!("output: {}", out.dir.display());
}
