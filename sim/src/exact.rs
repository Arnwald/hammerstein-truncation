//! Exact truncation error of single-level identification for two static
//! nonlinearities of the same radius rho: `rho arctan(x/rho)` and
//! `rho artanh(x/rho)`.
//!
//! The system is `y = F(x)`, with no filter: one kernel per order,
//! `G_n = c_n` (the Taylor coefficients of F), and all its harmonics under a
//! cosine of amplitude lambda are known in closed form
//! ([`StaticFn::harmonics_closed`]). The identification is made at
//! `lambda = 1`; N harmonics are kept, the triangular block `A_N` of the
//! mixing coefficients is inverted, and the harmonics are predicted at lambda.
//! Everything is exact: no sweep, no window, no noise; only the truncation
//! remains.
//!
//! With `g_hat = A_N^-1 h_N(1)` and `Delta = g_hat - c_N`, for each harmonic
//! `m <= N`:
//!
//! ```text
//!     e_m(lambda)  = h_m(lambda) - sum_{n<=N} a_mn lambda^n g_hat_n     (total error)
//!     e1_m(lambda) = h_m(lambda) - sum_{n<=N} a_mn lambda^n c_n         (missing orders)
//!     e2_m(lambda) = -sum_{n<=N} a_mn lambda^n Delta_n                  (pollution)
//! ```
//!
//! `e = e1 + e2` exactly. `e1` is the true function (truth minus truncated
//! truth): for the arctan it stays finite beyond rho, only its series
//! `sum_{n>N} a_mn c_n lambda^n` diverges; that series is written out as well
//! (SERIES_TERMS terms) to show it. The artanh exists only for `lambda < rho`
//! (and the identification requires `rho > 1`): NaN elsewhere.
//!
//! # Outputs (directory `--out`, `output/exact` by default)
//!
//! - `curves_{f}_rho{rho}.npy` `[5, M, L]`: `|e|, |e1|, |e2|, |series of e1|, |h|`
//!   for the M odd harmonics `<= N_CURVES`, on the grid of lambda (dB);
//! - `bounds_{f}_rho{rho}.npy` `[4, L]`: exact `||e2||_2`, its bound
//!   `kappa_2(A_N) max(lambda, lambda^N) ||B G_>||_2`, exact `||e1||_2`, its
//!   bound `lambda^(N+1) ||(sum_{n>N} |a_mn c_n|)_m||_2` (lambda <= 1);
//! - `floors_{f}.npy` `[len(N_LIST), R, M_MAX]`: low-level floor
//!   `|Delta_m / c_m|` (the limit of `|e_m / h_m|` as lambda -> 0), odd
//!   harmonics, NaN if `m > N`;
//! - `map_{f}.npy` `[R, L]`: relative error of the worst harmonic (dB), N_CURVES;
//! - `running_{f}.npy` `[3, M, U]`: running of the effective kernels with the
//!   matching scale mu (grid U in dB re rho, `N = N_CURVES`, odd orders):
//!   `g_hat_n(mu) / c_n`, `(g_hat_n(mu) - c_n) / c_n` (computed without
//!   subtraction, from the tail of the series, below rho/2) and
//!   `beta_n(mu) / c_n = (mu d g_hat_n / d mu) / c_n` from the closed flow
//!   (see [`running`]); NaN beyond rho for the artanh;
//! - `describing_{f}.npy` `[2, U]`: `N = 1`, `g(mu) = g_hat_1(mu)` (the
//!   describing function) and `beta(mu) = mu dg/dmu`;
//! - `taylor_{f}.npy` `[1 + P, X]`: F(x) and its Taylor polynomials of degrees
//!   TAYLOR_DEGREES on the grid `x / rho` (rho = 1); NaN beyond rho for the
//!   artanh;
//! - `meta.json`.
//!
//! # Running of the kernels
//!
//! Matching at mu means choosing `g_hat(mu)` so that the truncated model
//! reproduces the N harmonics exactly at mu:
//! `A_N Lambda_N(mu) g_hat(mu) = h_N(mu)`. The error of a prediction at
//! lambda made with the kernels matched at lambda_0 is then
//! `e(lambda) = A_N Lambda_N(lambda) [g_hat(lambda) - g_hat(lambda_0)]`: the
//! running of the kernels between the two levels. For the arctan and the
//! artanh, `h_m` is proportional to `s(mu)^m` and
//! `mu dh_m/dmu = m h_m / sqrt(1 + sigma mu^2 / rho^2)` (sigma = +1 for the
//! arctan, -1 for the artanh); the flow closes on `g_hat`:
//!
//! ```text
//!     mu d g_hat / d mu = [ Lambda^-1 A_N^-1 K A_N Lambda / sqrt(1 + sigma mu^2/rho^2) - K ] g_hat,
//!     K = diag(1 ... N).
//! ```
//!
//! For `N = 1`, `g = g_hat_1 = 2 / (1 + sqrt(1 + sigma mu^2 / rho^2))` and
//! `beta(g) = 2 g (g - 1) / (2 - g)`, the same function for both
//! nonlinearities, with no explicit dependence on mu.

use rayon::prelude::*;
use std::path::PathBuf;

use crate::linalg::singular_values;
use crate::mixing::cos_coef;
use crate::out::{Json, Out};
use crate::statics::{Shape, StaticFn};

/// Grid of prediction levels lambda (dB re the identification level).
const LAM_DB: (f64, f64, f64) = (-40.0, 14.0, 0.05);
/// Grid of radii for the map and the floors (dB); artanh: rho > 1 only.
const RHO_DB: (f64, f64, f64) = (-6.0, 14.0, 0.05);
/// Number of harmonics kept for the curves and the map.
const N_CURVES: usize = 7;
/// Numbers of harmonics kept for the floors.
const N_LIST: [usize; 4] = [3, 5, 7, 9];
/// Odd harmonics followed in the floors (m = 1, 3, ..., 2 M_MAX - 1).
const M_MAX: usize = 5;
/// Radii of the detailed curves.
const RHO_CURVES: [f64; 3] = [1.3, 2.0, 2.5];
/// Terms of the series of e1 (beyond rho it diverges).
const SERIES_TERMS: usize = 127;
/// Below `rho * SERIES_SWITCH`, e1 is summed by its series (no cancellation).
const SERIES_SWITCH: f64 = 0.5;

const FUNCTIONS: [&str; 2] = ["arctan", "artanh"];
/// Grid of `x / rho` and odd degrees of the Taylor polynomials of the radius figure.
const TAYLOR_X: (f64, f64, f64) = (0.0, 2.5, 0.005);
const TAYLOR_DEGREES: [usize; 4] = [1, 5, 9, 13];
/// Grid of the matching scale mu (dB re rho) for the running of the kernels.
const MU_DB: (f64, f64, f64) = (-30.0, 20.0, 0.05);

/// `lo, lo + step, ..., hi`.
pub fn grid((lo, hi, step): (f64, f64, f64)) -> Vec<f64> {
    let n = ((hi - lo) / step).round() as usize;
    (0..=n).map(|i| lo + step * i as f64).collect()
}

pub fn from_db(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

fn shape(name: &str, rho: f64) -> StaticFn {
    match name {
        "arctan" => StaticFn::new(Shape::Arctan { rho }, false),
        "artanh" => StaticFn::new(Shape::Artanh { rho }, false),
        _ => unreachable!(),
    }
}

/// Single-level identification of a scalar system `y = F(x)` at lambda = 1.
pub struct SingleLevel {
    pub f: StaticFn,
    pub n: usize,
    /// c_0 ... c_K (K = SERIES_TERMS).
    pub c: Vec<f64>,
    /// g_hat_0 ... g_hat_N (g_hat_0 unused).
    pub g_hat: Vec<f64>,
}

/// One harmonic at one level: `h`, the total error `e`, its two terms and
/// the partial sum of the series of `e1`.
#[derive(Clone, Copy, Debug)]
pub struct Errors {
    pub h: f64,
    pub e: f64,
    pub e1: f64,
    pub e2: f64,
    pub e1_series: f64,
}

impl SingleLevel {
    /// None if the identification is impossible (artanh with rho <= 1).
    pub fn new(f: StaticFn, n: usize) -> Option<Self> {
        if !f.defined(1.0) {
            return None;
        }
        let g_hat = back_substitute(&f.harmonics_closed(1.0, n), n);
        let c = f.taylor(SERIES_TERMS);
        Some(SingleLevel { f, n, c, g_hat })
    }

    /// Low-level floor `|Delta_m / c_m|` (the limit of `|e_m / h_m|` as lambda -> 0).
    pub fn floor(&self, m: usize) -> f64 {
        ((self.g_hat[m] - self.c[m]) / self.c[m]).abs()
    }

    /// The harmonics m = 1 ... N at lambda; None if F is not defined there.
    pub fn at(&self, lam: f64) -> Option<Vec<Errors>> {
        if !self.f.defined(lam) {
            return None;
        }
        let h = self.f.harmonics_closed(lam, self.n);
        let rho = self.f.rho();
        let pw: Vec<f64> = (0..=SERIES_TERMS).map(|k| lam.powi(k as i32)).collect();
        Some(
            (1..=self.n)
                .map(|m| {
                    let model: f64 = (m..=self.n)
                        .map(|k| cos_coef(k, m) * pw[k] * self.g_hat[k])
                        .sum();
                    let truncated: f64 = (m..=self.n)
                        .map(|k| cos_coef(k, m) * pw[k] * self.c[k])
                        .sum();
                    let e2: f64 = -(m..=self.n)
                        .map(|k| cos_coef(k, m) * pw[k] * (self.g_hat[k] - self.c[k]))
                        .sum::<f64>();
                    let series: f64 = (self.n + 1..=SERIES_TERMS)
                        .rev()
                        .map(|k| cos_coef(k, m) * pw[k] * self.c[k])
                        .sum();
                    // e1: by its series where it converges fast (no cancellation),
                    // otherwise truth minus truncated truth (the function, beyond
                    // rho included)
                    let e1 = if lam < SERIES_SWITCH * rho {
                        series
                    } else {
                        h[m] - truncated
                    };
                    Errors {
                        h: h[m],
                        e: h[m] - model,
                        e1,
                        e2,
                        e1_series: series,
                    }
                })
                .collect(),
        )
    }

    /// `||B G_>||_2` (= `||h_N(1) - A_N c_N||_2`) and `kappa_2(A_N)` (the
    /// complete block, both parities).
    pub fn bound_constants(&self) -> (f64, f64) {
        let h1 = self.f.harmonics_closed(1.0, self.n);
        let bg: f64 = (1..=self.n)
            .map(|m| {
                let t: f64 = (m..=self.n).map(|k| cos_coef(k, m) * self.c[k]).sum();
                (h1[m] - t).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let d: Vec<Vec<f64>> = (1..=self.n)
            .map(|m| (1..=self.n).map(|k| cos_coef(k, m)).collect())
            .collect();
        let sv = singular_values(&d);
        let kappa = sv.iter().copied().fold(0.0, f64::max)
            / sv.iter().copied().fold(f64::INFINITY, f64::min);
        (bg, kappa)
    }

    /// `sum_{n>N} |a_mn c_n|` (bound of e1 on [0, 1]); infinite if rho <= 1.
    pub fn tail_abs(&self, m: usize) -> f64 {
        if self.f.rho() <= 1.0 {
            return f64::INFINITY;
        }
        (self.n + 1..=SERIES_TERMS)
            .map(|k| (cos_coef(k, m) * self.c[k]).abs())
            .sum()
    }
}

/// Solves `A_N x = y` (upper triangular block of the mixing coefficients,
/// indices 1 ... N; index 0 unused).
fn back_substitute(y: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n + 1];
    for m in (1..=n).rev() {
        let s: f64 = (m + 1..=n).map(|k| cos_coef(k, m) * x[k]).sum();
        x[m] = (y[m] - s) / cos_coef(m, m);
    }
    x
}

/// Effective kernels matched at the scale mu (N harmonics, indices 1 ... N):
/// `g_hat(mu) = Lambda_N(mu)^-1 A_N^-1 h_N(mu)`, their deviation from the true
/// kernels `g_hat(mu) - c` (without subtraction below rho/2: from the tail of
/// the series), and their beta function `mu d g_hat / d mu` from the closed
/// flow of the module documentation. Only the arctan (sigma = +1) and the
/// artanh (sigma = -1); None outside the domain of the function.
#[allow(clippy::type_complexity)]
pub fn running(name: &str, rho: f64, n: usize, mu: f64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let sigma = match name {
        "arctan" => 1.0,
        "artanh" => -1.0,
        _ => panic!("closed flow: arctan or artanh only"),
    };
    let f = shape(name, rho);
    if !f.defined(mu) {
        return None;
    }
    let h = f.harmonics_closed(mu, n);
    let c = f.taylor(SERIES_TERMS);
    let pw = |k: usize| mu.powi(k as i32);
    let y = back_substitute(&h, n);
    let g: Vec<f64> = (0..=n)
        .map(|k| if k == 0 { 0.0 } else { y[k] / pw(k) })
        .collect();
    // tail h_m - sum_{k<=N} a_mk c_k mu^k: by the series where it converges fast
    let tail: Vec<f64> = (0..=n)
        .map(|m| {
            if m == 0 {
                0.0
            } else if mu < SERIES_SWITCH * rho {
                (n + 1..=SERIES_TERMS)
                    .rev()
                    .map(|k| cos_coef(k, m) * pw(k) * c[k])
                    .sum()
            } else {
                h[m] - (m..=n).map(|k| cos_coef(k, m) * pw(k) * c[k]).sum::<f64>()
            }
        })
        .collect();
    let z = back_substitute(&tail, n);
    let dev: Vec<f64> = (0..=n)
        .map(|k| if k == 0 { 0.0 } else { z[k] / pw(k) })
        .collect();
    let r = (1.0 + sigma * (mu / rho).powi(2)).sqrt();
    let kh: Vec<f64> = (0..=n).map(|m| m as f64 * h[m] / r).collect();
    let w = back_substitute(&kh, n);
    let beta: Vec<f64> = (0..=n)
        .map(|k| {
            if k == 0 {
                0.0
            } else {
                w[k] / pw(k) - k as f64 * g[k]
            }
        })
        .collect();
    Some((g, dev, beta))
}

fn odd_harmonics(n: usize) -> Vec<usize> {
    (1..=n).step_by(2).collect()
}

/// Relative error of the worst odd harmonic (dB); NaN outside the domain.
fn worst_db(s: &SingleLevel, lam: f64) -> f64 {
    match s.at(lam) {
        None => f64::NAN,
        Some(v) => {
            let worst = odd_harmonics(s.n)
                .iter()
                .map(|&m| (v[m - 1].e / v[m - 1].h).abs())
                .fold(0.0, f64::max);
            20.0 * worst.max(1e-300).log10()
        }
    }
}

pub fn run(argv: Vec<String>) {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../output/exact");
    let mut it = argv.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--out" => dir = PathBuf::from(it.next().expect("--out DIR")),
            _ => panic!("unknown option: {a}"),
        }
    }
    let out = Out::new(dir);
    let lam_db = grid(LAM_DB);
    let rho_db = grid(RHO_DB);
    let n_odd = odd_harmonics(N_CURVES).len();

    for name in FUNCTIONS {
        // Detailed curves and bounds
        for rho in RHO_CURVES {
            let s = SingleLevel::new(shape(name, rho), N_CURVES).expect("rho > 1");
            let rows: Vec<Vec<f64>> = lam_db
                .par_iter()
                .map(|&l| {
                    let mut col = vec![f64::NAN; 5 * n_odd];
                    if let Some(v) = s.at(from_db(l)) {
                        for (i, &m) in odd_harmonics(N_CURVES).iter().enumerate() {
                            let x = v[m - 1];
                            for (q, x) in [x.e, x.e1, x.e2, x.e1_series, x.h].iter().enumerate() {
                                col[q * n_odd + i] = x.abs();
                            }
                        }
                    }
                    col
                })
                .collect();
            // [lambda][q M + i] -> [q][i][lambda]
            let mut data = vec![f64::NAN; 5 * n_odd * lam_db.len()];
            for (li, col) in rows.iter().enumerate() {
                for (qi, &x) in col.iter().enumerate() {
                    data[qi * lam_db.len() + li] = x;
                }
            }
            // check: e = e1 + e2
            let mut worst_split: f64 = 0.0;
            for &l in &lam_db {
                if let Some(v) = s.at(from_db(l)) {
                    for x in v {
                        let scale = x.h.abs().max(x.e.abs()).max(1e-300);
                        worst_split = worst_split.max((x.e - x.e1 - x.e2).abs() / scale);
                    }
                }
            }
            println!("{name} rho = {rho}: |e - e1 - e2| / max(|h|, |e|) <= {worst_split:.1e}");
            out.f64(
                &format!("curves_{name}_rho{rho}.npy"),
                &[5, n_odd, lam_db.len()],
                &data,
            );

            let (bg, kappa) = s.bound_constants();
            let tails: Vec<f64> = (1..=N_CURVES).map(|m| s.tail_abs(m)).collect();
            let tail_norm = tails.iter().map(|t| t * t).sum::<f64>().sqrt();
            let mut b = vec![f64::NAN; 4 * lam_db.len()];
            for (li, &l) in lam_db.iter().enumerate() {
                let lam = from_db(l);
                b[lam_db.len() + li] = kappa * lam.max(lam.powi(N_CURVES as i32)) * bg;
                if lam <= 1.0 {
                    b[3 * lam_db.len() + li] = lam.powi(N_CURVES as i32 + 1) * tail_norm;
                }
                if let Some(v) = s.at(lam) {
                    b[li] = v.iter().map(|x| x.e2 * x.e2).sum::<f64>().sqrt();
                    b[2 * lam_db.len() + li] = v.iter().map(|x| x.e1 * x.e1).sum::<f64>().sqrt();
                }
            }
            out.f64(
                &format!("bounds_{name}_rho{rho}.npy"),
                &[4, lam_db.len()],
                &b,
            );
            println!(
                "    kappa_2(A_{N_CURVES}) = {kappa:.1}, ||B G_>||_2 = {bg:.3e}, floors {}",
                odd_harmonics(N_CURVES)
                    .iter()
                    .map(|&m| format!("m={m}: {:.1} dB", 20.0 * s.floor(m).log10()))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        // Low-level floors
        let mut floors = vec![f64::NAN; N_LIST.len() * rho_db.len() * M_MAX];
        for (ni, &n) in N_LIST.iter().enumerate() {
            for (ri, &r) in rho_db.iter().enumerate() {
                if let Some(s) = SingleLevel::new(shape(name, from_db(r)), n) {
                    for (i, m) in odd_harmonics(n).into_iter().enumerate().take(M_MAX) {
                        floors[(ni * rho_db.len() + ri) * M_MAX + i] = s.floor(m);
                    }
                }
            }
        }
        out.f64(
            &format!("floors_{name}.npy"),
            &[N_LIST.len(), rho_db.len(), M_MAX],
            &floors,
        );

        // Map (lambda, rho) of the worst harmonic, N = N_CURVES
        let map: Vec<f64> = rho_db
            .par_iter()
            .flat_map_iter(|&r| {
                let s = SingleLevel::new(shape(name, from_db(r)), N_CURVES);
                lam_db
                    .iter()
                    .map(move |&l| s.as_ref().map_or(f64::NAN, |s| worst_db(s, from_db(l))))
                    .collect::<Vec<_>>()
            })
            .collect();
        out.f64(
            &format!("map_{name}.npy"),
            &[rho_db.len(), lam_db.len()],
            &map,
        );
    }

    // Running of the effective kernels with the matching scale (rho = 1)
    let mu_db = grid(MU_DB);
    for name in FUNCTIONS {
        let c = shape(name, 1.0).taylor(N_CURVES);
        let ms = odd_harmonics(N_CURVES);
        let mut data = vec![f64::NAN; 3 * ms.len() * mu_db.len()];
        let mut desc = vec![f64::NAN; 2 * mu_db.len()];
        for (ui, &u) in mu_db.iter().enumerate() {
            if let Some((g, dev, beta)) = running(name, 1.0, N_CURVES, from_db(u)) {
                for (i, &m) in ms.iter().enumerate() {
                    for (q, x) in [g[m], dev[m], beta[m]].iter().enumerate() {
                        data[(q * ms.len() + i) * mu_db.len() + ui] = x / c[m];
                    }
                }
            }
            if let Some((g, _, beta)) = running(name, 1.0, 1, from_db(u)) {
                desc[ui] = g[1];
                desc[mu_db.len() + ui] = beta[1];
            }
        }
        out.f64(
            &format!("running_{name}.npy"),
            &[3, ms.len(), mu_db.len()],
            &data,
        );
        out.f64(&format!("describing_{name}.npy"), &[2, mu_db.len()], &desc);
        let (g, _, _) = running(name, 1.0, N_CURVES, 0.5).unwrap();
        println!(
            "{name}: g_hat_n(rho/2) / c_n = {}",
            ms.iter()
                .map(|&m| format!("{:.4}", g[m] / c[m]))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Taylor polynomials against the function (rho = 1)
    let xs = grid(TAYLOR_X);
    for name in FUNCTIONS {
        let f = shape(name, 1.0);
        let c = f.taylor(*TAYLOR_DEGREES.last().unwrap());
        let mut data = vec![f64::NAN; (1 + TAYLOR_DEGREES.len()) * xs.len()];
        for (xi, &x) in xs.iter().enumerate() {
            if f.defined(x) {
                data[xi] = f.eval(x);
            }
            for (di, &d) in TAYLOR_DEGREES.iter().enumerate() {
                let p: f64 = (1..=d).map(|k| c[k] * x.powi(k as i32)).sum();
                data[(1 + di) * xs.len() + xi] = p;
            }
        }
        out.f64(
            &format!("taylor_{name}.npy"),
            &[1 + TAYLOR_DEGREES.len(), xs.len()],
            &data,
        );
    }

    Json::new()
        .str("experiment", "exact")
        .strs("functions", &FUNCTIONS)
        .nums("lam_db", &lam_db)
        .nums("rho_db", &rho_db)
        .int("n_curves", N_CURVES as i64)
        .nums(
            "n_list",
            &N_LIST.iter().map(|&n| n as f64).collect::<Vec<_>>(),
        )
        .nums("rho_curves", &RHO_CURVES)
        .int("m_max", M_MAX as i64)
        .int("series_terms", SERIES_TERMS as i64)
        .strs("curve_quantities", &["e", "e1", "e2", "e1_series", "h"])
        .nums("mu_db", &mu_db)
        .nums("taylor_x", &xs)
        .nums(
            "taylor_degrees",
            &TAYLOR_DEGREES.iter().map(|&d| d as f64).collect::<Vec<_>>(),
        )
        .strs("running_quantities", &["g/c", "(g-c)/c", "beta/c"])
        .write(&out, "meta.json");
    println!("output: {}", out.dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error vanishes at lambda = 1, e = e1 + e2, the floor is the limit
    /// as lambda -> 0, and the series of e1 matches the function inside the disc.
    #[test]
    fn exact_split_and_limits() {
        for name in FUNCTIONS {
            for rho in [1.3, 2.0, 4.0] {
                for n in [3, 7, 9] {
                    let s = SingleLevel::new(shape(name, rho), n).unwrap();
                    for x in s.at(1.0).unwrap() {
                        assert!(
                            x.e.abs() <= 1e-13 * x.h.abs().max(1e-300),
                            "{name} rho={rho} N={n} lambda=1"
                        );
                        assert!((x.e - x.e1 - x.e2).abs() <= 1e-12 * x.h.abs(), "split");
                    }
                    let v = s.at(1e-3).unwrap();
                    for m in odd_harmonics(n) {
                        let r = (v[m - 1].e / v[m - 1].h).abs();
                        assert!(
                            (r / s.floor(m) - 1.0).abs() < 1e-4,
                            "{name} rho={rho} N={n} m={m}"
                        );
                    }
                    // above rho/2: e1 = truth - truncated truth, against the series
                    for x in s.at(0.7 * rho).unwrap() {
                        assert!(
                            (x.e1 - x.e1_series).abs() <= 1e-9 * x.h.abs(),
                            "{name} series"
                        );
                    }
                }
            }
        }
    }

    /// The numbers quoted in the paper for N = 7 and rho = 2: relative bias
    /// of the seventh kernel -0.33 (arctan) and +0.62 (artanh), leading term
    /// -+N / (4 rho^2), and kappa_2(A_7) = 221.
    #[test]
    fn quoted_values() {
        let at = SingleLevel::new(shape("arctan", 2.0), 7).unwrap();
        let ah = SingleLevel::new(shape("artanh", 2.0), 7).unwrap();
        let bias = |s: &SingleLevel| (s.g_hat[7] - s.c[7]) / s.c[7];
        assert!((bias(&at) + 0.331).abs() < 1e-3, "{}", bias(&at));
        assert!((bias(&ah) - 0.625).abs() < 1e-3, "{}", bias(&ah));
        // first term of the bias: (a_79 / a_77) c_9 / c_7 = -+ 7/16
        let lead = cos_coef(9, 7) / cos_coef(7, 7) * at.c[9] / at.c[7];
        assert!((lead + 7.0 / 16.0).abs() < 1e-14);
        let (_, kappa) = at.bound_constants();
        assert!((kappa - 220.8).abs() < 0.1, "{kappa}");
    }

    /// Running of the kernels: g_hat(1) is the single-level estimate, the
    /// error is the running between lambda_0 = 1 and lambda, the deviation
    /// computed without subtraction is the deviation, the beta function of
    /// the closed flow is the derivative, and the power-counting exponents
    /// are N + 2 - n.
    #[test]
    fn running_kernels() {
        for name in FUNCTIONS {
            for rho in [1.6, 2.0, 3.0] {
                let n = 7;
                let s = SingleLevel::new(shape(name, rho), n).unwrap();
                let (g1, _, _) = running(name, rho, n, 1.0).unwrap();
                for m in odd_harmonics(n) {
                    assert!(
                        (g1[m] - s.g_hat[m]).abs() <= 1e-12 * s.c[m].abs(),
                        "g_hat(1)"
                    );
                }
                for lam in [0.3, 0.8, 1.4] {
                    let (gl, dev, beta) = running(name, rho, n, lam).unwrap();
                    let v = s.at(lam).unwrap();
                    for m in odd_harmonics(n) {
                        // e(lambda) = A Lambda(lambda) [g_hat(lambda) - g_hat(1)]
                        let e: f64 = (m..=n)
                            .map(|k| cos_coef(k, m) * lam.powi(k as i32) * (gl[k] - g1[k]))
                            .sum();
                        assert!(
                            (e - v[m - 1].e).abs() <= 1e-11 * v[m - 1].h.abs(),
                            "identity"
                        );
                        assert!((dev[m] - (gl[m] - s.c[m])).abs() <= 1e-9 * s.c[m].abs());
                        // beta against a centred difference
                        let d = 1e-5 * lam;
                        let (gp, _, _) = running(name, rho, n, lam + d).unwrap();
                        let (gm, _, _) = running(name, rho, n, lam - d).unwrap();
                        let fd = lam * (gp[m] - gm[m]) / (2.0 * d);
                        assert!(
                            (fd - beta[m]).abs() <= 1e-6 * s.c[m].abs(),
                            "beta {name} m={m}"
                        );
                    }
                }
                for m in [3usize, 5, 7] {
                    let r = |mu: f64| (running(name, rho, n, mu).unwrap().1[m] / s.c[m]).abs();
                    let slope = (r(0.02 * rho) / r(0.01 * rho)).log2();
                    assert!((slope - (n + 2 - m) as f64).abs() < 1e-3, "exponent m={m}");
                }
            }
        }
    }

    /// N = 1: g = 2 / (1 + sqrt(1 + sigma mu^2 / rho^2)) and
    /// beta(g) = 2 g (g - 1) / (2 - g).
    #[test]
    fn describing_function_beta() {
        for (name, sigma) in [("arctan", 1.0), ("artanh", -1.0)] {
            for x in [0.1, 0.5, 0.9, 0.99, 2.0, 5.0] {
                let Some((g, _, beta)) = running(name, 2.0, 1, 2.0 * x) else {
                    assert!(sigma < 0.0 && x >= 1.0);
                    continue;
                };
                let g0 = 2.0 / (1.0 + (1.0 + sigma * x * x).sqrt());
                assert!((g[1] - g0).abs() < 1e-13);
                let b0 = 2.0 * g0 * (g0 - 1.0) / (2.0 - g0);
                assert!(
                    (beta[1] - b0).abs() < 1e-12 * b0.abs().max(1e-3),
                    "{name} x={x}"
                );
            }
        }
    }

    /// The arctan stays defined beyond rho, the artanh does not; the artanh
    /// cannot be identified if rho <= 1.
    #[test]
    fn domains() {
        let s = SingleLevel::new(shape("arctan", 0.7), 7).unwrap();
        assert!(s.at(3.0).is_some());
        assert!(SingleLevel::new(shape("artanh", 0.9), 7).is_none());
        let s = SingleLevel::new(shape("artanh", 2.0), 7).unwrap();
        assert!(s.at(1.99).is_some() && s.at(2.0).is_none());
    }
}
