//! Pure Hammerstein systems built from static nonlinearities with infinitely
//! many orders: their Taylor coefficients, their harmonics in closed form,
//! their alias-free simulation, and their exact harmonic responses.
//!
//! # The system
//!
//! A static nonlinearity maps each input sample to an output sample,
//! `y = F(x)`. A test system has a few branches in parallel,
//!
//! ```text
//!     y = sum_j eps_j B_j[ F_j(x) ]
//! ```
//!
//! where `F_j` is a static function, `B_j` a linear filter (a cascade of
//! biquads) and `eps_j` a gain. Replacing each `F_j` by its Taylor series
//! `sum_n c_n^(j) x^n` gives the parallel Hammerstein cascade
//! `y = sum_n g_n * x^n`, with kernels `G_n(f) = sum_j eps_j c_n^(j) B_j(f)`:
//! infinitely many kernels, whose frequency shapes are those of the filters.
//!
//! # The functions ([`Shape`])
//!
//! - `rho arctan(x/rho)`: odd, logarithmic branch points at `+-i rho`,
//!   `c_(2k+1) = (-1)^k rho^(-2k) / (2k+1)`. Bounded: its series diverges
//!   beyond rho, the function does not.
//! - `rho artanh(x/rho)`: odd, branch points at `+-rho`, on the real axis,
//!   `c_(2k+1) = rho^(-2k) / (2k+1)`: the same moduli without the alternating
//!   signs. The function itself diverges at `|x| = rho`.
//! - `(x/rho)^2 / (1 + (x/rho)^2)` (a Lorentzian): even, poles at `+-i rho`,
//!   `c_(2k) = (-1)^(k+1) rho^(-2k)`.
//! - `x`: the linear branch.
//!
//! # Harmonics
//!
//! For `x = A cos(theta)`, `F(A cos(theta))` is periodic and even in theta:
//!
//! ```text
//!     F(A cos(theta)) = b_0(A)/2 + sum_{m>=1} b_m(A) cos(m theta)
//! ```
//!
//! The `b_m(A)` are known in closed form for the three functions
//! ([`StaticFn::harmonics_closed`]), from the classical expansion
//! `1/(p - A cos(theta)) = (1/W) [1 + 2 sum_m t^m cos(m theta)]`,
//! `W = sqrt(p^2 - A^2)`, `t = A/(p + W)`, and its integral. They are also
//! computed by an FFT over theta ([`StaticFn::harmonics`]), which works for
//! any function; the two agree to rounding (tests).
//!
//! The harmonics decrease geometrically with their rank, `|b_m| ~ exp(-m eta)`
//! with `eta = Re acosh(x_s / A)` for a singularity at `x_s`
//! ([`StaticFn::decay_rate`]): shifting the integration path of the Fourier
//! coefficient into the complex plane, `t -> t - i eta`, moves `A cos(t)` onto
//! an ellipse of foci `+-A`, and the path can be shifted until the ellipse
//! reaches the singularity. For a singularity off the real axis eta is
//! positive at every amplitude: the harmonics exist at every level, even
//! beyond rho, where only their expansion in powers of A diverges.
//!
//! # Alias-free simulation ([`alias_free_static`])
//!
//! Evaluated sample by sample, a static nonlinearity produces harmonics above
//! the Nyquist frequency, which fold back into the band. Here the
//! nonlinearity is never evaluated on the samples. The input is a sweep,
//! locally a sinusoid: its analytic signal `z = A exp(i psi)` gives the
//! amplitude and the phase at each sample, `x = A cos(psi)` exactly, and
//!
//! ```text
//!     F(x) = b_0(A)/2 + sum_m b_m(A) cos(m psi).
//! ```
//!
//! Each term is a sweep at m times the instantaneous frequency, generated
//! with its exact phase `m psi(t)` and weighted by the anti-aliasing filter of
//! a converter at that frequency ([`crate::spectral::aa_gate`]). Nothing is
//! generated above the Nyquist frequency, so nothing folds back, and nothing
//! is resampled, so nothing is delayed. Since A varies along the sweep
//! (fades), `b_m(A)` is tabulated once ([`HarmonicTable`]): at a fixed A by an
//! FFT over theta, then, as a function of A, by piecewise Chebyshev
//! interpolation of degree 15 on enough pieces for the interpolation error to
//! stay near rounding ([`StaticFn::pieces_for`]). In each piece only the
//! harmonics above a relative tolerance of 1e-17 are kept
//! ([`StaticFn::harmonic_count`]).
//!
//! # Exact harmonic responses ([`StaticSystem::harmonics`])
//!
//! For a sine of amplitude lambda (the sweep is a sine), branch j produces
//! `b_m^(j)(lambda) cos(m w t - m pi/2)` at the output frequency
//! `f = m w / 2 pi`. Written as `Im(H exp(i m w t))`, like the fundamental,
//!
//! ```text
//!     H_m(f, lambda) = i^(1-m) sum_j eps_j B_j(f) b_m^(j)(lambda),
//! ```
//!
//! at every level, beyond rho included.
//!
//! # System files
//!
//! `sim/systems/*.toml`, read by [`StaticSystem::load`]:
//!
//! ```toml
//! name = "static_arctan2.0"
//! fs = 96000.0
//! [[branch]]
//! kind = "arctan"               # linear | arctan | artanh | lorentz
//! rho = 2.0
//! minus_linear = true           # remove c_1 x: the branch starts at order 3
//! eps = 0.042
//! level_db = -50.0              # optional: intended level of the first nonlinear
//! at_hz = 1000.0                # order, re |G_1|, at this frequency (checked by a test)
//! filter = { highpass = [68.8, 4], peq = [[2415.4, 1.01, -4.76]], lowpass = [8695.6, 2] }
//! ```
//!
//! The filter sections are Butterworth (cutoff, even order) and peaking
//! equalizers (frequency, Q, gain in dB), applied in the order high-pass,
//! peaking, low-pass; no filter means `B = 1`.

use num_complex::Complex64 as C64;
use rayon::prelude::*;
use serde::Deserialize;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use crate::filters::{Sos, butterworth, peq, sosfilt};
use crate::mixing::mixing_matrix;
use crate::spectral::{Fft, aa_gate};

/// Neglected harmonics: their sum stays below TOL times the largest one.
pub const TOL: f64 = 1e-17;
/// Interpolation of b_m(A): minimum number of pieces on [0, A_max], and nodes per piece.
pub const PIECES: usize = 16;
pub const NODES: usize = 16;
/// Taylor terms kept to evaluate `F(x) - c_1 x` without cancellation (|x| < rho/2).
const SMALL_TERMS: usize = 64;

fn sign(k: usize) -> f64 {
    if k.is_multiple_of(2) { 1.0 } else { -1.0 }
}

fn binom(n: usize, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
}

/// Coefficient of `cos(m theta)` in `cos^n(theta)`, in the convention
/// `b_0/2 + sum b_m cos(m theta)` (twice the DC term for m = 0).
fn cos_power_coef(n: usize, m: usize) -> f64 {
    if m > n || !(n - m).is_multiple_of(2) {
        0.0
    } else {
        binom(n, (n - m) / 2) / 2f64.powi(n as i32 - 1)
    }
}

fn horner(c: &[f64], x: f64) -> f64 {
    c.iter().rev().fold(0.0, |acc, &v| acc * x + v)
}

/// Phase factor `i^(1-m)` of the m-th harmonic of a sine.
pub fn harmonic_phase(m: usize) -> C64 {
    match (m + 3) % 4 {
        0 => C64::new(1.0, 0.0),
        1 => C64::new(0.0, -1.0),
        2 => C64::new(-1.0, 0.0),
        _ => C64::new(0.0, 1.0),
    }
}

// -----------------------------------------------------------------------------
// Static functions
// -----------------------------------------------------------------------------

/// Shape of a static nonlinearity.
#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    /// `x`: the linear branch.
    Linear,
    /// `rho arctan(x/rho)`: singularities at `+-i rho`.
    Arctan { rho: f64 },
    /// `rho artanh(x/rho)`: same `|c_n|` as the arctan, singularities on the
    /// real axis at `+-rho`; defined for `|x| < rho` only.
    Artanh { rho: f64 },
    /// `(x/rho)^2 / (1 + (x/rho)^2)`: even, poles at `+-i rho`.
    Lorentz { rho: f64 },
}

/// Parity of F: odd harmonics only, or even harmonics only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parity {
    Odd,
    Even,
}

/// A static nonlinearity F(x), possibly without its linear term.
#[derive(Clone, Debug)]
pub struct StaticFn {
    pub shape: Shape,
    /// Remove `c_1 x`: the branch starts at order 2 (even) or 3 (odd).
    pub minus_linear: bool,
    /// c_0 ... c_64 (c_1 = 0 if `minus_linear`).
    small: Vec<f64>,
}

impl StaticFn {
    pub fn new(shape: Shape, minus_linear: bool) -> Self {
        match &shape {
            Shape::Arctan { rho } | Shape::Artanh { rho } | Shape::Lorentz { rho } => {
                assert!(*rho > 0.0, "rho must be > 0")
            }
            Shape::Linear => {}
        }
        let mut f = StaticFn {
            shape,
            minus_linear,
            small: Vec::new(),
        };
        f.small = f.taylor(SMALL_TERMS);
        f
    }

    /// Radius of convergence of the Taylor series (infinite for the linear branch).
    pub fn rho(&self) -> f64 {
        match &self.shape {
            Shape::Linear => f64::INFINITY,
            Shape::Arctan { rho } | Shape::Artanh { rho } | Shape::Lorentz { rho } => *rho,
        }
    }

    /// Nearest singularity in the closed upper half-plane (the others are its
    /// conjugate and their opposites); real for the artanh; None for the
    /// linear branch.
    pub fn singularity(&self) -> Option<C64> {
        match &self.shape {
            Shape::Linear => None,
            Shape::Arctan { rho } | Shape::Lorentz { rho } => Some(C64::new(0.0, *rho)),
            Shape::Artanh { rho } => Some(C64::new(*rho, 0.0)),
        }
    }

    pub fn parity(&self) -> Parity {
        match &self.shape {
            Shape::Linear | Shape::Arctan { .. } | Shape::Artanh { .. } => Parity::Odd,
            Shape::Lorentz { .. } => Parity::Even,
        }
    }

    /// Taylor coefficient c_n of the complete function (with its linear term).
    fn raw_coef(&self, n: usize) -> f64 {
        let odd = !n.is_multiple_of(2);
        match &self.shape {
            Shape::Linear => (n == 1) as u8 as f64,
            Shape::Arctan { rho } if odd => {
                let k = (n - 1) / 2;
                sign(k) * rho.powi(-2 * k as i32) / n as f64
            }
            Shape::Artanh { rho } if odd => rho.powi(1 - n as i32) / n as f64,
            Shape::Lorentz { rho } if !odd && n >= 2 => sign(n / 2 + 1) * rho.powi(-(n as i32)),
            _ => 0.0,
        }
    }

    /// Taylor coefficients c_0 ... c_{n_max} (c_1 = 0 if `minus_linear`).
    pub fn taylor(&self, n_max: usize) -> Vec<f64> {
        (0..=n_max)
            .map(|n| {
                if n == 1 && self.minus_linear {
                    0.0
                } else {
                    self.raw_coef(n)
                }
            })
            .collect()
    }

    /// F(x) of the complete function, written without cancellation.
    fn eval_raw(&self, x: f64) -> f64 {
        match &self.shape {
            Shape::Linear => x,
            Shape::Arctan { rho } => rho * (x / rho).atan(),
            Shape::Artanh { rho } => {
                assert!(
                    x.abs() < *rho,
                    "artanh: |x| = {} >= rho = {rho}, the system diverges",
                    x.abs()
                );
                rho * (x / rho).atanh()
            }
            Shape::Lorentz { rho } => {
                let q = (x / rho).powi(2);
                if q < 1.0 {
                    q / (1.0 + q)
                } else {
                    1.0 / (1.0 + 1.0 / q)
                }
            }
        }
    }

    /// F(x) (minus `c_1 x` if `minus_linear`: by the series for |x| < rho/2,
    /// where the subtraction would lose digits).
    pub fn eval(&self, x: f64) -> f64 {
        if self.minus_linear {
            if x.abs() < 0.5 * self.rho() {
                horner(&self.small, x)
            } else {
                self.eval_raw(x) - self.raw_coef(1) * x
            }
        } else {
            self.eval_raw(x)
        }
    }

    /// F(z) for a complex z (Cauchy integrals of the tests).
    pub fn eval_complex(&self, z: C64) -> C64 {
        let one = C64::new(1.0, 0.0);
        let v = match &self.shape {
            Shape::Linear => z,
            Shape::Arctan { rho } => rho * (z / rho).atan(),
            Shape::Artanh { rho } => rho * (z / rho).atanh(),
            Shape::Lorentz { rho } => {
                let q = (z / rho).powi(2);
                q / (one + q)
            }
        };
        if self.minus_linear {
            v - self.raw_coef(1) * z
        } else {
            v
        }
    }

    /// Decay rate eta(A) of the harmonics, `|b_m(A)| <~ C exp(-m eta)`:
    /// `eta = Re acosh(x_s / A) = acosh((|x_s/A - 1| + |x_s/A + 1|) / 2)`, the
    /// parameter of the ellipse of foci +-1 through `x_s / A`. Infinite for
    /// the linear branch.
    pub fn decay_rate(&self, a: f64) -> f64 {
        match self.singularity() {
            Some(xs) if a != 0.0 => {
                let w = xs / a.abs();
                (0.5 * ((w - 1.0).norm() + (w + 1.0).norm())).acosh()
            }
            _ => f64::INFINITY,
        }
    }

    /// Free room around the real segment [0, a_max] in the plane of complex
    /// amplitudes, for the interpolation of b_m(A): distance from the segment
    /// to the half-lines `A = mu x_s` (and its conjugate), mu real, |mu| >= 1.
    /// The analytic continuation of b_m(A) is singular only at their ends
    /// `A = +-x_s`, so this distance is a lower bound (more pieces than
    /// needed, never fewer). Evaluated on a grid of 1001 points.
    fn amplitude_singular_distance(&self, a_max: f64) -> f64 {
        let Some(xs) = self.singularity() else {
            return f64::INFINITY;
        };
        let (r, cos_a, sin_a) = (xs.norm(), xs.re / xs.norm(), xs.im / xs.norm());
        (0..=1000)
            .map(|i| {
                let a = a_max * i as f64 / 1000.0;
                if (a * cos_a / r).abs() >= 1.0 {
                    a * sin_a.abs()
                } else {
                    (a - xs).norm().min((a + xs).norm())
                }
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Number of pieces of the table on [0, a_max]: at least PIECES, and
    /// enough for each half-width h to stay below d/8 (d: distance to the
    /// singularities in A); the interpolation of degree 15 then converges like
    /// `(d/h + sqrt((d/h)^2 + 1))^(-15) ~ 16^(-15)` (Bernstein ellipse).
    pub fn pieces_for(&self, a_max: f64) -> usize {
        let need = (4.0 * a_max / self.amplitude_singular_distance(a_max)).ceil();
        (need.min(1024.0) as usize).max(PIECES)
    }

    /// A number of harmonics certainly sufficient, from eta alone (FFT size).
    fn m_estimate(&self, a: f64) -> usize {
        let r = self.decay_rate(a);
        if r.is_finite() {
            ((1e17f64.ln() + 10.0) / r).ceil().min(1e6) as usize + 8
        } else {
            self.small.len()
        }
    }

    /// Degree of the polynomial (linear branch); None otherwise.
    fn degree(&self) -> Option<usize> {
        match &self.shape {
            Shape::Linear => Some(self.small.iter().rposition(|&c| c != 0.0).unwrap_or(0)),
            _ => None,
        }
    }

    /// b_0 ... b_N by an FFT of size `fft.n` (N = fft.n / 2): `F(A cos(theta_k))`
    /// at `theta_k = 2 pi k / n` (symmetrized, the function being even in
    /// theta), then `b_m = (2/n) Re F_m`; the harmonics of the wrong parity
    /// are set to zero.
    fn fourier_cos(&self, a: f64, fft: &Fft) -> Vec<f64> {
        let n = fft.n;
        let mut s = vec![0.0; n];
        for k in 0..=n / 2 {
            let v = self.eval(a * (2.0 * PI * k as f64 / n as f64).cos());
            s[k] = v;
            if k > 0 && k < n - k {
                s[n - k] = v;
            }
        }
        let scale = 2.0 / n as f64;
        let mut h: Vec<f64> = fft.forward(&s).iter().map(|c| c.re * scale).collect();
        self.clean_parity(&mut h);
        h
    }

    fn clean_parity(&self, h: &mut [f64]) {
        let skip = match self.parity() {
            Parity::Odd => 0,
            Parity::Even => 1,
        };
        h.iter_mut().skip(skip).step_by(2).for_each(|v| *v = 0.0);
    }

    /// b_0 ... b_{m_max} of a polynomial: `b_m = sum_n c_n A^n` times the
    /// coefficient of `cos(m theta)` in `cos^n(theta)`.
    fn poly_harmonics(&self, a: f64, m_max: usize) -> Vec<f64> {
        (0..=m_max)
            .map(|m| {
                self.small
                    .iter()
                    .enumerate()
                    .skip(m)
                    .map(|(n, &c)| c * a.powi(n as i32) * cos_power_coef(n, m))
                    .sum()
            })
            .collect()
    }

    /// Harmonics b_0(A) ... b_{m_max}(A) (convention `b_0/2 + sum b_m cos(m theta)`):
    /// exact for the linear branch, by FFT otherwise.
    pub fn harmonics(&self, a: f64, m_max: usize) -> Vec<f64> {
        self.assert_defined(a);
        if self.degree().is_some() {
            return self.poly_harmonics(a, m_max);
        }
        let n = (2 * m_max.max(self.m_estimate(a)) + 64).next_power_of_two();
        let mut h = self.fourier_cos(a, &Fft::new(n));
        h.truncate(m_max + 1);
        h
    }

    /// Is `F(A cos(theta))` defined at this amplitude? (Real singularity of
    /// the artanh: only for `A < rho`.)
    pub fn defined(&self, a: f64) -> bool {
        match self.singularity() {
            Some(xs) if xs.im == 0.0 => a.abs() < xs.re.abs(),
            _ => true,
        }
    }

    pub fn assert_defined(&self, a: f64) {
        assert!(
            self.defined(a),
            "amplitude {} >= rho = {}: real singularity, the system diverges",
            a.abs(),
            self.rho()
        );
    }

    /// Harmonics in closed form. With `s = A / (rho + R)`:
    /// arctan, `R = sqrt(rho^2 + A^2)`: `b_(2k+1) = 2 rho (-1)^k s^(2k+1) / (2k+1)`;
    /// artanh, `R = sqrt(rho^2 - A^2)`: `b_(2k+1) = 2 rho s^(2k+1) / (2k+1)`;
    /// Lorentzian, `R = sqrt(rho^2 + A^2)`: `b_0 = 2 A^2 / (R (R + rho))`,
    /// `b_(2k) = 2 rho (-1)^(k+1) s^(2k) / R`.
    pub fn harmonics_closed(&self, a: f64, m_max: usize) -> Vec<f64> {
        // With minus_linear, b_1 - c_1 A is rewritten without subtraction.
        match &self.shape {
            Shape::Linear => self.poly_harmonics(a, m_max),
            Shape::Arctan { rho } => {
                let r = rho.hypot(a);
                let s = a / (rho + r);
                (0..=m_max)
                    .map(|m| match m {
                        // 2 rho s - A = A (rho - R) / (rho + R) = -A^3 / (rho + R)^2
                        1 if self.minus_linear => -a.powi(3) / (rho + r).powi(2),
                        _ if m % 2 == 1 => {
                            2.0 * rho * sign((m - 1) / 2) * s.powi(m as i32) / m as f64
                        }
                        _ => 0.0,
                    })
                    .collect()
            }
            Shape::Artanh { rho } => {
                self.assert_defined(a);
                let r = ((rho - a) * (rho + a)).sqrt();
                let s = a / (rho + r);
                (0..=m_max)
                    .map(|m| match m {
                        // 2 rho s - A = A (rho - R) / (rho + R) = A^3 / (rho + R)^2
                        1 if self.minus_linear => a.powi(3) / (rho + r).powi(2),
                        _ if m % 2 == 1 => 2.0 * rho * s.powi(m as i32) / m as f64,
                        _ => 0.0,
                    })
                    .collect()
            }
            Shape::Lorentz { rho } => {
                let r = rho.hypot(a);
                let s = a / (rho + r);
                (0..=m_max)
                    .map(|m| match m {
                        0 => 2.0 * a * a / (r * (r + rho)),
                        _ if m % 2 == 0 => 2.0 * rho * sign(m / 2 + 1) * s.powi(m as i32) / r,
                        _ => 0.0,
                    })
                    .collect()
            }
        }
    }

    /// Number of harmonics m_max(A) to keep: the tail `sum_{m > m_max} |b_m|`
    /// stays below TOL times `max_m |b_m|`. The b_m are computed by FFT; their
    /// geometric envelope `K exp(-eta m)` (K fitted on the b_m above rounding,
    /// doubled for safety) gives the tail.
    pub fn harmonic_count(&self, a: f64) -> usize {
        self.assert_defined(a);
        if let Some(d) = self.degree() {
            return d;
        }
        if a == 0.0 {
            return 0;
        }
        let r = self.decay_rate(a);
        let n = (2 * self.m_estimate(a) + 64).next_power_of_two();
        let h = self.fourier_cos(a, &Fft::new(n));
        let mag = |m: usize| if m == 0 { 0.5 * h[0].abs() } else { h[m].abs() };
        let top = (0..h.len()).map(mag).fold(0.0, f64::max);
        if top == 0.0 {
            return 0;
        }
        let mut ln_k = f64::NEG_INFINITY;
        let mut last = 0;
        for m in 1..h.len() {
            if mag(m) > 1e-12 * top {
                ln_k = ln_k.max(mag(m).ln() + r * m as f64);
                last = m;
            }
        }
        if last == 0 {
            return 0;
        }
        let ln_k = ln_k + 2f64.ln();
        let m = ((ln_k - (TOL * top * (1.0 - (-r).exp())).ln()) / r - 1.0).ceil();
        (m.max(last as f64) as usize).min(n / 2 - 32)
    }
}

// -----------------------------------------------------------------------------
// Table of harmonics and alias-free synthesis
// -----------------------------------------------------------------------------

/// b_m(A) on [0, A_max], interpolated piecewise: at least PIECES equal pieces
/// (see [`StaticFn::pieces_for`]), a Chebyshev polynomial of degree NODES - 1
/// per piece and per harmonic.
pub struct HarmonicTable {
    width: f64,
    m_max: usize,
    piece_m: Vec<usize>,
    /// Chebyshev coefficients: [piece][m in 0..=m_max][j in 0..NODES].
    coef: Vec<f64>,
    parity: Parity,
    at_zero: f64,
}

impl HarmonicTable {
    pub fn new(f: &StaticFn, a_max: f64) -> Self {
        f.assert_defined(a_max);
        let parity = f.parity();
        let at_zero = f.eval(0.0);
        if a_max.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return HarmonicTable {
                width: 0.0,
                m_max: 0,
                piece_m: Vec::new(),
                coef: Vec::new(),
                parity,
                at_zero,
            };
        }
        let pieces = f.pieces_for(a_max);
        let width = a_max / pieces as f64;
        // b_m decreases faster at small amplitudes: the top of the piece decides
        let piece_m: Vec<usize> = (0..pieces)
            .into_par_iter()
            .map(|p| f.harmonic_count(width * (p + 1) as f64))
            .collect();
        let m_max = piece_m.iter().copied().max().unwrap_or(0);
        let fft = Fft::new((2 * m_max + 64).next_power_of_two());
        // Chebyshev nodes x_k = cos((2k+1) pi / 2N) and T_j(x_k) = cos(j (2k+1) pi / 2N)
        let angle = |j: usize, k: usize| (j * (2 * k + 1)) as f64 * PI / (2 * NODES) as f64;
        let nodes: Vec<f64> = (0..NODES).map(|k| angle(1, k).cos()).collect();
        let cheb: Vec<Vec<f64>> = (0..NODES)
            .map(|j| (0..NODES).map(|k| angle(j, k).cos()).collect())
            .collect();
        let coef: Vec<f64> = (0..pieces)
            .into_par_iter()
            .map(|p| {
                let (mid, half) = ((p as f64 + 0.5) * width, 0.5 * width);
                let vals: Vec<Vec<f64>> = nodes
                    .iter()
                    .map(|&x| f.fourier_cos(mid + half * x, &fft))
                    .collect();
                let mut out = vec![0.0; (m_max + 1) * NODES];
                for m in 0..=m_max {
                    for (j, row) in cheb.iter().enumerate() {
                        let s: f64 = (0..NODES).map(|k| vals[k][m] * row[k]).sum();
                        let w = if j == 0 { 1.0 } else { 2.0 } / NODES as f64;
                        out[m * NODES + j] = s * w;
                    }
                }
                out
            })
            .collect::<Vec<Vec<f64>>>()
            .concat();
        HarmonicTable {
            width,
            m_max,
            piece_m,
            coef,
            parity,
            at_zero,
        }
    }

    /// Highest tabulated harmonic (at A_max).
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    /// Piece of A and Chebyshev polynomials T_0 ... T_{N-1} at A mapped to [-1, 1].
    fn locate(&self, a: f64) -> (usize, [f64; NODES]) {
        let p = ((a / self.width) as usize).min(self.piece_m.len() - 1);
        let s = ((a - (p as f64 + 0.5) * self.width) / (0.5 * self.width)).clamp(-1.0, 1.0);
        let mut t = [0.0; NODES];
        t[0] = 1.0;
        t[1] = s;
        for j in 2..NODES {
            t[j] = 2.0 * s * t[j - 1] - t[j - 2];
        }
        (p, t)
    }

    #[inline]
    fn coef_at(&self, p: usize, m: usize, t: &[f64; NODES]) -> f64 {
        let base = (p * (self.m_max + 1) + m) * NODES;
        self.coef[base..base + NODES]
            .iter()
            .zip(t)
            .map(|(c, t)| c * t)
            .sum()
    }

    /// Interpolated b_0(A) ... b_{m_max}(A).
    pub fn eval(&self, a: f64) -> Vec<f64> {
        if self.piece_m.is_empty() || a <= 0.0 {
            let mut h = vec![0.0; self.m_max + 1];
            h[0] = 2.0 * self.at_zero;
            return h;
        }
        let (p, t) = self.locate(a);
        (0..=self.m_max).map(|m| self.coef_at(p, m, &t)).collect()
    }

    /// One output sample: `b_0/2 + sum_m b_m(A) gate(m f) cos(m psi)`, with
    /// `A = |level| |z|` and `exp(i psi) = +-z / |z|` (sign of `level`); stops
    /// at the last harmonic of the piece or at the first `m f >= fs / 2`.
    #[inline]
    fn sample(&self, z: C64, level: f64, f: f64, fs: f64, pass: f64) -> f64 {
        let r = z.norm();
        let a = level.abs() * r;
        if a == 0.0 || self.piece_m.is_empty() {
            return self.at_zero;
        }
        let (p, t) = self.locate(a);
        // level < 0: level Re z = |level| Re(-z), the phase advances by pi
        let u = if level < 0.0 { -z / r } else { z / r };
        let mut sum = if self.parity == Parity::Odd {
            0.0
        } else {
            0.5 * self.coef_at(p, 0, &t)
        };
        let start = match self.parity {
            Parity::Odd => 1,
            Parity::Even => 2,
        };
        let ustep = u * u;
        let mut um = if start == 1 { u } else { u * u };
        let mut m = start;
        while m <= self.piece_m[p] {
            let fm = m as f64 * f;
            if fm >= 0.5 * fs {
                break;
            }
            sum += aa_gate(fm, fs, pass) * self.coef_at(p, m, &t) * um.re;
            um *= ustep;
            m += 2;
        }
        sum
    }
}

/// `F(level Re z)` without aliasing: `z` is the analytic signal of the input,
/// `f_inst` its instantaneous frequency (Hz) sample by sample; each harmonic
/// is weighted by `aa_gate(m f_inst, fs, pass)`. `level` may be negative
/// (polarity inversion).
pub fn alias_free_static(
    f: &StaticFn,
    z: &[C64],
    f_inst: &[f64],
    fs: f64,
    pass: f64,
    level: f64,
) -> Vec<f64> {
    assert_eq!(z.len(), f_inst.len(), "z and f_inst of different lengths");
    if f.shape == Shape::Linear {
        let k = if f.minus_linear { 0.0 } else { level };
        return z
            .iter()
            .zip(f_inst)
            .map(|(zk, &fk)| k * zk.re * aa_gate(fk, fs, pass))
            .collect();
    }
    let a_max = level.abs() * z.iter().map(|v| v.norm()).fold(0.0, f64::max);
    let table = HarmonicTable::new(f, a_max * (1.0 + 1e-12));
    z.par_iter()
        .zip(f_inst.par_iter())
        .map(|(&zk, &fk)| table.sample(zk, level, fk, fs, pass))
        .collect()
}

// -----------------------------------------------------------------------------
// System
// -----------------------------------------------------------------------------

/// Frequency response of a cascade of biquads at f (Hz):
/// `prod (b0 + b1 w + b2 w^2) / (a0 + a1 w + a2 w^2)`, `w = exp(-2 i pi f / fs)`.
///
/// Evaluated without cancellation. At low frequencies `w ~ 1` and the terms
/// of `b0 + b1 w + b2 w^2` nearly cancel (a high-pass is `b0 (1 - w)^2`
/// there): written as such, it would lose about 1e-11 in relative terms. The
/// polynomial is therefore expanded around `w = 1`, with
/// `v = w - 1 = -2 sin^2(theta/2) - i sin(theta)` computed directly:
/// `b0 + b1 w + b2 w^2 = (b0 + b1 + b2) + (b1 + 2 b2) v + b2 v^2`. Above
/// `fs / 4`, the same around `w = -1` (zeros of the low-pass filters at the
/// Nyquist frequency).
pub fn sos_response(sos: &[Sos], f: f64, fs: f64) -> C64 {
    let low = f <= 0.25 * fs;
    // d = w -+ 1; q(c) = c0 +- c1 + c2 + (c1 +- 2 c2) d + c2 d^2
    let (d, sg) = if low {
        let th = 2.0 * PI * f / fs;
        (C64::new(-2.0 * (0.5 * th).sin().powi(2), -th.sin()), 1.0)
    } else {
        let th = 2.0 * PI * (0.5 * fs - f) / fs;
        (C64::new(2.0 * (0.5 * th).sin().powi(2), -th.sin()), -1.0)
    };
    let q = |c0: f64, c1: f64, c2: f64| {
        let (k0, k1) = ((c0 + sg * c1) + c2, c1 + sg * 2.0 * c2);
        k0 + (k1 + c2 * d) * d
    };
    sos.iter().fold(C64::new(1.0, 0.0), |acc, s| {
        acc * q(s[0], s[1], s[2]) / q(s[3], s[4], s[5])
    })
}

/// One branch: `eps B[F(x)]`.
#[derive(Clone, Debug)]
pub struct StaticBranch {
    pub f: StaticFn,
    pub eps: f64,
    pub sos: Vec<Sos>,
    /// Intended level of the first nonlinear order (dB re |G_1|) and its frequency.
    pub nominal: Option<(f64, f64)>,
}

/// `y = sum_j eps_j B_j[F_j(x)]`, filters designed at `fs`.
#[derive(Clone, Debug)]
pub struct StaticSystem {
    pub name: String,
    pub fs: f64,
    pub branches: Vec<StaticBranch>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SystemToml {
    name: Option<String>,
    fs: f64,
    branch: Vec<BranchToml>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BranchToml {
    kind: String,
    rho: Option<f64>,
    #[serde(default)]
    minus_linear: bool,
    eps: f64,
    level_db: Option<f64>,
    at_hz: Option<f64>,
    #[serde(default)]
    filter: FilterToml,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct FilterToml {
    highpass: Option<(f64, usize)>,
    lowpass: Option<(f64, usize)>,
    peq: Vec<(f64, f64, f64)>,
}

impl StaticSystem {
    /// Directory of the system files of the repository (`sim/systems`).
    pub fn systems_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("systems")
    }

    /// A system of `sim/systems/` by its name (without `.toml`).
    pub fn standard(name: &str) -> Self {
        Self::load(&Self::systems_dir().join(format!("{name}.toml")))
    }

    pub fn load(path: &Path) -> Self {
        let text =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self::parse(&text, &stem)
    }

    /// Reads a TOML description (format in the module documentation);
    /// `default_name` is used if the file gives no `name`.
    pub fn parse(text: &str, default_name: &str) -> Self {
        let t: SystemToml = toml::from_str(text).unwrap_or_else(|e| panic!("{default_name}: {e}"));
        let branches = t
            .branch
            .into_iter()
            .map(|b| {
                let rho = || {
                    b.rho
                        .unwrap_or_else(|| panic!("branch {}: rho missing", b.kind))
                };
                let shape = match b.kind.as_str() {
                    "linear" => Shape::Linear,
                    "arctan" => Shape::Arctan { rho: rho() },
                    "artanh" => Shape::Artanh { rho: rho() },
                    "lorentz" => Shape::Lorentz { rho: rho() },
                    other => panic!("unknown branch kind: {other}"),
                };
                let fl = &b.filter;
                let mut sos = Vec::new();
                if let Some((fc, o)) = fl.highpass {
                    sos.extend(butterworth(o, fc, t.fs, true));
                }
                for &(fc, q, g) in &fl.peq {
                    sos.push(peq(fc, q, g, t.fs));
                }
                if let Some((fc, o)) = fl.lowpass {
                    sos.extend(butterworth(o, fc, t.fs, false));
                }
                let nominal = match (b.level_db, b.at_hz) {
                    (Some(l), Some(f)) => Some((l, f)),
                    (None, None) => None,
                    _ => panic!("level_db and at_hz go together"),
                };
                StaticBranch {
                    f: StaticFn::new(shape, b.minus_linear),
                    eps: b.eps,
                    sos,
                    nominal,
                }
            })
            .collect();
        StaticSystem {
            name: t.name.unwrap_or_else(|| default_name.to_string()),
            fs: t.fs,
            branches,
        }
    }

    /// Smallest radius of convergence of the branches.
    pub fn rho(&self) -> f64 {
        self.branches
            .iter()
            .map(|b| b.f.rho())
            .fold(f64::INFINITY, f64::min)
    }

    /// B_j(f) of each branch: `[branch][frequency]`.
    pub fn branch_responses(&self, freqs: &[f64]) -> Vec<Vec<C64>> {
        self.branches
            .iter()
            .map(|b| {
                freqs
                    .iter()
                    .map(|&f| sos_response(&b.sos, f, self.fs))
                    .collect()
            })
            .collect()
    }

    /// True kernels G_1 ... G_{n_max} at the given frequencies,
    /// `G_n(f) = sum_j eps_j c_n^(j) B_j(f)`: `[n - 1][frequency]`.
    pub fn kernels(&self, n_max: usize, freqs: &[f64]) -> Vec<Vec<C64>> {
        let w: Vec<Vec<f64>> = self
            .branches
            .iter()
            .map(|b| b.f.taylor(n_max).iter().map(|c| b.eps * c).collect())
            .collect();
        let l = self.branch_responses(freqs);
        (1..=n_max)
            .map(|n| {
                (0..freqs.len())
                    .map(|k| (0..self.branches.len()).map(|j| w[j][n] * l[j][k]).sum())
                    .collect()
            })
            .collect()
    }

    /// Level (dB re |G_1|) of the contribution of branch j at its first
    /// nonlinear order, at `at_hz`.
    pub fn branch_level_db(&self, j: usize, at_hz: f64) -> f64 {
        let b = &self.branches[j];
        let c = b.f.taylor(SMALL_TERMS);
        let n0 = (2..c.len())
            .find(|&n| c[n] != 0.0)
            .expect("branch without a nonlinear order");
        let g1 = self.kernels(1, &[at_hz])[0][0].norm();
        let l = sos_response(&b.sos, at_hz, self.fs).norm();
        20.0 * (b.eps * c[n0].abs() * l / g1).log10()
    }

    /// Output of the system for the input `level Re z` (`z`: analytic signal,
    /// `f_inst`: instantaneous frequency per sample):
    /// `sum_j eps_j B_j[F_j]`, each static branch synthesized without aliasing.
    pub fn respond(&self, z: &[C64], f_inst: &[f64], fs: f64, pass: f64, level: f64) -> Vec<f64> {
        assert!(
            (fs - self.fs).abs() <= 1e-9 * self.fs,
            "{}: filters designed at {} Hz, signal at {fs} Hz",
            self.name,
            self.fs
        );
        let parts: Vec<Vec<f64>> = (0..self.branches.len())
            .into_par_iter()
            .map(|j| {
                let b = &self.branches[j];
                let mut s = alias_free_static(&b.f, z, f_inst, fs, pass, level);
                s.iter_mut().for_each(|v| *v *= b.eps);
                sosfilt(&b.sos, &mut s);
                s
            })
            .collect();
        let mut y = vec![0.0; z.len()];
        for p in &parts {
            y.iter_mut().zip(p).for_each(|(y, v)| *y += v);
        }
        y
    }

    /// Exact steady-state harmonic responses: for a sine of amplitude lambda
    /// at `f / m`, the m-th harmonic at the OUTPUT frequency f is
    /// `H_m(f, lambda) = i^(1-m) sum_j eps_j B_j(f) b_m^(j)(lambda)`, with the
    /// `b_m` in closed form, at every lambda (beyond rho included). Result:
    /// `[m - 1][frequency]`, m = 1 ... m_max.
    pub fn harmonics(&self, m_max: usize, freqs: &[f64], lambda: f64) -> Vec<Vec<C64>> {
        let l = self.branch_responses(freqs);
        let a: Vec<Vec<f64>> = self
            .branches
            .iter()
            .map(|b| b.f.harmonics_closed(lambda, m_max))
            .collect();
        (1..=m_max)
            .map(|m| {
                let ph = harmonic_phase(m);
                (0..freqs.len())
                    .map(|k| {
                        ph * (0..self.branches.len())
                            .map(|j| self.branches[j].eps * a[j][m] * l[j][k])
                            .sum::<C64>()
                    })
                    .collect()
            })
            .collect()
    }

    /// Partial sum of the series of the cascade,
    /// `sum_{n <= n_max} i^(1-m) a_mn lambda^n G_n(f)`: equal to
    /// [`StaticSystem::harmonics`] inside the disc `lambda < rho` as n_max
    /// grows, divergent beyond. `[m - 1][frequency]`.
    pub fn series_harmonics(
        &self,
        m_max: usize,
        n_max: usize,
        freqs: &[f64],
        lambda: f64,
    ) -> Vec<Vec<C64>> {
        let g = self.kernels(n_max, freqs);
        let a = mixing_matrix(n_max);
        (1..=m_max)
            .map(|m| {
                (0..freqs.len())
                    .map(|k| {
                        (m..=n_max)
                            .map(|n| a[n - 1][m - 1] * lambda.powi(n as i32) * g[n - 1][k])
                            .sum()
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::{butterworth, peq, sosfilt};
    use crate::spectral::analytic;

    /// SplitMix64: a small reproducible generator for the tests.
    struct Rng(u64);

    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }
        fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.next_f64()
        }
    }

    /// Test functions defined at every amplitude.
    fn shapes() -> Vec<(&'static str, StaticFn)> {
        vec![
            (
                "arctan rho=2",
                StaticFn::new(Shape::Arctan { rho: 2.0 }, false),
            ),
            (
                "arctan-x rho=1.6",
                StaticFn::new(Shape::Arctan { rho: 1.6 }, true),
            ),
            (
                "lorentz rho=2",
                StaticFn::new(Shape::Lorentz { rho: 2.0 }, false),
            ),
            (
                "lorentz rho=1.3",
                StaticFn::new(Shape::Lorentz { rho: 1.3 }, false),
            ),
        ]
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    fn max_abs(a: &[f64]) -> f64 {
        a.iter().map(|v| v.abs()).fold(0.0, f64::max)
    }

    /// Small orders against the elementary series.
    #[test]
    fn taylor_small_orders() {
        for rho in [0.7, 1.3, 2.0, 2.5] {
            let at = StaticFn::new(Shape::Arctan { rho }, false).taylor(15);
            let ah = StaticFn::new(Shape::Artanh { rho }, false).taylor(15);
            let lo = StaticFn::new(Shape::Lorentz { rho }, false).taylor(15);
            for i in 0..8 {
                let (odd, even) = (2 * i + 1, 2 * i);
                let r2i = rho.powi(2 * i as i32);
                assert!((at[odd] * odd as f64 * r2i - sign(i)).abs() < 1e-14);
                assert!((ah[odd] * odd as f64 * r2i - 1.0).abs() < 1e-14);
                assert_eq!(at[even], 0.0);
                assert_eq!(ah[even], 0.0);
                assert_eq!(lo[odd], 0.0);
                if i > 0 {
                    assert!((lo[even] * r2i + sign(i)).abs() < 1e-14);
                }
            }
            let m = StaticFn::new(Shape::Arctan { rho }, true).taylor(5);
            assert_eq!(m[1], 0.0);
            assert_eq!(m[3], at[3]);
        }
    }

    /// Cauchy integral `c_n = (1/2 pi i) oint F(z) z^(-n-1) dz` on the circle
    /// `r = rho (1 - 1/(n+2))`, trapezoidal rule with N points.
    fn cauchy(f: &StaticFn, n: usize, big_n: usize) -> f64 {
        let r = f.rho() * (1.0 - 1.0 / (n + 2) as f64);
        let mut acc = C64::new(0.0, 0.0);
        for j in 0..big_n {
            let zj = C64::from_polar(r, 2.0 * PI * j as f64 / big_n as f64);
            let e = C64::from_polar(1.0, -2.0 * PI * ((n * j) % big_n) as f64 / big_n as f64);
            acc += f.eval_complex(zj) * e;
        }
        acc.re / big_n as f64 / r.powi(n as i32)
    }

    /// Large orders against the Cauchy integral, n <= 120. The error is
    /// referred to the envelope `rho^(-n) max_k |c_k| rho^k`.
    #[test]
    fn taylor_large_orders_cauchy() {
        let n_max = 120;
        let mut all = shapes();
        all.push((
            "artanh rho=2",
            StaticFn::new(Shape::Artanh { rho: 2.0 }, false),
        ));
        for (name, f) in all {
            let c = f.taylor(n_max);
            let rho = f.rho();
            let env = c
                .iter()
                .enumerate()
                .map(|(n, v)| v.abs() * rho.powi(n as i32))
                .fold(0.0, f64::max);
            let mut worst: f64 = 0.0;
            for (n, &cn) in c.iter().enumerate() {
                let want = cauchy(&f, n, 16_384);
                worst = worst.max((cn - want).abs() * rho.powi(n as i32) / env);
            }
            println!("{name:18} n <= {n_max}: error / envelope {worst:.1e}");
            assert!(worst < 1e-13, "{name}: {worst:e}");
        }
    }

    /// Harmonics by FFT against the closed forms, from A << rho to A = 8 rho,
    /// all harmonics up to beyond `harmonic_count`.
    #[test]
    fn harmonics_fft_vs_closed_forms() {
        for (name, f) in shapes() {
            let mut worst: f64 = 0.0;
            for a_rel in [0.02, 0.3, 0.9, 1.0, 1.5, 3.0, 8.0] {
                let a = a_rel * f.rho();
                let m_max = f.harmonic_count(a) + 20;
                let want = f.harmonics_closed(a, m_max);
                let got = f.harmonics(a, m_max);
                worst = worst.max(max_abs_diff(&got, &want) / max_abs(&want));
            }
            println!("{name:18} FFT vs closed form: max error / max|b_m| {worst:.1e}");
            assert!(worst < 2e-15, "{name}: {worst:e}");
        }
    }

    /// Inside the disc, `b_m(A) = sum_n c_n a_mn A^n`: the closed forms
    /// against the Taylor series summed up to n = 400.
    #[test]
    fn closed_forms_vs_series_inside_disc() {
        let mut all = shapes();
        all.push((
            "artanh rho=2",
            StaticFn::new(Shape::Artanh { rho: 2.0 }, false),
        ));
        for (name, f) in all {
            let c = f.taylor(400);
            let mut worst: f64 = 0.0;
            for a_rel in [0.05, 0.3, 0.6, 0.85] {
                let a = a_rel * f.rho();
                let m_max = 30;
                let got = f.harmonics_closed(a, m_max);
                let want: Vec<f64> = (0..=m_max)
                    .map(|m| {
                        (m..=400)
                            .rev()
                            .map(|n| c[n] * a.powi(n as i32) * cos_power_coef(n, m))
                            .sum()
                    })
                    .collect();
                worst = worst.max(max_abs_diff(&got, &want) / max_abs(&want));
            }
            println!("{name:18} closed form vs Taylor series (A <= 0.85 rho): {worst:.1e}");
            assert!(worst < 5e-15, "{name}: {worst:e}");
        }
    }

    /// The tail neglected by `harmonic_count` is below TOL (closed forms).
    #[test]
    fn neglected_tail() {
        for (name, f) in shapes() {
            for a_rel in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0] {
                let a = a_rel * f.rho();
                let m_max = f.harmonic_count(a);
                let c = f.harmonics_closed(a, 4 * m_max + 400);
                let tail: f64 = c[m_max + 1..].iter().map(|v| v.abs()).sum();
                assert!(
                    tail <= TOL * max_abs(&c),
                    "{name} A = {a}: tail {:.1e}",
                    tail / max_abs(&c)
                );
            }
        }
    }

    /// Chebyshev interpolation in A against the closed forms, A at random on [0, A_max].
    #[test]
    fn table_interpolation() {
        let mut rng = Rng(7);
        for (name, f) in shapes() {
            let a_max = 8.0;
            let table = HarmonicTable::new(&f, a_max);
            let top = max_abs(&f.harmonics_closed(a_max, table.m_max()));
            let mut worst: f64 = 0.0;
            for _ in 0..200 {
                let a = rng.uniform(0.0, a_max);
                let want = f.harmonics_closed(a, table.m_max());
                worst = worst.max(max_abs_diff(&table.eval(a), &want) / top);
            }
            println!(
                "{name:18} table A <= 8 ({} harmonics): max error / max|b_m(A_max)| {worst:.1e}",
                table.m_max()
            );
            assert!(worst < 5e-15, "{name}: {worst:e}");
        }
    }

    /// Without the anti-aliasing gate (f_inst = 0), the synthesis gives back
    /// F(x) sample by sample.
    #[test]
    fn synthesis_without_gate_is_pointwise() {
        let mut rng = Rng(11);
        let z: Vec<C64> = (0..20_000)
            .map(|i| {
                let a = if i == 0 { 1.0 } else { rng.next_f64() };
                C64::from_polar(a, rng.uniform(-PI, PI))
            })
            .collect();
        let f_inst = vec![0.0; z.len()];
        for (name, f) in shapes() {
            let mut worst: f64 = 0.0;
            for level in [0.3, 1.0, 4.0, 8.0, -1.0] {
                let y = alias_free_static(&f, &z, &f_inst, 96_000.0, 0.44, level);
                let want: Vec<f64> = z.iter().map(|v| f.eval(level * v.re)).collect();
                worst = worst.max(max_abs_diff(&y, &want) / max_abs(&want));
            }
            println!("{name:18} max |synthesis - F(x)| / max|F|: {worst:.1e}");
            assert!(worst < 5e-15, "{name}: {worst:e}");
        }
    }

    /// Exponential sweep with spectrally compact fades (a step smoothed by a
    /// Gaussian of standard deviation sigma, centred 7 sigma from the ends):
    /// the sweep stays locally sinusoidal from the first to the last sample.
    fn compact_sweep(fs: f64, n: usize, f1: f64, f2: f64, sigma: f64) -> (Vec<f64>, Vec<f64>) {
        let s = sigma * fs;
        let g: Vec<f64> = (0..n)
            .map(|k| (-0.5 * ((k as f64 - 7.0 * s) / s).powi(2)).exp())
            .collect();
        let total: f64 = (0..(14.0 * s) as usize).map(|k| g[k]).sum();
        let mut acc = 0.0;
        let rise: Vec<f64> = g
            .iter()
            .map(|v| {
                acc += v;
                (acc / total).min(1.0)
            })
            .collect();
        let l = n as f64 / fs / (f2 / f1).ln();
        let f_inst = (0..n).map(|i| f1 * (i as f64 / fs / l).exp()).collect();
        let x = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                rise[i] * rise[n - 1 - i] * (2.0 * PI * f1 * l * ((t / l).exp() - 1.0)).sin()
            })
            .collect();
        (x, f_inst)
    }

    /// Synthesis with the anti-aliasing gate against a reference oversampled
    /// 64 times: band-limited interpolation of the input, F point by point (no
    /// aliasing up to 32 fs), linear anti-aliasing filter of the same shape
    /// applied to the spectrum, back to fs; energy of the difference below
    /// 0.4 fs. A 4 s sweep from 100 Hz to 40 kHz, at 0 dB (A <= 1), +12 dB
    /// (A <= 4), and +12 dB after a peaking filter of +6 dB at 1 kHz (A <= 8).
    /// The naive sample-by-sample evaluation is reported for comparison.
    #[test]
    fn synthesis_matches_oversampling() {
        let (fs, n, r) = (96_000.0, 384_000, 64);
        let pass = 0.44;
        let (sweep, f_inst) = compact_sweep(fs, n, 100.0, 40_000.0, 0.02);
        let mut boosted = sweep.clone();
        sosfilt(&[peq(1000.0, 3.0, 6.0, fs)], &mut boosted);
        let funcs = [
            (
                "arctan-x rho=1.3",
                StaticFn::new(Shape::Arctan { rho: 1.3 }, true),
            ),
            (
                "lorentz rho=1.3",
                StaticFn::new(Shape::Lorentz { rho: 1.3 }, false),
            ),
        ];
        let small = Fft::new(n);
        let big = Fft::new(n * r);
        let band: Vec<usize> = (0..n / 2)
            .filter(|&k| (k as f64 * fs / n as f64) < 0.4 * fs)
            .collect();
        let err_db = |y: &[f64], s_ref: &[C64]| {
            let s = small.forward(y);
            let num: f64 = band.iter().map(|&k| (s[k] - s_ref[k]).norm_sqr()).sum();
            let den: f64 = band.iter().map(|&k| s_ref[k].norm_sqr()).sum();
            10.0 * (num / den).log10()
        };
        for (input, x, gains) in [
            ("sweep", &sweep, vec![0.0, 12.0]),
            ("peaking +6 dB", &boosted, vec![12.0]),
        ] {
            let z = analytic(x);
            let spec = small.forward(x);
            for gain in gains {
                let level = 10f64.powf(gain / 20.0);
                let a_max = level * z.iter().map(|v| v.norm()).fold(0.0, f64::max);
                let mut up = vec![C64::new(0.0, 0.0); n * r / 2 + 1];
                for (k, s) in spec.iter().enumerate().take(n / 2) {
                    up[k] = *s * (r as f64 * level);
                }
                let x_up = big.inverse(up);
                for (name, f) in &funcs {
                    let y_up: Vec<f64> = x_up.par_iter().map(|&v| f.eval(v)).collect();
                    let s_up = big.forward(&y_up);
                    drop(y_up);
                    let s_ref: Vec<C64> = (0..=n / 2)
                        .map(|k| s_up[k] * aa_gate(k as f64 * fs / n as f64, fs, pass) / r as f64)
                        .collect();
                    drop(s_up);
                    let y = alias_free_static(f, &z, &f_inst, fs, pass, level);
                    let naive: Vec<f64> = x.iter().map(|&v| f.eval(level * v)).collect();
                    let (e, e_naive) = (err_db(&y, &s_ref), err_db(&naive, &s_ref));
                    println!(
                        "{input:13} {gain:+3} dB (A <= {a_max:.2}) {name:18}: synthesis {e:7.1} dB, \
                         naive F(x) {e_naive:6.1} dB"
                    );
                    assert!(e < -180.0, "{input} {gain} dB {name}: {e} dB");
                }
            }
        }
    }

    /// `sos_response` against the exact modulus of a bilinear Butterworth
    /// filter pre-warped at fc: `|H|^2 = 1 / (1 + W^(2N))`,
    /// `W = tan(pi f / fs) / tan(pi fc / fs)` (low-pass) or its inverse
    /// (high-pass), from 1 Hz to 47.9 kHz; the direct formula
    /// `b0 + b1 w + b2 w^2` is reported for comparison.
    #[test]
    fn sos_response_matches_butterworth() {
        let fs = 96_000.0;
        let naive = |sos: &[Sos], f: f64| {
            let w = C64::from_polar(1.0, -2.0 * PI * f / fs);
            sos.iter().fold(C64::new(1.0, 0.0), |acc, s| {
                acc * (s[0] + s[1] * w + s[2] * w * w) / (s[3] + s[4] * w + s[5] * w * w)
            })
        };
        let (mut worst, mut worst_naive): (f64, f64) = (0.0, 0.0);
        for (order, fc, hp) in [
            (2, 500.0, true),
            (4, 500.0, true),
            (4, 3000.0, true),
            (2, 20_000.0, false),
            (4, 18_000.0, false),
            (4, 2000.0, false),
        ] {
            let sos = butterworth(order, fc, fs, hp);
            for f in [
                1.0, 5.0, 20.0, 50.0, 300.0, 1000.0, 10_000.0, 30_000.0, 45_000.0, 47_900.0,
            ] {
                let r = (PI * f / fs).tan() / (PI * fc / fs).tan();
                let omega = if hp { 1.0 / r } else { r };
                let want = 1.0 / (1.0 + omega.powi(2 * order as i32)).sqrt();
                worst = worst.max((sos_response(&sos, f, fs).norm() / want - 1.0).abs());
                worst_naive = worst_naive.max((naive(&sos, f).norm() / want - 1.0).abs());
            }
        }
        println!(
            "Butterworth, 1 Hz - 47.9 kHz: relative error of the modulus {worst:.1e} \
             (direct formula {worst_naive:.1e})"
        );
        assert!(
            worst < 1e-12 && worst_naive > 1e-9,
            "{worst:e} {worst_naive:e}"
        );
    }

    /// The test system: G_1 = B_1, and |G_2|, |G_3| at -50 dB re |G_1| at 1 kHz.
    #[test]
    fn test_system_levels() {
        let sys = StaticSystem::standard("static_arctan2.0");
        assert_eq!(sys.rho(), 2.0);
        assert_eq!(sys.branches.len(), 3);
        for (j, b) in sys.branches.iter().enumerate() {
            let Some((level, at)) = b.nominal else {
                continue;
            };
            let got = sys.branch_level_db(j, at);
            println!("branch {j} ({:?}): {got:.9} dB at {at} Hz", b.f.shape);
            assert!((got - level).abs() < 1e-6, "branch {j}: {got} dB");
        }
        let g = sys.kernels(3, &[1000.0]);
        let l1 = sos_response(&sys.branches[0].sos, 1000.0, sys.fs);
        assert!((g[0][0] - l1).norm() < 1e-15 * l1.norm(), "G_1 != B_1");
        for n in [2, 3] {
            let db = 20.0 * (g[n - 1][0].norm() / l1.norm()).log10();
            assert!((db + 50.0).abs() < 1e-6, "|G_{n}| = {db} dB");
        }
    }

    /// Inside the disc, `H_m(f, lambda) = sum_n i^(1-m) a_mn lambda^n G_n(f)`
    /// (sum up to n = 150, converged for lambda <= 0.7 rho).
    #[test]
    fn harmonics_match_series_inside_disc() {
        let freqs = [100.0, 1000.0, 5000.0, 15_000.0];
        let sys = StaticSystem::standard("static_arctan2.0");
        let rho = sys.rho();
        let mut worst: f64 = 0.0;
        for lam_rel in [0.1, 0.4, 0.7] {
            let lam = lam_rel * rho;
            let h = sys.harmonics(7, &freqs, lam);
            let s = sys.series_harmonics(7, 150, &freqs, lam);
            for k in 0..freqs.len() {
                let top = (0..7).map(|m| h[m][k].norm()).fold(0.0, f64::max);
                for m in 0..7 {
                    worst = worst.max((h[m][k] - s[m][k]).norm() / top);
                }
            }
        }
        println!("lambda <= 0.7 rho: |H_m - series| / max_m |H_m| {worst:.1e}");
        assert!(worst < 1e-14, "{worst:e}");
    }

    /// Beyond rho the partial sums of the series diverge while the exact
    /// harmonics stay finite, bounded by `sum_j eps_j |B_j| 2 max|F_j|`.
    #[test]
    fn beyond_rho_series_diverges_harmonics_stay_finite() {
        let sys = StaticSystem::standard("static_arctan2.0");
        let freqs = [1000.0];
        for lam in [3.0, 6.0, 12.0] {
            let h = sys.harmonics(5, &freqs, lam);
            let l = sys.branch_responses(&freqs);
            for (m, hm) in h.iter().enumerate() {
                let bound: f64 = sys
                    .branches
                    .iter()
                    .zip(&l)
                    .map(|(b, lj)| {
                        let fmax = (0..=200)
                            .map(|i| b.f.eval(lam * i as f64 / 200.0).abs())
                            .fold(0.0, f64::max);
                        b.eps * lj[0].norm() * 2.0 * fmax
                    })
                    .sum();
                assert!(hm[0].norm() <= bound, "m = {}", m + 1);
            }
            let s31 = sys.series_harmonics(5, 31, &freqs, lam)[2][0].norm();
            let s63 = sys.series_harmonics(5, 63, &freqs, lam)[2][0].norm();
            assert!(
                s63 > 1e3 * s31 && s31 > 10.0 * h[2][0].norm(),
                "lambda = {lam}"
            );
        }
    }

    /// Steady state in the time domain: a sine `lambda sin(2 pi 1000 t)`
    /// through the system (alias-free synthesis, filters included), last
    /// second analysed by a DFT (1000 exact periods), against the closed-form
    /// harmonic responses, below and beyond rho.
    #[test]
    fn steady_state_sine_matches_harmonics() {
        let fs = 96_000.0;
        let f0 = 1000.0;
        let per = 96;
        let n = 3 * 96_000;
        let z: Vec<C64> = (0..n)
            .map(|k| C64::from_polar(1.0, 2.0 * PI * (k % per) as f64 / per as f64 - PI / 2.0))
            .collect();
        let f_inst = vec![f0; n];
        let freqs: Vec<f64> = (1..=10).map(|m| m as f64 * f0).collect();
        let sys = StaticSystem::standard("static_arctan2.0");
        for lam in [0.5, 1.0, 3.0, 6.0] {
            let y = sys.respond(&z, &f_inst, fs, 0.44, lam);
            let h = sys.harmonics(10, &freqs, lam);
            let tail = &y[n - 96_000..];
            let mut worst: f64 = 0.0;
            for m in 1..=10 {
                let acc: C64 = tail
                    .iter()
                    .enumerate()
                    .map(|(k, &v)| {
                        let ph = 2.0 * PI * ((m * k) % per) as f64 / per as f64;
                        v * C64::from_polar(1.0, -ph)
                    })
                    .sum();
                let meas = C64::new(0.0, 2.0) * acc / tail.len() as f64;
                worst = worst.max((meas - h[m - 1][m - 1]).norm() / h[0][0].norm());
            }
            println!(
                "lambda = {lam} ({:.2} rho): max_m |DFT - H_m| / |H_1| = {worst:.1e}",
                lam / sys.rho()
            );
            assert!(worst < 1e-12, "lambda = {lam}: {worst:e}");
        }
    }

    /// The artanh mirrors the arctan: same |c_n| without the alternating
    /// signs, `F_+(x) = -i F_-(i x)`, same closed-form harmonics up to signs
    /// with `s` built on `sqrt(rho^2 - A^2)`, and FFT harmonics that agree.
    #[test]
    fn artanh_mirror_of_arctan() {
        let rho = 2.0;
        let at = StaticFn::new(Shape::Arctan { rho }, false);
        let ah = StaticFn::new(Shape::Artanh { rho }, false);
        let (ca, ch) = (at.taylor(41), ah.taylor(41));
        for n in 0..=41 {
            assert_eq!(ca[n].abs(), ch[n].abs(), "n = {n}");
            assert!(ch[n] >= 0.0);
        }
        for x in [0.1, 0.7, 1.5] {
            let lhs = at.eval(x);
            let rhs = (C64::new(0.0, -1.0) * ah.eval_complex(C64::new(0.0, x))).re;
            assert!((lhs - rhs).abs() < 1e-14);
        }
        for a in [0.2, 1.0, 1.9, 1.999] {
            let m_max = ah.harmonic_count(a).min(200);
            let got = ah.harmonics(a, m_max);
            let want = ah.harmonics_closed(a, m_max);
            let err = max_abs_diff(&got, &want) / max_abs(&want);
            println!("artanh A = {a}: FFT vs closed form {err:.1e}");
            assert!(err < 1e-12, "A = {a}: {err:e}");
        }
        assert_eq!(ah.singularity(), Some(C64::new(rho, 0.0)));
        assert_eq!(at.singularity(), Some(C64::new(0.0, rho)));
    }

    #[test]
    #[should_panic(expected = "real singularity")]
    fn artanh_refuses_amplitudes_beyond_rho() {
        StaticFn::new(Shape::Artanh { rho: 2.0 }, false).harmonics_closed(2.0, 7);
    }

    #[test]
    fn toml_format() {
        let sys = StaticSystem::parse(
            r#"
            fs = 48000.0
            [[branch]]
            kind = "linear"
            eps = 1.0
            [[branch]]
            kind = "artanh"
            rho = 3.0
            minus_linear = true
            eps = 0.1
            filter = { highpass = [100.0, 2], peq = [[1000.0, 2.0, 3.0]], lowpass = [8000.0, 4] }
            "#,
            "inline",
        );
        assert_eq!(sys.name, "inline");
        assert_eq!(sys.branches.len(), 2);
        assert!(sys.branches[0].sos.is_empty());
        assert_eq!(sys.branches[1].sos.len(), 1 + 1 + 2);
        assert_eq!(sys.rho(), 3.0);
        assert!(sys.branches[1].f.minus_linear);
    }
}
