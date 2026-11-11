//! Harmonic mixing of a Hammerstein cascade driven by a sinusoid, and the
//! single-level estimator that inverts its leading block.
//!
//! A power of a cosine is a sum of harmonics,
//! `cos^n(t) = a_0n + sum_{m=1..n} a_mn cos(m t)`, with
//! `a_mn = 2^(1-n) C(n, (n-m)/2)` for `n >= m >= 1`, `n = m (mod 2)`, and 0
//! otherwise. The branch of order n of the cascade, driven at amplitude
//! lambda, therefore contributes `a_mn lambda^n G_n(f)` to the m-th harmonic
//! response: `H_m(f, lambda) = sum_{n>=m} a_mn lambda^n G_n(f)`.

use num_complex::Complex64 as C64;

/// `[index][bin]`: one complex spectrum per harmonic or per order.
pub type Spectra = Vec<Vec<C64>>;

/// Coefficient `a_mn` of `cos(m t)` in `cos^n(t)` (m >= 1).
pub fn cos_coef(n: usize, m: usize) -> f64 {
    if m > n || !(n - m).is_multiple_of(2) {
        return 0.0;
    }
    let k = (n - m) / 2;
    (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64) / 2f64.powi(n as i32 - 1)
}

/// Mixing matrix for a *sine* sweep, as measured: entry `[n-1][m-1]` is
/// `i^(1-m) a_mn`. With a sine instead of a cosine the m-th harmonic carries
/// the extra phase `-m pi/2`, hence the factor `i^(1-m)`, which depends on m
/// only; it changes neither the kernels nor the moduli of the errors.
pub fn mixing_matrix(nmax: usize) -> Vec<Vec<C64>> {
    (1..=nmax)
        .map(|n| {
            (1..=nmax)
                .map(|m| C64::new(0.0, 1.0).powi(1 - m as i32) * cos_coef(n, m))
                .collect()
        })
        .collect()
}

/// Single-level estimator: the N harmonics measured at the identification
/// level are attributed to N kernels, `H_m = sum_{n<=N} a_mn G_n`, solved by
/// back substitution (the block is triangular). `h_ref[m-1][bin]`, returns
/// `g[n-1][bin]`.
pub fn single_level(h_ref: &Spectra, a: &[Vec<C64>]) -> Spectra {
    let m_max = h_ref.len();
    let f_len = h_ref[0].len();
    let mut g = vec![vec![C64::new(0.0, 0.0); f_len]; m_max];
    for m in (0..m_max).rev() {
        for f in 0..f_len {
            let mut s = h_ref[m][f];
            for n in m + 1..m_max {
                s -= a[n][m] * g[n][f];
            }
            g[m][f] = s / a[m][m];
        }
    }
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// cos^n(t) rebuilt from its harmonics at a few angles.
    #[test]
    fn cos_powers() {
        for n in 1..=12 {
            for t in [0.3f64, 1.1, 2.5] {
                let binom = (0..n / 2).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
                let dc = if n % 2 == 0 {
                    binom / 2f64.powi(n as i32)
                } else {
                    0.0
                };
                let s: f64 = (1..=n)
                    .map(|m| cos_coef(n, m) * (m as f64 * t).cos())
                    .sum::<f64>()
                    + dc;
                assert!((s - t.cos().powi(n as i32)).abs() < 1e-14, "n = {n}");
            }
        }
    }

    /// The estimator returns the kernels of a cascade of order N exactly.
    #[test]
    fn single_level_is_exact_for_order_n() {
        let n = 7;
        let a = mixing_matrix(n);
        let g: Spectra = (1..=n)
            .map(|k| vec![C64::new(1.0 / k as f64, 0.3 * k as f64)])
            .collect();
        let h: Spectra = (0..n)
            .map(|m| vec![(m..n).map(|k| a[k][m] * g[k][0]).sum()])
            .collect();
        let est = single_level(&h, &a);
        for k in 0..n {
            assert!((est[k][0] - g[k][0]).norm() < 1e-13 * g[k][0].norm());
        }
    }
}
